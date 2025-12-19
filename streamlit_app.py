import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import backtrader as bt
import datetime
from streamlit_lightweight_charts import renderLightweightCharts
import collections.abc
import numpy as np
import gc
import warnings

# --- 1. 系統設定 ---
warnings.filterwarnings("ignore")
collections.Iterable = collections.abc.Iterable
st.set_page_config(page_title="VIX 皇權策略回測", layout="wide")

st.markdown("""
<style>
    header {visibility: hidden;}
    .block-container {padding-top: 0rem !important; padding-bottom: 1rem !important;}
    .stApp {background-color: #0e1117;}
    input {font-weight: bold; color: #00e676 !important;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 數據下載
# ==========================================
@st.cache_data(ttl=3600)
def get_data(symbol, start):
    end = datetime.date.today()
    try:
        # 下載個股
        df = yf.download(symbol, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty: return pd.DataFrame()
        df.index = df.index.tz_localize(None)

        # 下載 VIX
        vix_df = yf.download("^VIX", start=start, end=end, progress=False)
        if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.get_level_values(0)
        
        if not vix_df.empty:
            vix_df.index = vix_df.index.tz_localize(None)
            df['vix'] = vix_df['Close'].reindex(df.index).ffill()
        else:
            df['vix'] = 0 

        return df
    except:
        return pd.DataFrame()

# ==========================================
# 3. Backtrader VIX 資料結構
# ==========================================
class PandasDataPlus(bt.feeds.PandasData):
    lines = ('vix',)
    params = (('vix', -1),)

# ==========================================
# 4. 策略核心：VIX 皇權 + 邏輯切換
# ==========================================
class VixSovereignStrategy(bt.Strategy):
    params = (('config', {}),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.vix = self.datas[0].vix
        self.c = self.params.config
        
        self.trade_list = []
        self.cash_history = []
        self.value_history = []
        
        # --- 指標計算 ---
        # 1. MA
        self.ma_short = bt.indicators.SMA(self.datas[0], period=int(self.c['ma_short_len']))
        self.ma_trend = bt.indicators.SMA(self.datas[0], period=int(self.c['ma_trend_len']))
        
        # 2. ROC
        self.roc = bt.indicators.ROC(self.datas[0], period=int(self.c['roc_len']))
        self.roc_ma = bt.indicators.SMA(self.roc, period=int(self.c['roc_ma_len']))
        
        # 3. ADX
        self.adx = bt.indicators.ADX(self.datas[0], period=int(self.c['adx_len']))
        self.di_plus = bt.indicators.PlusDI(self.datas[0], period=int(self.c['adx_len']))
        self.di_minus = bt.indicators.MinusDI(self.datas[0], period=int(self.c['adx_len']))

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.trade_list.append({
                'Date': bt.num2date(order.executed.dt),
                'Type': 'Buy' if order.isbuy() else 'Sell',
                'Price': order.executed.price,
                'Value': order.executed.value,
                'Reason': getattr(order.info, 'name', 'Signal')
            })

    def next(self):
        self.cash_history.append(self.broker.getcash())
        self.value_history.append(self.broker.getvalue())
        
        if len(self) < 100: return

        # =========================================
        # 第一層：VIX 皇權 (絕對命令)
        # =========================================
        force_buy = False
        force_sell = False
        
        # VIX 強制買入 (恐慌撿便宜)
        if self.c['use_vix_force'] and self.vix[0] >= self.c['vix_force_buy']:
            force_buy = True
            
        # VIX 強制賣出 (貪婪逃頂)
        if self.c['use_vix_force'] and self.vix[0] <= self.c['vix_force_sell']:
            force_sell = True

        # 如果觸發皇權，直接執行並跳出
        if force_buy:
            if not self.position:
                cash = self.broker.getcash()
                size = int((cash * 0.98) / self.dataclose[0])
                if size > 0: self.buy(size=size, info={'name': 'VIX_Panic_Buy'})
            return # 強制買入後，不看其他指標

        if force_sell:
            if self.position:
                self.close(info={'name': 'VIX_Greed_Sell'})
            return # 強制賣出後，不看其他指標

        # =========================================
        # 第二層：複合指標判斷 (當 VIX 正常時)
        # =========================================
        
        # 1. 收集各個指標的訊號 (True/False)
        sig_ma_buy = (self.dataclose[0] > self.ma_trend[0]) if self.c['use_ma'] else False
        sig_ma_sell = (self.dataclose[0] < self.ma_short[0]) if self.c['use_ma'] else False
        
        sig_roc_buy = (self.roc[0] > self.roc_ma[0]) if self.c['use_roc'] else False
        sig_roc_sell = (self.roc[0] < self.roc_ma[0]) if self.c['use_roc'] else False
        
        sig_adx_buy = (self.adx[0] > self.c['adx_thres'] and self.di_plus[0] > self.di_minus[0]) if self.c['use_adx'] else False
        # ADX 賣出條件：高檔轉折且死叉
        sig_adx_sell = (self.adx[-1] > self.c['adx_strong'] and self.adx[0] < self.adx[-1]) if self.c['use_adx'] else False

        # 2. 根據「邏輯模式」整合訊號
        final_buy = False
        final_sell = False
        mode = self.c['logic_mode']

        # 計算啟用的指標數量 (分母)
        active_indicators = sum([self.c['use_ma'], self.c['use_roc'], self.c['use_adx']])
        if active_indicators == 0: active_indicators = 1 # 避免除以0

        if mode == "嚴格共識 (AND)":
            # 必須「所有啟用」的指標都說 Buy
            conditions = []
            if self.c['use_ma']: conditions.append(sig_ma_buy)
            if self.c['use_roc']: conditions.append(sig_roc_buy)
            if self.c['use_adx']: conditions.append(sig_adx_buy)
            
            # 如果 conditions 為空(都沒勾選)，則不買
            if conditions and all(conditions):
                final_buy = True
                
            # 賣出通常只要觸發一個止損即可 (OR)
            if sig_ma_sell or sig_roc_sell or sig_adx_sell:
                final_sell = True

        elif mode == "寬鬆投票 (OR)":
            # 只要「任一啟用」的指標說 Buy
            if sig_ma_buy or sig_roc_buy or sig_adx_buy:
                final_buy = True
            if sig_ma_sell or sig_roc_sell or sig_adx_sell:
                final_sell = True

        elif mode == "僅 VIX (Only)":
            # 不做任何事，因為 VIX 皇權在上面已經處理過了
            pass

        # =========================================
        # 第三層：執行交易
        # =========================================
        if not self.position:
            if final_buy:
                cash = self.broker.getcash()
                size = int((cash * 0.98) / self.dataclose[0])
                if size > 0: self.buy(size=size, info={'name': 'Signal_Buy'})
        else:
            if final_sell:
                self.close(info={'name': 'Signal_Exit'})

# ==========================================
# 5. 控制台
# ==========================================
st.sidebar.header("👑 VIX 皇權回測系統")

# A. 基礎
with st.sidebar.expander("1. 基礎與手續費", expanded=True):
    symbol = st.text_input("股票代碼", "NVDA")
    start_date = st.date_input("回測開始", datetime.date(2023, 1, 1))
    init_cash = st.number_input("本金", 10000.0)
    comm_pct = st.number_input("手續費 (%)", 0.1, step=0.01)

# B. VIX 皇權設定
st.sidebar.subheader("2. VIX 皇權 (最高優先級)")
use_vix_force = st.sidebar.checkbox("啟用 VIX 強制買賣", value=True)
c1, c2 = st.sidebar.columns(2)
vix_force_buy = c1.number_input("VIX > 強制買入 (Panic)", value=30.0, step=0.5, disabled=not use_vix_force)
vix_force_sell = c2.number_input("VIX < 強制賣出 (Greed)", value=13.0, step=0.5, disabled=not use_vix_force)

# C. 邏輯模式選擇 (關鍵!)
st.sidebar.subheader("3. 複合指標邏輯")
logic_mode = st.sidebar.selectbox(
    "多指標達成條件", 
    ["嚴格共識 (AND)", "寬鬆投票 (OR)", "僅 VIX (Only)"],
    help="嚴格共識: 所有勾選指標都要符合才買。\n寬鬆投票: 任一指標符合就買。\n僅 VIX: 完全忽略下方指標。"
)

# D. 其他指標參數 (無限制輸入)
with st.sidebar.expander("4. 輔助指標參數 (自由輸入)", expanded=True):
    st.caption("數值無上下限，請輸入您想測試的數字")
    
    use_ma = st.checkbox("啟用 MA", True)
    ma_short_len = st.number_input("MA 短線 (止損)", value=20.0, disabled=not use_ma)
    ma_trend_len = st.number_input("MA 長線 (趨勢)", value=50.0, disabled=not use_ma)
    
    use_roc = st.checkbox("啟用 ROC", True)
    roc_len = st.number_input("ROC 週期", value=12.0, disabled=not use_roc)
    roc_ma_len = st.number_input("ROC MA 週期", value=6.0, disabled=not use_roc)
    
    use_adx = st.checkbox("啟用 ADX", True)
    adx_len = st.number_input("ADX 週期", value=14.0, disabled=not use_adx)
    adx_thres = st.number_input("ADX 買入門檻", value=20.0, disabled=not use_adx)
    adx_strong = st.number_input("ADX 高檔轉折點", value=25.0, disabled=not use_adx)

config = {
    'use_vix_force': use_vix_force, 'vix_force_buy': vix_force_buy, 'vix_force_sell': vix_force_sell,
    'logic_mode': logic_mode,
    'use_ma': use_ma, 'ma_short_len': ma_short_len, 'ma_trend_len': ma_trend_len,
    'use_roc': use_roc, 'roc_len': roc_len, 'roc_ma_len': roc_ma_len,
    'use_adx': use_adx, 'adx_len': adx_len, 'adx_thres': adx_thres, 'adx_strong': adx_strong
}

btn = st.sidebar.button("🚀 執行策略", type="primary")

# ==========================================
# 6. 主程式
# ==========================================
if btn:
    gc.collect()
    with st.spinner("正在計算..."):
        df = get_data(symbol, start_date)
        if df.empty: st.error("無數據"); st.stop()

        cerebro = bt.Cerebro()
        cerebro.adddata(PandasDataPlus(dataname=df))
        cerebro.addstrategy(VixSovereignStrategy, config=config)
        cerebro.broker.setcash(init_cash)
        cerebro.broker.setcommission(commission=comm_pct/100.0)
        
        results = cerebro.run()
        strat = results[0]
        final_val = cerebro.broker.getvalue()
        
        # 數據處理
        dates = df.index[-len(strat.value_history):]
        eq_data = [{"time": d.strftime('%Y-%m-%d'), "value": v} for d, v in zip(dates, strat.value_history)]
        cash_data = [{"time": d.strftime('%Y-%m-%d'), "value": v} for d, v in zip(dates, strat.cash_history)]
        trade_log = pd.DataFrame(strat.trade_list)
        
        # Buy & Hold
        bh_val = (df['Close'].iloc[-1] / df['Close'].iloc[0]) * init_cash

    # UI 顯示
    st.title(f"👑 {symbol} 策略戰報")
    st.caption(f"模式：{logic_mode} | VIX 強制買入 > {vix_force_buy} | VIX 強制賣出 < {vix_force_sell}")

    # 1. 績效
    c1, c2, c3 = st.columns(3)
    c1.metric("最終權益", f"${final_val:,.0f}", delta=f"{((final_val-init_cash)/init_cash)*100:.1f}%")
    c2.metric("Buy & Hold", f"${bh_val:,.0f}", delta=f"{((bh_val-init_cash)/init_cash)*100:.1f}%")
    c3.metric("手續費", f"{comm_pct}%")

    # 2. 圖表
    st.subheader("📈 資產曲線")
    chart_opts = {
        "chart": {"height": 300, "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"}},
        "series": [
            {"type": "Area", "data": eq_data, "options": {"lineColor": "#00E676", "topColor": "rgba(0,230,118,0.3)", "bottomColor": "rgba(0,0,0,0)", "title": "總權益"}},
            {"type": "Area", "data": cash_data, "options": {"lineColor": "#2962FF", "topColor": "rgba(41,98,255,0.3)", "bottomColor": "rgba(0,0,0,0)", "title": "現金水位"}}
        ]
    }
    renderLightweightCharts([chart_opts], key="main")

    # 3. 交易明細 (含原因)
    if not trade_log.empty:
        st.subheader("📋 交易紀錄")
        
        # 標記 VIX 觸發的特殊交易
        def highlight_vix(val):
            color = 'white'
            if 'VIX' in str(val): color = '#FFD700' # 金色代表皇權觸發
            return f'color: {color}'
        
        display_df = trade_log.copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        display_df['Price'] = display_df['Price'].map('{:,.2f}'.format)
        display_df['Value'] = display_df['Value'].map('{:,.0f}'.format)
        
        st.dataframe(display_df.style.applymap(highlight_vix, subset=['Reason']), use_container_width=True)
    else:
        st.warning("無交易產生。請檢查 VIX 條件是否太嚴苛 (例如 VIX 買入設太高)。")

    # 4. K線驗證
    st.subheader("🕯️ 訊號還原")
    kline_data = [{"time": i.strftime('%Y-%m-%d'), "open": r['Open'], "high": r['High'], "low": r['Low'], "close": r['Close']} for i, r in df.iterrows()]
    series = [{"type": 'Candlestick', "data": kline_data, "options": {"upColor": '#089981', "downColor": '#f23645'}}]
    
    if not trade_log.empty:
        markers = []
        for _, t in trade_log.iterrows():
            is_buy = t['Type'] == 'Buy'
            txt = "V" if "VIX" in t['Reason'] else "S" # V代表VIX觸發, S代表Signal
            markers.append({
                "time": t['Date'].strftime('%Y-%m-%d'),
                "position": "belowBar" if is_buy else "aboveBar",
                "color": "#FFD700" if "VIX" in t['Reason'] else ("#00E676" if is_buy else "#FF5252"),
                "shape": "arrowUp" if is_buy else "arrowDown",
                "text": txt
            })
        series[0]["markers"] = markers
    
    renderLightweightCharts([{"chart": {"height": 400, "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"}}, "series": series}], key="candle")
