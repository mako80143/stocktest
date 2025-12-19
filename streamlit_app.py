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
st.set_page_config(page_title="DMS 自選模組回測", layout="wide")

# CSS 優化
st.markdown("""
<style>
    header {visibility: hidden;}
    .block-container {padding-top: 0rem !important; padding-bottom: 1rem !important;}
    .stApp {background-color: #0e1117;}
    input {font-weight: bold; color: #00e676 !important;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 數據下載 (含 VIX)
# ==========================================
@st.cache_data(ttl=3600)
def get_data(symbol, start):
    end = datetime.date.today()
    try:
        # 1. 下載個股數據
        df = yf.download(symbol, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty: return pd.DataFrame()
        df.index = df.index.tz_localize(None) # 去除時區

        # 2. 下載 VIX 數據 (不管用不用都先抓下來備用)
        vix_df = yf.download("^VIX", start=start, end=end, progress=False)
        if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.get_level_values(0)
        
        if not vix_df.empty:
            vix_df.index = vix_df.index.tz_localize(None)
            # 將 VIX 併入 df (使用 ffill 處理缺漏值)
            df['vix'] = vix_df['Close'].reindex(df.index).ffill()
        else:
            df['vix'] = 0 # 若抓不到 VIX 補 0

        return df
    except:
        return pd.DataFrame()

# ==========================================
# 3. Backtrader 資料結構 (擴充 VIX)
# ==========================================
class PandasDataPlus(bt.feeds.PandasData):
    lines = ('vix',)
    params = (('vix', -1),) # 自動對應 dataframe 中的 'vix' 欄位

# ==========================================
# 4. 策略核心：模組化過濾器
# ==========================================
class ModularStrategy(bt.Strategy):
    params = (('config', {}),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.vix = self.datas[0].vix
        self.c = self.params.config
        
        self.trade_list = []
        self.cash_history = []
        self.value_history = []
        
        # --- 指標初始化 (不管有無勾選，先算好備用) ---
        # 1. MA
        self.ma_short = bt.indicators.SMA(self.datas[0], period=int(self.c['ma_short_len']))
        self.ma_trend = bt.indicators.SMA(self.datas[0], period=int(self.c['ma_trend_len']))
        
        # 2. ROC
        self.roc = bt.indicators.ROC(self.datas[0], period=int(self.c['roc_len']))
        self.roc_ma = bt.indicators.SMA(self.roc, period=int(self.c['roc_ma_len']))
        
        # 3. ADX (Backtrader 內建包含 DI+, DI-)
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
        
        if len(self) < 100: return # 等數據穩定

        # ==========================
        # 核心邏輯：AND Gate (且)
        # 預設買入信號為 True，只要有一個啟用的條件不滿足，就變成 False
        # ==========================
        
        buy_signal = True
        sell_signal = False # 賣出通常是滿足任一條件

        # --- A. 進場檢核 (Buy Filters) ---
        
        # 1. VIX 濾網 (恐慌指數過高不買)
        if self.c['use_vix']:
            if self.vix[0] > self.c['vix_max']:
                buy_signal = False

        # 2. MA 趨勢濾網 (價格 > 長均線)
        if self.c['use_ma']:
            if self.dataclose[0] < self.ma_trend[0]:
                buy_signal = False

        # 3. ROC 動能濾網 (動能 > 平均動能)
        if self.c['use_roc']:
            if self.roc[0] < self.roc_ma[0]:
                buy_signal = False

        # 4. ADX 強度濾網 (ADX > 門檻 且 DI+ > DI-)
        if self.c['use_adx']:
            is_strong = (self.adx[0] > self.c['adx_thres'])
            is_bull = (self.di_plus[0] > self.di_minus[0])
            if not (is_strong and is_bull):
                buy_signal = False

        # --- B. 出場檢核 (Sell Triggers) ---
        # 滿足任一啟用的條件即賣出
        
        # 1. 跌破短均線 (止損)
        if self.c['use_ma'] and (self.dataclose[0] < self.ma_short[0]):
            sell_signal = True
            
        # 2. 動能與趨勢衰竭 (ADX轉折向下 且 ROC轉弱)
        if self.c['use_adx'] and self.c['use_roc']:
            adx_fading = (self.adx[-1] > self.c['adx_strong']) and (self.adx[0] < self.adx[-1])
            momentum_lost = (self.roc[0] < self.roc_ma[0])
            if adx_fading and momentum_lost:
                sell_signal = True
        
        # 3. VIX 過低 (過度貪婪)
        # (這裡設計一個選項：如果 VIX 低於某值是否要賣出)
        # 若您希望 VIX 只是買入濾網，可忽略此段，目前先不強制加入 VIX 賣出邏輯

        # --- C. 執行交易 ---
        
        if not self.position:
            if buy_signal:
                # 全倉買入 (保留緩衝)
                cash = self.broker.getcash()
                size = int((cash * 0.98) / self.dataclose[0]) # 0.98 保留現金給手續費
                if size > 0:
                    self.buy(size=size, info={'name': 'Entry'})
        else:
            if sell_signal:
                self.close(info={'name': 'Exit'})

# ==========================================
# 5. 控制台 (完全解除限制)
# ==========================================
st.sidebar.header("🎛️ 策略參數自訂")

# 1. 基礎設定
with st.sidebar.expander("1. 基礎設定 & 手續費", expanded=True):
    symbol = st.text_input("股票代碼", "NVDA")
    start_date = st.date_input("開始日期", datetime.date(2023, 1, 1))
    init_cash = st.number_input("初始本金", value=10000.0, format="%.2f")
    # 手續費輸入：0.1 代表 0.1%
    comm_pct = st.number_input("手續費 (%)", value=0.1, step=0.01, format="%.4f")

# 2. 指標選擇 (Checkbox + 無限制輸入)
st.sidebar.subheader("2. 指標模組 (勾選啟用)")

# VIX 模組
use_vix = st.sidebar.checkbox("啟用 VIX 濾網 (避險)", value=False)
vix_max = st.sidebar.number_input("VIX 買入上限 (高於此值不買)", value=30.0, step=0.1, format="%.2f", disabled=not use_vix)

# MA 模組
use_ma = st.sidebar.checkbox("啟用 MA 均線 (趨勢/止損)", value=True)
c1, c2 = st.sidebar.columns(2)
ma_short_len = c1.number_input("MA 短線 (止損)", value=20, step=1, disabled=not use_ma)
ma_trend_len = c2.number_input("MA 長線 (趨勢)", value=50, step=1, disabled=not use_ma)

# ROC 模組
use_roc = st.sidebar.checkbox("啟用 ROC 動能 (加速)", value=True)
c3, c4 = st.sidebar.columns(2)
roc_len = c3.number_input("ROC 週期", value=12, step=1, disabled=not use_roc)
roc_ma_len = c4.number_input("ROC 平均週期", value=6, step=1, disabled=not use_roc)

# ADX 模組
use_adx = st.sidebar.checkbox("啟用 ADX 強度 (濾掉盤整)", value=True)
c5, c6 = st.sidebar.columns(2)
adx_len = c5.number_input("ADX 週期", value=14, step=1, disabled=not use_adx)
adx_thres = c6.number_input("ADX 門檻 (低於此不買)", value=20.0, step=0.1, format="%.2f", disabled=not use_adx)
adx_strong = st.sidebar.number_input("ADX 強趨勢判斷值 (出場用)", value=25.0, step=0.1, format="%.2f", disabled=not use_adx)

# 參數包裝
config = {
    'use_vix': use_vix, 'vix_max': vix_max,
    'use_ma': use_ma, 'ma_short_len': ma_short_len, 'ma_trend_len': ma_trend_len,
    'use_roc': use_roc, 'roc_len': roc_len, 'roc_ma_len': roc_ma_len,
    'use_adx': use_adx, 'adx_len': adx_len, 'adx_thres': adx_thres, 'adx_strong': adx_strong
}

btn = st.sidebar.button("🔥 開始回測", type="primary")

# ==========================================
# 6. 主程式執行
# ==========================================
if btn:
    gc.collect()
    with st.spinner(f"正在分析 {symbol} ..."):
        # A. 數據下載
        df = get_data(symbol, start_date)
        if df.empty: st.error("無數據或代碼錯誤"); st.stop()

        # B. Backtrader 設定
        cerebro = bt.Cerebro()
        # 使用自訂的 DataFeed 以讀取 VIX
        data = PandasDataPlus(dataname=df)
        cerebro.adddata(data)
        
        cerebro.addstrategy(ModularStrategy, config=config)
        cerebro.broker.setcash(init_cash)
        
        # 手續費設定：使用者輸入 0.1 -> 0.001
        commission_val = comm_pct / 100.0
        cerebro.broker.setcommission(commission=commission_val) 
        
        results = cerebro.run()
        strat = results[0]
        final_val = cerebro.broker.getvalue()
        
        # C. 準備圖表資料
        dates = df.index[-len(strat.value_history):]
        eq_data = [{"time": d.strftime('%Y-%m-%d'), "value": v} for d, v in zip(dates, strat.value_history)]
        cash_data = [{"time": d.strftime('%Y-%m-%d'), "value": v} for d, v in zip(dates, strat.cash_history)]
        
        trade_log = pd.DataFrame(strat.trade_list)
        
        # Buy & Hold 比較基準
        initial_price = df['Close'].iloc[0]
        bh_val = (df['Close'].iloc[-1] / initial_price) * init_cash

    # === UI 呈現 ===
    st.title(f"🛠️ {symbol} 策略回測報告")

    # 1. 績效區塊
    ret_strat = ((final_val - init_cash) / init_cash) * 100
    ret_bh = ((bh_val - init_cash) / init_cash) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("策略最終權益", f"${final_val:,.0f}", delta=f"{ret_strat:.2f}%")
    col2.metric("Buy & Hold", f"${bh_val:,.0f}", delta=f"{ret_bh:.2f}%")
    col3.metric("手續費設定", f"{comm_pct}%")
    col4.metric("目前狀態", "空手 (現金)" if strat.cash_history[-1] > final_val*0.9 else "持有中")

    # 2. 權益曲線 & 現金水位
    st.subheader("📈 權益與現金水位")
    chart_options = {
        "chart": {"height": 350, "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"}},
        "series": [
            {"type": "Area", "data": eq_data, "options": {"lineColor": "#00E676", "topColor": "rgba(0, 230, 118, 0.3)", "bottomColor": "rgba(0, 230, 118, 0.0)", "title": "策略權益"}},
            {"type": "Area", "data": cash_data, "options": {"lineColor": "#2962FF", "topColor": "rgba(41, 98, 255, 0.3)", "bottomColor": "rgba(41, 98, 255, 0.0)", "title": "持有現金"}}
        ]
    }
    renderLightweightCharts([chart_options], key="main_chart")

    # 3. K線與進出場點
    st.subheader("🕯️ 詳細交易點位")
    
    # 計算繪圖用的指標 (僅視覺用)
    df['MA_Short'] = ta.sma(df['Close'], length=int(ma_short_len)) if use_ma else np.nan
    df['MA_Trend'] = ta.sma(df['Close'], length=int(ma_trend_len)) if use_ma else np.nan

    kline_data = [{"time": i.strftime('%Y-%m-%d'), "open": r['Open'], "high": r['High'], "low": r['Low'], "close": r['Close']} for i, r in df.iterrows()]
    
    series_list = [{"type": 'Candlestick', "data": kline_data, "options": {"upColor": '#089981', "downColor": '#f23645', "borderVisible": False}}]
    
    if use_ma:
        ma1_data = [{"time": i.strftime('%Y-%m-%d'), "value": v} for i, v in df['MA_Short'].items() if not pd.isna(v)]
        ma2_data = [{"time": i.strftime('%Y-%m-%d'), "value": v} for i, v in df['MA_Trend'].items() if not pd.isna(v)]
        series_list.append({"type": "Line", "data": ma1_data, "options": {"color": "#FF5252", "lineWidth": 1, "title": "止損均線"}})
        series_list.append({"type": "Line", "data": ma2_data, "options": {"color": "#2962FF", "lineWidth": 2, "title": "趨勢均線"}})

    if not trade_log.empty:
        markers = []
        for _, t in trade_log.iterrows():
            is_buy = t['Type'] == 'Buy'
            markers.append({
                "time": t['Date'].strftime('%Y-%m-%d'),
                "position": "belowBar" if is_buy else "aboveBar",
                "color": "#00E676" if is_buy else "#FF5252",
                "shape": "arrowUp" if is_buy else "arrowDown",
                "text": t['Reason']
            })
        series_list[0]["markers"] = markers

    renderLightweightCharts([{"chart": {"height": 500, "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"}}, "series": series_list}], key="k_chart")

    # 4. 交易明細
    if not trade_log.empty:
        st.subheader("📋 交易紀錄")
        
        # 簡單的樣式處理
        def color_trade(val):
            color = '#00E676' if val == 'Buy' else '#FF5252'
            return f'color: {color}; font-weight: bold'

        trade_view = trade_log.copy()
        trade_view['Date'] = trade_view['Date'].dt.strftime('%Y-%m-%d')
        trade_view['Price'] = trade_view['Price'].map('{:,.2f}'.format)
        trade_view['Value'] = trade_view['Value'].map('{:,.0f}'.format)
        
        st.dataframe(trade_view.style.applymap(color_trade, subset=['Type']), use_container_width=True)
    else:
        st.info("⚠️ 在此設定下，沒有觸發任何交易 (可能是條件太嚴格，或 VIX 濾掉了所有機會)。")
