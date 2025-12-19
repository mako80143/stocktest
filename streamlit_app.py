import streamlit as st
import yfinance as yf
import pandas as pd
import backtrader as bt
import datetime
from streamlit_lightweight_charts import renderLightweightCharts
import collections.abc
import numpy as np
import gc
import warnings
from scipy.signal import argrelextrema

# --- 1. 系統設定 ---
warnings.filterwarnings("ignore")
collections.Iterable = collections.abc.Iterable
st.set_page_config(page_title="VIX 階梯全參數修正版", layout="wide")

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
        df = yf.download(symbol, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty: return pd.DataFrame()
        df.index = df.index.tz_localize(None)

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

# 上帝視角
def calculate_god_mode(df, init_cash):
    data = df['Close'].values
    min_idx = argrelextrema(data, np.less, order=5)[0]
    max_idx = argrelextrema(data, np.greater, order=5)[0]
    cash = init_cash; shares = 0; god_curve = []
    for i in range(len(df)):
        price = data[i]
        if i in min_idx and cash > 0: shares = cash/price; cash = 0
        elif i in max_idx and shares > 0: cash = shares*price; shares = 0
        god_curve.append({"time": df.index[i].strftime('%Y-%m-%d'), "value": cash + (shares * price)})
    return god_curve

# ==========================================
# 3. Backtrader 策略 (Bug 修復版)
# ==========================================
class PandasDataPlus(bt.feeds.PandasData):
    lines = ('vix',)
    params = (('vix', -1),)

class FullyConfigurableStrategy(bt.Strategy):
    params = (('config', {}),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.vix = self.datas[0].vix
        self.c = self.params.config
        
        self.trade_list = []
        self.cash_history = []
        self.value_history = []
        
        # 狀態字典：確保每個條件獨立運作，互不干擾
        self.states = {
            'vix_1': False, 'vix_2': False, 'vix_3': False,
            'ma': False, 'roc': False, 'adx': False
        }

        # 指標初始化
        self.ma = bt.indicators.SMA(self.datas[0], period=int(self.c['ma_len']))
        self.roc = bt.indicators.ROC(self.datas[0], period=int(self.c['roc_len']))
        self.adx = bt.indicators.ADX(self.datas[0], period=int(self.c['adx_len']))

    def notify_order(self, order):
        if order.status in [order.Completed]:
            # 【重要修正】正確讀取交易原因，避免 TypeError
            reason = order.info.get('name', 'Signal') if isinstance(order.info, dict) else 'Signal'
            
            self.trade_list.append({
                'Date': bt.num2date(order.executed.dt),
                'Type': 'Buy' if order.isbuy() else 'Sell',
                'Price': order.executed.price,
                'Value': order.executed.value,
                'Size': order.executed.size,
                'Reason': reason
            })

    def next(self):
        self.cash_history.append(self.broker.getcash())
        self.value_history.append(self.broker.getvalue())
        if len(self) < 50: return

        # 資產基數
        portfolio_val = self.broker.getvalue()
        cash = self.broker.getcash()

        # =============================================
        # 1. VIX 階梯加碼邏輯 (3 Levels)
        # =============================================
        for i in range(1, 4):
            key = f'vix_{i}'
            if not self.c[f'vix_en_{i}']: continue

            # 嚴格讀取變數，確保買賣不混用
            buy_threshold = self.c[f'vix_b_trig_{i}']
            sell_threshold = self.c[f'vix_s_trig_{i}']
            buy_pct = self.c[f'vix_b_pct_{i}']
            sell_pct = self.c[f'vix_s_pct_{i}']

            # Buy: VIX > 設定值 且 尚未買入
            if self.vix[0] > buy_threshold and not self.states[key]:
                amt = portfolio_val * (buy_pct / 100.0)
                if cash >= amt and amt > 0:
                    self.buy(size=int(amt/self.dataclose[0]), info={'name': f'VIX_Lv{i}_Buy'})
                    self.states[key] = True

            # Sell: VIX < 設定值 (必須比買入值低) 且 已經買入
            elif self.vix[0] < sell_threshold and self.states[key]:
                if self.position.size > 0:
                    size_sell = int(self.position.size * (sell_pct / 100.0))
                    if size_sell > 0:
                        self.sell(size=size_sell, info={'name': f'VIX_Lv{i}_Sell'})
                    self.states[key] = False

        # =============================================
        # 2. MA 策略 (乖離率概念)
        # =============================================
        if self.c['use_ma']:
            ma_val = self.ma[0]
            # Diff% = (Price - MA) / MA * 100
            diff_pct = ((self.dataclose[0] - ma_val) / ma_val) * 100
            
            # 買入：乖離率 > 設定值
            if diff_pct > self.c['ma_b_trig'] and not self.states['ma']:
                amt = portfolio_val * (self.c['ma_b_pct'] / 100.0)
                if cash >= amt:
                    self.buy(size=int(amt/self.dataclose[0]), info={'name': 'MA_Buy'})
                    self.states['ma'] = True
            
            # 賣出：乖離率 < 設定值
            elif diff_pct < self.c['ma_s_trig'] and self.states['ma']:
                if self.position.size > 0:
                    size_sell = int(self.position.size * (self.c['ma_s_pct'] / 100.0))
                    if size_sell > 0:
                        self.sell(size=size_sell, info={'name': 'MA_Sell'})
                    self.states['ma'] = False

        # =============================================
        # 3. ROC 策略
        # =============================================
        if self.c['use_roc']:
            # 買入：ROC > 設定值
            if self.roc[0] > self.c['roc_b_trig'] and not self.states['roc']:
                amt = portfolio_val * (self.c['roc_b_pct'] / 100.0)
                if cash >= amt:
                    self.buy(size=int(amt/self.dataclose[0]), info={'name': 'ROC_Buy'})
                    self.states['roc'] = True
            
            # 賣出：ROC < 設定值
            elif self.roc[0] < self.c['roc_s_trig'] and self.states['roc']:
                if self.position.size > 0:
                    size_sell = int(self.position.size * (self.c['roc_s_pct'] / 100.0))
                    if size_sell > 0:
                        self.sell(size=size_sell, info={'name': 'ROC_Sell'})
                    self.states['roc'] = False

        # =============================================
        # 4. ADX 策略
        # =============================================
        if self.c['use_adx']:
            # 買入：ADX > 設定值
            if self.adx[0] > self.c['adx_b_trig'] and not self.states['adx']:
                amt = portfolio_val * (self.c['adx_b_pct'] / 100.0)
                if cash >= amt:
                    self.buy(size=int(amt/self.dataclose[0]), info={'name': 'ADX_Buy'})
                    self.states['adx'] = True
            
            # 賣出：ADX < 設定值
            elif self.adx[0] < self.c['adx_s_trig'] and self.states['adx']:
                if self.position.size > 0:
                    size_sell = int(self.position.size * (self.c['adx_s_pct'] / 100.0))
                    if size_sell > 0:
                        self.sell(size=size_sell, info={'name': 'ADX_Sell'})
                    self.states['adx'] = False

# ==========================================
# 4. UI 設定
# ==========================================
st.sidebar.header("🎛️ VIX 階梯全參數設定")

with st.sidebar.expander("1. 基礎設定", expanded=True):
    symbol = st.text_input("股票代碼", "NVDA")
    start_date = st.date_input("開始日期", datetime.date(2023, 1, 1))
    init_cash = st.number_input("本金", value=10000.0)
    comm_pct = st.number_input("手續費 (%)", value=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("2. VIX 恐慌階梯 (買低賣高)")
st.sidebar.caption("提示：為避免刷單，賣出數值請設得比買入數值低 (例如 買>20, 賣<18)")

config = {}

def create_vix_ui(level, def_b_trig, def_b_pct, def_s_trig, def_s_pct):
    st.markdown(f"**Level {level}**")
    c1, c2, c3, c4, c5 = st.sidebar.columns([0.5, 1, 1, 1, 1])
    en = c1.checkbox(f"開", True, key=f"en_{level}")
    b_trig = c2.number_input(f"買入 >", value=def_b_trig, key=f"vb_{level}")
    b_pct = c3.number_input(f"買資%", value=def_b_pct, key=f"vp_{level}")
    s_trig = c4.number_input(f"賣出 <", value=def_s_trig, key=f"vs_{level}")
    s_pct = c5.number_input(f"賣倉%", value=def_s_pct, key=f"vsp_{level}")
    
    config[f'vix_en_{level}'] = en
    config[f'vix_b_trig_{level}'] = b_trig
    config[f'vix_b_pct_{level}'] = b_pct
    config[f'vix_s_trig_{level}'] = s_trig
    config[f'vix_s_pct_{level}'] = s_pct

create_vix_ui(1, 20.0, 10.0, 18.0, 50.0)
create_vix_ui(2, 25.0, 20.0, 20.0, 50.0)
create_vix_ui(3, 30.0, 30.0, 25.0, 100.0)

st.sidebar.markdown("---")
st.sidebar.subheader("3. 技術指標 (買賣點自訂)")

# MA 設定
with st.sidebar.expander("MA (均線乖離)", expanded=True):
    use_ma = st.checkbox("啟用 MA", True)
    ma_len = st.number_input("MA 週期", 20)
    c1, c2 = st.columns(2)
    ma_b_trig = c1.number_input("買入 (乖離率 >)", value=0.0)
    ma_b_pct = c2.number_input("MA 買入資金 %", value=10.0)
    c3, c4 = st.columns(2)
    ma_s_trig = c3.number_input("賣出 (乖離率 <)", value=-1.0)
    ma_s_pct = c4.number_input("MA 賣出持倉 %", value=50.0)

config.update({'use_ma': use_ma, 'ma_len': ma_len, 'ma_b_trig': ma_b_trig, 'ma_b_pct': ma_b_pct, 'ma_s_trig': ma_s_trig, 'ma_s_pct': ma_s_pct})

# ROC 設定
with st.sidebar.expander("ROC (動能)", expanded=False):
    use_roc = st.checkbox("啟用 ROC", False)
    roc_len = st.number_input("ROC 週期", 12)
    r1, r2 = st.columns(2)
    roc_b_trig = r1.number_input("買入 (ROC >)", value=0.0)
    roc_b_pct = r2.number_input("ROC 買入資金 %", value=10.0)
    r3, r4 = st.columns(2)
    roc_s_trig = r3.number_input("賣出 (ROC <)", value=-2.0)
    roc_s_pct = r4.number_input("ROC 賣出持倉 %", value=100.0)

config.update({'use_roc': use_roc, 'roc_len': roc_len, 'roc_b_trig': roc_b_trig, 'roc_b_pct': roc_b_pct, 'roc_s_trig': roc_s_trig, 'roc_s_pct': roc_s_pct})

# ADX 設定
with st.sidebar.expander("ADX (趨勢)", expanded=False):
    use_adx = st.checkbox("啟用 ADX", False)
    adx_len = st.number_input("ADX 週期", 14)
    a1, a2 = st.columns(2)
    adx_b_trig = a1.number_input("買入 (ADX >)", value=25.0)
    adx_b_pct = a2.number_input("ADX 買入資金 %", value=10.0)
    a3, a4 = st.columns(2)
    adx_s_trig = a3.number_input("賣出 (ADX <)", value=20.0)
    adx_s_pct = a4.number_input("ADX 賣出持倉 %", value=100.0)

config.update({'use_adx': use_adx, 'adx_len': adx_len, 'adx_b_trig': adx_b_trig, 'adx_b_pct': adx_b_pct, 'adx_s_trig': adx_s_trig, 'adx_s_pct': adx_s_pct})

btn = st.sidebar.button("🚀 執行除錯後的回測", type="primary")

# ==========================================
# 5. 主程式
# ==========================================
if btn:
    gc.collect()
    with st.spinner("計算中..."):
        df = get_data(symbol, start_date)
        if df.empty: st.error("無數據"); st.stop()

        god_curve = calculate_god_mode(df, init_cash)
        
        cerebro = bt.Cerebro()
        cerebro.adddata(PandasDataPlus(dataname=df))
        cerebro.addstrategy(FullyConfigurableStrategy, config=config)
        cerebro.broker.setcash(init_cash)
        cerebro.broker.setcommission(commission=comm_pct/100.0)
        
        results = cerebro.run()
        strat = results[0]
        final_val = cerebro.broker.getvalue()
        
        # 數據整理
        dates = df.index[-len(strat.value_history):]
        eq_data = [{"time": d.strftime('%Y-%m-%d'), "value": v} for d, v in zip(dates, strat.value_history)]
        cash_data = [{"time": d.strftime('%Y-%m-%d'), "value": v} for d, v in zip(dates, strat.cash_history)]
        trade_log = pd.DataFrame(strat.trade_list)
        bh_val = (df['Close'].iloc[-1] / df['Close'].iloc[0]) * init_cash

    # UI 顯示
    st.title(f"🛠️ {symbol} 參數化回測戰報")
    c1, c2, c3 = st.columns(3)
    c1.metric("最終權益", f"${final_val:,.0f}", delta=f"{((final_val-init_cash)/init_cash)*100:.1f}%")
    c2.metric("Buy & Hold", f"${bh_val:,.0f}", delta=f"{((bh_val-init_cash)/init_cash)*100:.1f}%")
    c3.metric("總交易次數", f"{len(trade_log)}")

    # Chart
    st.subheader("📈 資金成長曲線")
    chart_opts = {
        "chart": {"height": 350, "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"}},
        "series": [
            {"type": "Line", "data": god_curve, "options": {"color": "#FFD700", "lineWidth": 1, "title": "God Mode"}},
            {"type": "Area", "data": eq_data, "options": {"lineColor": "#00E676", "topColor": "rgba(0, 230, 118, 0.2)", "bottomColor": "rgba(0,0,0,0)", "title": "策略權益"}},
            {"type": "Area", "data": cash_data, "options": {"lineColor": "#2962FF", "topColor": "rgba(41,98,255,0.4)", "bottomColor": "rgba(41,98,255,0.1)", "title": "現金水位"}}
        ]
    }
    renderLightweightCharts([chart_opts], key="main")

    # K線 + 買賣點
    st.subheader("🕯️ 買賣點還原")
    kline_data = [{"time": i.strftime('%Y-%m-%d'), "open": r['Open'], "high": r['High'], "low": r['Low'], "close": r['Close']} for i, r in df.iterrows()]
    series = [{"type": 'Candlestick', "data": kline_data, "options": {"upColor": '#089981', "downColor": '#f23645'}}]
    
    if not trade_log.empty:
        markers = []
        for _, t in trade_log.iterrows():
            is_buy = t['Type'] == 'Buy'
            color = "#00E676" if is_buy else "#FF5252"
            
            # 【重要修正】強制轉換為字串，防止 TypeError
            raw_reason = str(t['Reason'])
            label = raw_reason.replace('_Buy','').replace('_Sell','') 
            
            markers.append({
                "time": t['Date'].strftime('%Y-%m-%d'),
                "position": "belowBar" if is_buy else "aboveBar",
                "color": color,
                "shape": "arrowUp" if is_buy else "arrowDown",
                "text": label
            })
        series[0]["markers"] = markers
    
    renderLightweightCharts([{"chart": {"height": 500, "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"}}, "series": series}], key="candle")

    if not trade_log.empty:
        st.subheader("📋 交易詳細紀錄")
        trade_log['Amount'] = trade_log['Price'] * trade_log['Size']
        display_df = trade_log.copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        display_df['Price'] = display_df['Price'].map('{:,.2f}'.format)
        display_df['Amount'] = display_df['Amount'].map('{:,.0f}'.format)
        
        def highlight(row):
            c = '#00E676' if row['Type']=='Buy' else '#FF5252'
            bg = 'rgba(255, 215, 0, 0.15)' if 'VIX' in str(row['Reason']) else 'transparent'
            return [f'color: {c}; background-color: {bg}'] * len(row)
        st.dataframe(display_df.style.apply(highlight, axis=1), use_container_width=True)
    else:
        st.warning("⚠️ 沒有交易產生。可能原因：\n1. VIX 門檻設太高\n2. 賣出門檻比買入門檻還高 (邏輯錯誤)\n3. 資金已用完")
