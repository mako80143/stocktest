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
st.set_page_config(page_title="多指標獨立區間回測", layout="wide")

st.markdown("""
<style>
    header {visibility: hidden;}
    .block-container {padding-top: 0rem !important; padding-bottom: 1rem !important;}
    .stApp {background-color: #0e1117;}
    input {font-weight: bold; color: #00e676 !important;}
    div[data-baseweb="input"] > div {background-color: #1e2130; color: white;}
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

# 上帝視角計算
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
# 3. Backtrader 策略 (獨立區間邏輯)
# ==========================================
class PandasDataPlus(bt.feeds.PandasData):
    lines = ('vix',)
    params = (('vix', -1),)

class IndependentModuleStrategy(bt.Strategy):
    params = (('config', {}),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.vix = self.datas[0].vix
        self.c = self.params.config
        
        self.trade_list = []
        self.cash_history = []
        self.value_history = []
        
        # 狀態機 (State Machine)
        # 用來記錄每個模組目前是 "空手(Neutral)" 還是 "持有(Long)"
        # 這樣才能實現：觸發買入後 -> 進入持有狀態 -> 等待觸發賣出 -> 回到空手狀態
        self.states = {
            'vix': 'neutral',
            'ma': 'neutral',
            'roc': 'neutral',
            'adx': 'neutral'
        }

        # --- 指標初始化 ---
        self.ma = bt.indicators.SMA(self.datas[0], period=int(self.c['ma_len']))
        self.roc = bt.indicators.ROC(self.datas[0], period=int(self.c['roc_len']))
        self.adx = bt.indicators.ADX(self.datas[0], period=int(self.c['adx_len']))

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.trade_list.append({
                'Date': bt.num2date(order.executed.dt),
                'Type': 'Buy' if order.isbuy() else 'Sell',
                'Price': order.executed.price,
                'Value': order.executed.value,
                'Size': order.executed.size,
                'Reason': getattr(order.info, 'name', 'Signal')
            })

    def next(self):
        self.cash_history.append(self.broker.getcash())
        self.value_history.append(self.broker.getvalue())
        if len(self) < 100: return

        # 總資產 (用於計算買入金額)
        portfolio_value = self.broker.getvalue()
        current_cash = self.broker.getcash()

        # =========================================
        # 1. VIX 模組 (恐慌買入，平靜賣出)
        # =========================================
        if self.c['use_vix']:
            # 買入條件：VIX 高於設定值 且 目前此模組空手
            if self.vix[0] > self.c['vix_buy_at'] and self.states['vix'] == 'neutral':
                amt = portfolio_value * (self.c['vix_buy_pct'] / 100.0)
                if current_cash >= amt and amt > 0:
                    self.buy(size=int(amt/self.dataclose[0]), info={'name': f"VIX>{self.c['vix_buy_at']}"})
                    self.states['vix'] = 'long' # 狀態轉為持有

            # 賣出條件：VIX 低於設定值 且 目前此模組持有
            # 這裡解決了 "剛買就賣" 的問題，因為中間有寬度
            elif self.vix[0] < self.c['vix_sell_at'] and self.states['vix'] == 'long':
                if self.position.size > 0:
                    # 賣出目前持股的 %
                    size_sell = int(self.position.size * (self.c['vix_sell_pct'] / 100.0))
                    if size_sell > 0:
                        self.sell(size=size_sell, info={'name': f"VIX<{self.c['vix_sell_at']}"})
                    self.states['vix'] = 'neutral' # 狀態轉為空手，準備下一次進攻

        # =========================================
        # 2. MA 模組 (趨勢跟隨)
        # =========================================
        if self.c['use_ma']:
            # 買入邏輯：股價站上均線的 X% (可以設為 1.0 代表剛好站上，或 1.02 代表站穩 2%)
            # 這裡簡單化：股價 > 均線
            if self.dataclose[0] > self.ma[0] and self.states['ma'] == 'neutral':
                amt = portfolio_value * (self.c['ma_buy_pct'] / 100.0)
                if current_cash >= amt and amt > 0:
                    self.buy(size=int(amt/self.dataclose[0]), info={'name': "MA_Cross"})
                    self.states['ma'] = 'long'

            # 賣出邏輯：股價跌破均線
            # 或者您可以設定：跌破均線的 98% 才賣 (避免假跌破) -> logic: price < ma * 0.98
            elif self.dataclose[0] < (self.ma[0] * 0.99) and self.states['ma'] == 'long':
                if self.position.size > 0:
                    size_sell = int(self.position.size * (self.c['ma_sell_pct'] / 100.0))
                    if size_sell > 0:
                        self.sell(size=size_sell, info={'name': "MA_Break"})
                    self.states['ma'] = 'neutral'

        # =========================================
        # 3. ROC 模組 (動能爆發)
        # =========================================
        if self.c['use_roc']:
            # 買入：ROC > 買入值
            if self.roc[0] > self.c['roc_buy_at'] and self.states['roc'] == 'neutral':
                amt = portfolio_value * (self.c['roc_buy_pct'] / 100.0)
                if current_cash >= amt and amt > 0:
                    self.buy(size=int(amt/self.dataclose[0]), info={'name': f"ROC>{self.c['roc_buy_at']}"})
                    self.states['roc'] = 'long'
            
            # 賣出：ROC < 賣出值 (通常設 0 或負數)
            elif self.roc[0] < self.c['roc_sell_at'] and self.states['roc'] == 'long':
                if self.position.size > 0:
                    size_sell = int(self.position.size * (self.c['roc_sell_pct'] / 100.0))
                    if size_sell > 0:
                        self.sell(size=size_sell, info={'name': f"ROC<{self.c['roc_sell_at']}"})
                    self.states['roc'] = 'neutral'

        # =========================================
        # 4. ADX 模組 (強趨勢)
        # =========================================
        if self.c['use_adx']:
            # 買入：ADX > 買入值 (趨勢形成)
            if self.adx[0] > self.c['adx_buy_at'] and self.states['adx'] == 'neutral':
                amt = portfolio_value * (self.c['adx_buy_pct'] / 100.0)
                if current_cash >= amt and amt > 0:
                    self.buy(size=int(amt/self.dataclose[0]), info={'name': f"ADX>{self.c['adx_buy_at']}"})
                    self.states['adx'] = 'long'

            # 賣出：ADX < 賣出值 (趨勢冷卻)
            elif self.adx[0] < self.c['adx_sell_at'] and self.states['adx'] == 'long':
                if self.position.size > 0:
                    size_sell = int(self.position.size * (self.c['adx_sell_pct'] / 100.0))
                    if size_sell > 0:
                        self.sell(size=size_sell, info={'name': f"ADX<{self.c['adx_sell_at']}"})
                    self.states['adx'] = 'neutral'

# ==========================================
# 4. 控制台 UI
# ==========================================
st.sidebar.header("🛠️ 獨立區間回測系統")

with st.sidebar.expander("1. 基礎設定", expanded=True):
    symbol = st.text_input("股票代碼", "NVDA")
    start_date = st.date_input("開始日期", datetime.date(2023, 1, 1))
    init_cash = st.number_input("本金", value=10000.0)
    comm_pct = st.number_input("手續費 (%)", value=0.1)

st.sidebar.markdown("---")
st.sidebar.caption("提示：Buy% 是指買入**總資金**的幾趴。Sell% 是指賣出**持股**的幾趴。")

# --- VIX 設定 ---
st.sidebar.subheader("2. VIX 恐慌區間")
use_vix = st.sidebar.checkbox("啟用 VIX", True)
c1, c2 = st.sidebar.columns(2)
vix_buy_at = c1.number_input("買入當 VIX >", value=30.0)
vix_buy_pct = c2.number_input("VIX 買入資金 %", value=30.0)
c3, c4 = st.sidebar.columns(2)
vix_sell_at = c3.number_input("賣出當 VIX <", value=20.0, help="一定要比買入值低，才能形成區間")
vix_sell_pct = c4.number_input("VIX 賣出持倉 %", value=100.0)

# --- MA 設定 ---
st.sidebar.subheader("3. MA 均線設定")
use_ma = st.sidebar.checkbox("啟用 MA", True)
ma_len = st.sidebar.number_input("MA 週期", value=20)
m1, m2 = st.sidebar.columns(2)
ma_buy_pct = m1.number_input("MA 買入資金 %", value=10.0)
ma_sell_pct = m2.number_input("MA 賣出持倉 %", value=100.0)
st.sidebar.caption("邏輯：收盤價站上 MA 買入，跌破 MA (1%緩衝) 賣出")

# --- ROC 設定 ---
st.sidebar.subheader("4. ROC 動能區間")
use_roc = st.sidebar.checkbox("啟用 ROC", False)
roc_len = st.sidebar.number_input("ROC 週期", value=12)
r1, r2 = st.sidebar.columns(2)
roc_buy_at = r1.number_input("買入當 ROC >", value=0.0)
roc_buy_pct = r2.number_input("ROC 買入資金 %", value=10.0)
r3, r4 = st.sidebar.columns(2)
roc_sell_at = r3.number_input("賣出當 ROC <", value=-2.0)
roc_sell_pct = r4.number_input("ROC 賣出持倉 %", value=100.0)

# --- ADX 設定 ---
st.sidebar.subheader("5. ADX 趨勢區間")
use_adx = st.sidebar.checkbox("啟用 ADX", False)
adx_len = st.sidebar.number_input("ADX 週期", value=14)
a1, a2 = st.sidebar.columns(2)
adx_buy_at = a1.number_input("買入當 ADX >", value=25.0)
adx_buy_pct = a2.number_input("ADX 買入資金 %", value=10.0)
a3, a4 = st.sidebar.columns(2)
adx_sell_at = a3.number_input("賣出當 ADX <", value=20.0, help="低於此值代表趨勢結束")
adx_sell_pct = a4.number_input("ADX 賣出持倉 %", value=100.0)

config = {
    'use_vix': use_vix, 'vix_buy_at': vix_buy_at, 'vix_buy_pct': vix_buy_pct, 'vix_sell_at': vix_sell_at, 'vix_sell_pct': vix_sell_pct,
    'use_ma': use_ma, 'ma_len': ma_len, 'ma_buy_pct': ma_buy_pct, 'ma_sell_pct': ma_sell_pct,
    'use_roc': use_roc, 'roc_len': roc_len, 'roc_buy_at': roc_buy_at, 'roc_buy_pct': roc_buy_pct, 'roc_sell_at': roc_sell_at, 'roc_sell_pct': roc_sell_pct,
    'use_adx': use_adx, 'adx_len': adx_len, 'adx_buy_at': adx_buy_at, 'adx_buy_pct': adx_buy_pct, 'adx_sell_at': adx_sell_at, 'adx_sell_pct': adx_sell_pct,
}

btn = st.sidebar.button("🚀 執行修正版回測", type="primary")

# ==========================================
# 5. 主程式
# ==========================================
if btn:
    gc.collect()
    with st.spinner("計算中..."):
        df = get_data(symbol, start_date)
        if df.empty: st.error("無數據"); st.stop()

        # 上帝視角
        god_curve = calculate_god_mode(df, init_cash)
        
        # 策略執行
        cerebro = bt.Cerebro()
        cerebro.adddata(PandasDataPlus(dataname=df))
        cerebro.addstrategy(IndependentModuleStrategy, config=config)
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

    # UI 呈現
    st.title(f"🛠️ {symbol} 區間策略戰報")
    
    god_final = god_curve[-1]['value']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("😇 上帝視角", f"${god_final:,.0f}")
    c2.metric("😈 您的策略", f"${final_val:,.0f}", delta=f"{((final_val-init_cash)/init_cash)*100:.1f}%")
    c3.metric("😴 Buy & Hold", f"${bh_val:,.0f}", delta=f"{((bh_val-init_cash)/init_cash)*100:.1f}%")
    c4.metric("手續費", f"{comm_pct}%")

    st.subheader("📈 資金成長")
    chart_opts = {
        "chart": {"height": 400, "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"}},
        "series": [
            {"type": "Line", "data": god_curve, "options": {"color": "#FFD700", "lineWidth": 1, "lineStyle": 2, "title": "上帝極限"}},
            {"type": "Area", "data": eq_data, "options": {"lineColor": "#00E676", "topColor": "rgba(0, 230, 118, 0.2)", "bottomColor": "rgba(0,0,0,0)", "title": "策略權益"}},
            {"type": "Area", "data": cash_data, "options": {"lineColor": "#2962FF", "topColor": "rgba(41,98,255,0.4)", "bottomColor": "rgba(41,98,255,0.1)", "title": "現金水位"}}
        ]
    }
    renderLightweightCharts([chart_opts], key="main_chart")
    
    if not trade_log.empty:
        st.subheader("📋 交易明細")
        trade_log['Amount'] = trade_log['Price'] * trade_log['Size']
        
        display_df = trade_log.copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        display_df['Price'] = display_df['Price'].map('{:,.2f}'.format)
        display_df['Amount'] = display_df['Amount'].map('{:,.0f}'.format)
        
        def highlight(row):
            c = '#00E676' if row['Type']=='Buy' else '#FF5252'
            return [f'color: {c}'] * len(row)
            
        st.dataframe(display_df.style.apply(highlight, axis=1), use_container_width=True)
    else:
        st.info("無交易，請檢查條件是否過於嚴苛。")
