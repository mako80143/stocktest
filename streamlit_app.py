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
st.set_page_config(page_title="VIX 階梯佈局系統", layout="wide")

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

def calculate_god_mode(df, init_cash):
    data = df['Close'].values
    min_idx = argrelextrema(data, np.less, order=5)[0]
    max_idx = argrelextrema(data, np.greater, order=5)[0]
    cash = init_cash
    shares = 0
    god_curve = []
    for i in range(len(df)):
        price = data[i]
        if i in min_idx and cash > 0:
            shares = cash / price; cash = 0
        elif i in max_idx and shares > 0:
            cash = shares * price; shares = 0
        god_curve.append({"time": df.index[i].strftime('%Y-%m-%d'), "value": cash + (shares * price)})
    return god_curve

# ==========================================
# 3. Backtrader 策略 (獨立資金模組)
# ==========================================
class PandasDataPlus(bt.feeds.PandasData):
    lines = ('vix',)
    params = (('vix', -1),)

class GridAllocationStrategy(bt.Strategy):
    params = (('config', {}),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.vix = self.datas[0].vix
        self.c = self.params.config
        
        self.trade_list = []
        self.cash_history = []
        self.value_history = []
        
        # 狀態追蹤：避免重複買入
        # 我們使用字典來記錄每個條件是否已經持倉 (True = 已買入)
        self.state = {
            'vix_L1': False, 'vix_L2': False, 'vix_L3': False, 'vix_L4': False, 'vix_L5': False,
            'ma': False, 'roc': False, 'adx': False
        }
        
        # 指標初始化
        self.ma_short = bt.indicators.SMA(self.datas[0], period=int(self.c['ma_short_len']))
        self.ma_trend = bt.indicators.SMA(self.datas[0], period=int(self.c['ma_trend_len']))
        
        self.roc = bt.indicators.ROC(self.datas[0], period=int(self.c['roc_len']))
        self.roc_ma = bt.indicators.SMA(self.roc, period=int(self.c['roc_ma_len']))
        
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
                'Size': order.executed.size,
                'Reason': getattr(order.info, 'name', 'Signal')
            })

    def next(self):
        self.cash_history.append(self.broker.getcash())
        self.value_history.append(self.broker.getvalue())
        if len(self) < 100: return
        
        # 總資產 (用於計算買入金額的基準)
        portfolio_value = self.broker.getvalue()
        current_cash = self.broker.getcash()

        # =========================================
        # A. VIX 階梯模組 (5層網格)
        # =========================================
        vix_levels = self.c['vix_grid']
        
        for i, level in enumerate(vix_levels):
            key = f'vix_L{i+1}'
            threshold = level['trigger']
            buy_pct = level['buy_pct']
            
            # 觸發條件：VIX 超過門檻 且 尚未針對此層級買入
            if self.vix[0] >= threshold and not self.state[key]:
                # 執行買入
                amount = portfolio_value * (buy_pct / 100.0)
                if current_cash >= amount and amount > 0:
                    size = int(amount / self.dataclose[0])
                    if size > 0:
                        self.buy(size=size, info={'name': f'VIX_Lv{i+1}_{threshold}'})
                        self.state[key] = True # 標記已買
            
            # 賣出條件 (釋放資金)：VIX 回落低於門檻 且 目前持有此層級
            # 這裡賣出邏輯：回落就賣掉該層級對應的部位 (這是一種波段操作)
            # 或者您可以設定成 "VIX 回落到很低才賣"，這裡先預設為 "脫離危險區就獲利了結/止損"
            elif self.vix[0] < threshold and self.state[key]:
                # 計算要賣多少：賣出持倉的 % (這裡使用設定的賣出趴數)
                sell_pct_of_pos = level['sell_pct'] # 使用者設定的 "賣出多少"
                
                # 注意：這裡賣出是針對 "總持倉" 的百分比，還是 "當初買的那一份"？
                # 為了符合您的需求 "賣多少"，我們解釋為 "賣出目前總持股的 X%"
                if self.position.size > 0:
                    size_to_sell = int(self.position.size * (sell_pct_of_pos / 100.0))
                    if size_to_sell > 0:
                        self.sell(size=size_to_sell, info={'name': f'VIX_Off_Lv{i+1}'})
                    self.state[key] = False # 重置狀態，下次VIX再飆高可以再買

        # =========================================
        # B. 技術指標模組 (獨立運作)
        # =========================================
        
        # 1. MA 模組
        if self.c['use_ma']:
            # 買進：收盤 > 長均線 且 尚未持有MA部位
            if self.dataclose[0] > self.ma_trend[0] and not self.state['ma']:
                amt = portfolio_value * (self.c['ma_buy_pct'] / 100.0)
                if current_cash >= amt and amt > 0:
                    self.buy(size=int(amt/self.dataclose[0]), info={'name': 'MA_Buy'})
                    self.state['ma'] = True
            
            # 賣出：收盤 < 短均線 (止損) 且 持有MA部位
            elif self.dataclose[0] < self.ma_short[0] and self.state['ma']:
                if self.position.size > 0:
                    size_sell = int(self.position.size * (self.c['ma_sell_pct'] / 100.0))
                    self.sell(size=size_sell, info={'name': 'MA_Exit'})
                self.state['ma'] = False

        # 2. ROC 模組
        if self.c['use_roc']:
            # 買進：動能轉強
            if self.roc[0] > self.roc_ma[0] and not self.state['roc']:
                amt = portfolio_value * (self.c['roc_buy_pct'] / 100.0)
                if current_cash >= amt and amt > 0:
                    self.buy(size=int(amt/self.dataclose[0]), info={'name': 'ROC_Boost'})
                    self.state['roc'] = True
            
            # 賣出：動能轉弱
            elif self.roc[0] < self.roc_ma[0] and self.state['roc']:
                if self.position.size > 0:
                    size_sell = int(self.position.size * (self.c['roc_sell_pct'] / 100.0))
                    self.sell(size=size_sell, info={'name': 'ROC_Cut'})
                self.state['roc'] = False

        # 3. ADX 模組
        if self.c['use_adx']:
            # 買進：趨勢強勁且多頭
            is_bull = self.di_plus[0] > self.di_minus[0]
            is_strong = self.adx[0] > self.c['adx_thres']
            if is_bull and is_strong and not self.state['adx']:
                amt = portfolio_value * (self.c['adx_buy_pct'] / 100.0)
                if current_cash >= amt and amt > 0:
                    self.buy(size=int(amt/self.dataclose[0]), info={'name': 'ADX_Trend'})
                    self.state['adx'] = True
            
            # 賣出：ADX 轉折
            is_weakening = (self.adx[0] < self.adx[-1]) and (self.adx[-1] > self.c['adx_strong'])
            if is_weakening and self.state['adx']:
                if self.position.size > 0:
                    size_sell = int(self.position.size * (self.c['adx_sell_pct'] / 100.0))
                    self.sell(size=size_sell, info={'name': 'ADX_Fade'})
                self.state['adx'] = False

# ==========================================
# 4. 控制台 UI
# ==========================================
st.sidebar.header("🪜 階梯式資金佈局")

with st.sidebar.expander("1. 基礎設定", expanded=True):
    symbol = st.text_input("股票代碼", "NVDA")
    start_date = st.date_input("開始日期", datetime.date(2023, 1, 1))
    init_cash = st.number_input("本金", value=10000.0)
    comm_pct = st.number_input("手續費 (%)", value=0.1)

# A. VIX 網格設定
st.sidebar.subheader("2. VIX 恐慌階梯 (Grid)")
st.caption("設定 VIX 超過多少(Trigger)時，買入總資金的幾趴(Buy%)；當 VIX 回落時，賣出持倉的幾趴(Sell%)")

vix_grid = []
cols = st.sidebar.columns([1, 1, 1])
cols[0].markdown("**Trigger >**")
cols[1].markdown("**Buy %**")
cols[2].markdown("**Sell %**")

# 產生 5 層輸入框 (Level 1 ~ 5)
defaults = [
    (20.0, 5.0, 10.0),   # Lv1: VIX>20, Buy 5%, Sell 10%
    (25.0, 10.0, 20.0),  # Lv2
    (30.0, 20.0, 50.0),  # Lv3
    (40.0, 30.0, 50.0),  # Lv4
    (50.0, 50.0, 100.0)  # Lv5
]

for i in range(5):
    c1, c2, c3 = st.sidebar.columns([1, 1, 1])
    d_trig, d_buy, d_sell = defaults[i]
    
    t = c1.number_input(f"Lv{i+1} 觸發", value=d_trig, key=f"v_t_{i}")
    b = c2.number_input(f"買入 %", value=d_buy, key=f"v_b_{i}")
    s = c3.number_input(f"賣出 %", value=d_sell, key=f"v_s_{i}")
    
    vix_grid.append({'trigger': t, 'buy_pct': b, 'sell_pct': s})

# B. 指標獨立設定
st.sidebar.subheader("3. 技術指標獨立設定")
st.caption("Buy%: 買入總資金百分比 | Sell%: 賣出持倉百分比")

# MA
with st.sidebar.expander("MA 均線", expanded=True):
    use_ma = st.checkbox("啟用 MA", True)
    m1, m2 = st.columns(2)
    ma_short = m1.number_input("短均線", value=20.0)
    ma_trend = m2.number_input("長均線", value=50.0)
    m3, m4 = st.columns(2)
    ma_buy_pct = m3.number_input("MA 買入 %", value=10.0)
    ma_sell_pct = m4.number_input("MA 賣出 %", value=50.0)

# ROC
with st.sidebar.expander("ROC 動能", expanded=False):
    use_roc = st.checkbox("啟用 ROC", True)
    r1, r2 = st.columns(2)
    roc_len = r1.number_input("ROC 週期", value=12.0)
    roc_ma_len = r2.number_input("ROC MA", value=6.0)
    r3, r4 = st.columns(2)
    roc_buy_pct = r3.number_input("ROC 買入 %", value=10.0)
    roc_sell_pct = r4.number_input("ROC 賣出 %", value=50.0)

# ADX
with st.sidebar.expander("ADX 趨勢", expanded=False):
    use_adx = st.checkbox("啟用 ADX", True)
    a1, a2, a3 = st.columns(3)
    adx_len = a1.number_input("週期", value=14.0)
    adx_thres = a2.number_input("買入 >", value=20.0)
    adx_strong = a3.number_input("賣出(轉折) >", value=25.0)
    a4, a5 = st.columns(2)
    adx_buy_pct = a4.number_input("ADX 買入 %", value=10.0)
    adx_sell_pct = a5.number_input("ADX 賣出 %", value=50.0)

config = {
    'vix_grid': vix_grid,
    'use_ma': use_ma, 'ma_short_len': ma_short, 'ma_trend_len': ma_trend, 'ma_buy_pct': ma_buy_pct, 'ma_sell_pct': ma_sell_pct,
    'use_roc': use_roc, 'roc_len': roc_len, 'roc_ma_len': roc_ma_len, 'roc_buy_pct': roc_buy_pct, 'roc_sell_pct': roc_sell_pct,
    'use_adx': use_adx, 'adx_len': adx_len, 'adx_thres': adx_thres, 'adx_strong': adx_strong, 'adx_buy_pct': adx_buy_pct, 'adx_sell_pct': adx_sell_pct
}

btn = st.sidebar.button("🚀 執行階梯回測", type="primary")

# ==========================================
# 5. 主程式
# ==========================================
if btn:
    gc.collect()
    with st.spinner("正在進行資金推演..."):
        df = get_data(symbol, start_date)
        if df.empty: st.error("無數據"); st.stop()

        # 上帝視角
        god_curve = calculate_god_mode(df, init_cash)
        
        # 策略回測
        cerebro = bt.Cerebro()
        cerebro.adddata(PandasDataPlus(dataname=df))
        cerebro.addstrategy(GridAllocationStrategy, config=config)
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
        
        # Buy & Hold
        bh_val = (df['Close'].iloc[-1] / df['Close'].iloc[0]) * init_cash

    # UI 顯示
    st.title(f"🪜 {symbol} 階梯佈局戰報")
    
    # 績效比較
    god_final = god_curve[-1]['value']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("😇 上帝視角", f"${god_final:,.0f}")
    c2.metric("😈 階梯策略", f"${final_val:,.0f}", delta=f"{((final_val-init_cash)/init_cash)*100:.1f}%")
    c3.metric("😴 Buy & Hold", f"${bh_val:,.0f}", delta=f"{((bh_val-init_cash)/init_cash)*100:.1f}%")
    c4.metric("目前庫存", f"{strat.position.size} 股")

    # 資產圖表
    st.subheader("🏆 資金成長與水位")
    st.caption("觀察階梯狀的買入點 (藍色區域下降) 是否發生在市場低點")
    
    chart_opts = {
        "chart": {"height": 400, "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"}},
        "series": [
            {"type": "Line", "data": god_curve, "options": {"color": "#FFD700", "lineWidth": 1, "lineStyle": 2, "title": "上帝極限"}},
            {"type": "Area", "data": eq_data, "options": {"lineColor": "#00E676", "topColor": "rgba(0, 230, 118, 0.2)", "bottomColor": "rgba(0,0,0,0)", "title": "策略權益"}},
            {"type": "Area", "data": cash_data, "options": {"lineColor": "#2962FF", "topColor": "rgba(41,98,255,0.4)", "bottomColor": "rgba(41,98,255,0.1)", "title": "現金水位"}}
        ]
    }
    renderLightweightCharts([chart_opts], key="main_chart")
    
    # 交易明細
    if not trade_log.empty:
        st.subheader("📋 階梯買賣紀錄")
        trade_log['Amount'] = trade_log['Price'] * trade_log['Size']
        
        display_df = trade_log.copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        display_df['Price'] = display_df['Price'].map('{:,.2f}'.format)
        display_df['Amount'] = display_df['Amount'].map('{:,.0f}'.format)
        
        def highlight(row):
            c = '#00E676' if row['Type']=='Buy' else '#FF5252'
            # 如果是 VIX 觸發的，背景標示為黃色
            bg = 'rgba(255, 215, 0, 0.15)' if 'VIX' in row['Reason'] else 'transparent'
            return [f'color: {c}; background-color: {bg}'] * len(row)
            
        st.dataframe(display_df.style.apply(highlight, axis=1), use_container_width=True)
    else:
        st.info("無交易紀錄。請檢查您的 VIX 或指標門檻是否設得太高。")
