import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import backtrader as bt
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_lightweight_charts import renderLightweightCharts
import collections.abc
import numpy as np
from scipy.signal import argrelextrema
import gc # 記憶體管理
import warnings

# --- 1. 環境修復區 ---
# 忽略 Backtrader 在 Python 3.13 的語法警告
warnings.filterwarnings("ignore", category=SyntaxWarning)

# 修復 Iterable 相容性
collections.Iterable = collections.abc.Iterable

st.set_page_config(page_title="全能戰情室 v24.2", layout="wide")
st.markdown("""
<style>
    header {visibility: hidden;}
    .block-container {padding-top: 0rem !important;}
    .stApp {background-color: #0e1117;}
    input {font-weight: bold; color: #00e676 !important;}
    div[data-testid="stMetric"] {background-color: #262730; border: 1px solid #464b5f; border-radius: 5px;}
    div[data-testid="stMetricLabel"] {color: #babcbf;}
    div[data-testid="stMetricValue"] {color: #ffffff;}
    div[data-testid="stExpander"] {background-color: #262730; border: 1px solid #464b5f;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心運算：數據下載 (含快取)
# ==========================================
@st.cache_data(ttl=3600)
def get_data_with_indicators(symbol, start):
    end = datetime.date.today()
    
    # 下載主數據
    df = yf.download(symbol, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if df.empty: return pd.DataFrame() 

    # 移除時區
    df.index = df.index.tz_localize(None)
    
    # 下載 VIX
    vix_df = yf.download("^VIX", start=start, end=end, progress=False)
    if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.get_level_values(0)
    
    if not vix_df.empty:
        vix_df.index = vix_df.index.tz_localize(None)
        # 合併數據
        df['vix'] = vix_df['Close'].reindex(df.index).ffill()
    else:
        df['vix'] = 0 # 防呆
    
    return df

# ==========================================
# 3. 數學運算：上帝視角
# ==========================================
def calculate_god_mode(df, init_cash):
    data = df['Close'].values
    # 尋找極值 (前後5天)
    min_idx = argrelextrema(data, np.less, order=5)[0]
    max_idx = argrelextrema(data, np.greater, order=5)[0]
    
    cash = init_cash
    shares = 0
    god_curve = pd.Series(index=df.index, dtype=float)
    god_curve.iloc[0] = init_cash
    
    for i in range(len(df)):
        price = data[i]
        
        # 遇到低點全買
        if i in min_idx and cash > 0:
            shares = cash / price
            cash = 0
        # 遇到高點全賣
        elif i in max_idx and shares > 0:
            cash = shares * price
            shares = 0
            
        # 計算當日市值
        if shares > 0:
            val = shares * price
        else:
            val = cash
        god_curve.iloc[i] = val
        
    return god_curve.ffill()

# ==========================================
# 4. Backtrader 策略 (資金流 + 獨立邏輯)
# ==========================================
class IntegratedStrategy(bt.Strategy):
    params = (('config', {}),)

    def __init__(self):
        super().__init__()
        self.dataclose = self.datas[0].close
        self.c = self.params.config
        
        # 綁定 VIX
        self.vix = self.datas[0].vix if hasattr(self.datas[0], 'vix') else None
        
        self.trade_list = []
        self.skipped_list = []
        
        self.inds = {}
        if self.c.get('use_ema'): 
            self.inds['ema'] = bt.indicators.EMA(self.datas[0], period=int(self.c.get('ema_len', 20)))
        if self.c.get('use_macd'):
            self.inds['macd'] = bt.indicators.MACD(self.datas[0], period_me1=12, period_me2=26, period_signal=9)
        if self.c.get('use_rsi'):
            self.inds['rsi'] = bt.indicators.RSI(self.datas[0], period=int(self.c.get('rsi_len', 14)))

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.trade_list.append({
                'Date': bt.num2date(order.executed.dt),
                'Type': 'Buy' if order.isbuy() else 'Sell',
                'Price': order.executed.price,
                'Value': order.executed.value,
                'Reason': getattr(order.info, 'name', 'Signal')
            })

    # 買入執行 (檢查資金)
    def attempt_buy(self, pct, reason):
        cash = self.broker.getcash()
        if cash < 100: 
            self.skipped_list.append({'Date': self.datas[0].datetime.date(0), 'Reason': f"{reason} (沒錢)"})
            return
        
        target = cash * (pct / 100.0) * 0.998 # 手續費緩衝
        size = int(target / self.dataclose[0])
        
        if size > 0: 
            self.buy(size=size, info={'name': reason})
        else:
            self.skipped_list.append({'Date': self.datas[0].datetime.date(0), 'Reason': f"{reason} (買不起)"})

    # 賣出執行 (檢查持倉)
    def attempt_sell(self, pct, reason):
        size = self.position.size
        if size > 0:
            target = int(size * (pct / 100.0))
            if target > 0: self.sell(size=target, info={'name': reason})

    def next(self):
        # 1. VIX 邏輯
        if self.c.get('use_vix') and self.vix:
            if self.vix[0] > self.c['vix_buy_thres']:
                if self.vix[-1] <= self.c['vix_buy_thres']: # 剛突破
                    self.attempt_buy(self.c['vix_buy_pct'], f"VIX>{int(self.c['vix_buy_thres'])}")
            
            if self.vix[0] < self.c['vix_sell_thres']:
                if self.vix[-1] >= self.c['vix_sell_thres']: # 剛跌破
                    self.attempt_sell(self.c['vix_sell_pct'], f"VIX<{int(self.c['vix_sell_thres'])}")

        # 2. EMA 邏輯
        if self.c.get('use_ema'):
            if self.dataclose[0] > self.inds['ema'][0] and self.dataclose[-1] <= self.inds['ema'][-1]:
                self.attempt_buy(self.c['ema_buy_pct'], "EMA Buy")
            if self.dataclose[0] < self.inds['ema'][0] and self.dataclose[-1] >= self.inds['ema'][-1]:
                self.attempt_sell(self.c['ema_sell_pct'], "EMA Sell")

        # 3. MACD 邏輯
        if self.c.get('use_macd'):
            if self.inds['macd'].macd[0] > self.inds['macd'].signal[0] and self.inds['macd'].macd[-1] <= self.inds['macd'].signal[-1]:
                self.attempt_buy(self.c['macd_buy_pct'], "MACD Buy")
            if self.inds['macd'].macd[0] < self.inds['macd'].signal[0] and self.inds['macd'].macd[-1] >= self.inds['macd'].signal[-1]:
                self.attempt_sell(self.c['macd_sell_pct'], "MACD Sell")

        # 4. RSI 邏輯
        if self.c.get('use_rsi'):
            if self.inds['rsi'][0] < self.c['rsi_buy_val'] and self.inds['rsi'][-1] >= self.c['rsi_buy_val']:
                self.attempt_buy(self.c['rsi_buy_pct'], "RSI Buy")
            if self.inds['rsi'][0] > self.c['rsi_sell_val'] and self.inds['rsi'][-1] <= self.c['rsi_sell_val']:
                self.attempt_sell(self.c['rsi_sell_pct'], "RSI Sell")

class PandasDataPlus(bt.feeds.PandasData):
    lines = ('vix',)
    params = (('vix', -1),)

# ==========================================
# 5. UI 與 參數設定
# ==========================================
st.sidebar.header("🎛️ 戰情控制台 v24.2")

symbol = st.sidebar.text_input("股票代碼", "NVDA")
start_date = st.sidebar.date_input("開始日期", datetime.date(2023, 1, 1))
init_cash = st.sidebar.number_input("初始本金", value=100000.0)

with st.sidebar.expander("1. VIX 策略 (獨立觸發)", expanded=True):
    use_vix = st.checkbox("啟用 VIX", True)
    c1, c2 = st.columns(2)
    vix_b_thres = c1.number_input("VIX > 多少買", 26.0)
    vix_b_pct = c2.number_input("買入資金 % (VIX)", 100.0)
    c3, c4 = st.columns(2)
    vix_s_thres = c3.number_input("VIX < 多少賣", 14.0)
    vix_s_pct = c4.number_input("賣出持倉 % (VIX)", 100.0)

with st.sidebar.expander("2. 其他指標", expanded=False):
    # EMA
    use_ema = st.checkbox("啟用 EMA", True)
    ema_len = st.number_input("EMA 週期", 20)
    c1, c2 = st.columns(2)
    ema_b_pct = c1.number_input("EMA 買 %", 30.0)
    ema_s_pct = c2.number_input("EMA 賣 %", 50.0)
    
    st.divider()
    # MACD
    use_macd = st.checkbox("啟用 MACD", False)
    c3, c4 = st.columns(2)
    macd_b_pct = c3.number_input("MACD 買 %", 30.0)
    macd_s_pct = c4.number_input("MACD 賣 %", 50.0)

    st.divider()
    # RSI
    use_rsi = st.checkbox("啟用 RSI", False)
    rsi_len = st.number_input("RSI 週期", 14)
    c5, c6 = st.columns(2)
    rsi_b_val = c5.number_input("RSI < 多少買", 30)
    rsi_b_pct = c6.number_input("RSI 買 %", 30.0)
    rsi_s_val = c5.number_input("RSI > 多少賣", 70)
    rsi_s_pct = c6.number_input("RSI 賣 %", 50.0)

config = {
    'use_vix': use_vix, 'vix_buy_thres': vix_b_thres, 'vix_buy_pct': vix_b_pct, 
    'vix_sell_thres': vix_s_thres, 'vix_sell_pct': vix_s_pct,
    'use_ema': use_ema, 'ema_len': ema_len, 'ema_buy_pct': ema_b_pct, 'ema_sell_pct': ema_s_pct,
    'use_macd': use_macd, 'macd_buy_pct': macd_b_pct, 'macd_sell_pct': macd_s_pct,
    'use_rsi': use_rsi, 'rsi_len': rsi_len, 'rsi_buy_val': rsi_b_val, 'rsi_buy_pct': rsi_b_pct,
    'rsi_sell_val': rsi_s_val, 'rsi_sell_pct': rsi_s_pct
}

btn_run = st.sidebar.button("🚀 執行完整運算", type="primary")

# ==========================================
# 6. 主程式執行
# ==========================================
if btn_run:
    gc.collect() # 清除記憶體

    with st.spinner("數據下載與策略運算中..."):
        df = get_data_with_indicators(symbol, start_date)
        
        if df.empty:
            st.error(f"❌ 找不到 {symbol} 的數據，請檢查代碼或日期。")
            st.stop()
            
        # 1. 計算上帝視角
        god_curve = calculate_god_mode(df, init_cash)
        god_final = god_curve.iloc[-1]
        
        # 2. 計算 Buy & Hold
        bh_curve = (df['Close'] / df['Close'].iloc[0]) * init_cash
        bh_final = bh_curve.iloc[-1]
        
        # 3. 執行 Backtrader
        cerebro = bt.Cerebro()
        cerebro.adddata(PandasDataPlus(dataname=df))
        cerebro.addstrategy(IntegratedStrategy, config=config)
        cerebro.broker.setcash(init_cash)
        cerebro.broker.setcommission(commission=0.001425)
        
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
        results = cerebro.run()
        strat = results[0]
        
        final_val = cerebro.broker.getvalue()
        roi = (final_val - init_cash) / init_cash * 100
        
        # 整理曲線
        t_ret = strat.analyzers.timereturn.get_analysis()
        equity_curve = pd.Series(t_ret).fillna(0)
        equity_curve = (1 + equity_curve).cumprod() * init_cash
        
        # 交易明細
        trade_log = pd.DataFrame(strat.trade_list)
        skipped_log = pd.DataFrame(strat.skipped_list)

    # UI 呈現
    st.title(f"⚡ {symbol} 終極戰報 (v24.2)")
    
    # 績效看板
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("😇 上帝視角", f"${god_final:,.0f}", delta=f"{(god_final-init_cash)/init_cash*100:.0f}%")
    c2.metric("😈 我的策略", f"${final_val:,.0f}", delta=f"{roi:.2f}%")
    c3.metric("😴 Buy & Hold", f"${bh_final:,.0f}")
    c4.metric("交易次數", len(trade_log))
    
    # 曲線圖
    st.subheader("📈 凡人 vs 上帝")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=god_curve.index, y=god_curve.values, mode='lines', name='上帝視角', line=dict(color='#FFD700', width=2)))
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name='我的策略', line=dict(color='#00e676', width=2)))
    fig.add_trace(go.Scatter(x=bh_curve.index, y=bh_curve.values, mode='lines', name='B&H', line=dict(color='#555555', dash='dash')))
    
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500, yaxis_type="log", title="對數座標 (Log Scale)")
    st.plotly_chart(fig, use_container_width=True)
    
    # K線圖
    st.subheader("🕯️ K 線訊號")
    kline_data = [{"time": i.strftime('%Y-%m-%d'), "open": r['Open'], "high": r['High'], "low": r['Low'], "close": r['Close']} for i, r in df.iterrows()]
    series_main = [{"type": 'Candlestick', "data": kline_data, "options": {"upColor": '#089981', "downColor": '#f23645', "borderVisible": False}}]
    
    if not trade_log.empty:
        markers = []
        for _, t in trade_log.iterrows():
            txt = "B" if t['Type']=='Buy' else "S"
            if "VIX" in str(t['Reason']): txt = "V"
            markers.append({
                "time": t['Date'].strftime('%Y-%m-%d'), "position": "belowBar" if t['Type']=='Buy' else "aboveBar",
                "color": "#089981" if t['Type']=='Buy' else "#f23645", "shape": "arrowUp" if t['Type']=='Buy' else "arrowDown", "text": txt
            })
        series_main[0]["markers"] = markers
        
    renderLightweightCharts([{"chart": {"height": 450, "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"}}, "series": series_main}], key="v24_main")

    # 明細
    c_log1, c_log2 = st.columns(2)
    with c_log1:
        st.subheader("✅ 交易日記")
        if not trade_log.empty:
            trade_log['Date'] = trade_log['Date'].dt.strftime('%Y-%m-%d')
            trade_log['Value'] = trade_log['Value'].abs().map('{:.0f}'.format)
            st.dataframe(trade_log, use_container_width=True)
        else:
            st.info("無交易")

    with c_log2:
        st.subheader("🚫 資金不足 (Skipped)")
        if not skipped_log.empty:
            skipped_log['Date'] = skipped_log['Date'].astype(str)
            st.dataframe(skipped_log, use_container_width=True)
        else:
            st.info("資金充足")
