import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import backtrader as bt
import datetime
import plotly.graph_objects as go
from streamlit_lightweight_charts import renderLightweightCharts
import collections.abc

# 1. 兼容性修復
collections.Iterable = collections.abc.Iterable

# 2. 頁面設定
st.set_page_config(page_title="獨立因子回測 v17", layout="wide")
st.markdown("""
<style>
    header {visibility: hidden;}
    .block-container {padding-top: 0rem !important;}
    .stApp {background-color: #0e1117;}
    input {font-weight: bold; color: #00e676 !important;}
    div[data-testid="stMetric"] {background-color: #262730; border: 1px solid #464b5f; border-radius: 5px;}
    div[data-testid="stMetricLabel"] {color: #babcbf;}
    div[data-testid="stMetricValue"] {color: #ffffff;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. Backtrader 策略 (獨立觸發邏輯)
# ==========================================
class IndependentStrategy(bt.Strategy):
    params = (('config', {}),)

    def __init__(self):
        super().__init__()
        self.dataclose = self.datas[0].close
        self.c = self.params.config
        self.order = None
        self.trade_list = []
        
        # VIX
        self.vix = self.datas[0].vix if hasattr(self.datas[0], 'vix') else None

        # 指標
        self.inds = {}
        if self.c['use_ema']:
            self.inds['ema'] = bt.indicators.EMA(self.datas[0], period=int(self.c['ema_len']))
        
        if self.c['use_rsi']:
            self.inds['rsi'] = bt.indicators.RSI(self.datas[0], period=int(self.c['rsi_len']))
            
        if self.c['use_macd']:
            self.inds['macd'] = bt.indicators.MACD(self.datas[0], 
                                                   period_me1=int(self.c['macd_fast']), 
                                                   period_me2=int(self.c['macd_slow']), 
                                                   period_signal=int(self.c['macd_sig']))

    def notify_order(self, order):
        if order.status in [order.Completed]:
            # 記錄是誰觸發的交易 (透過 order.ref 或是我們自己在 buy 時傳入 info，這裡簡化處理)
            action = 'Buy' if order.isbuy() else 'Sell'
            self.trade_list.append({
                'Date': bt.num2date(order.executed.dt),
                'Type': action,
                'Price': order.executed.price,
                'Size': order.executed.size,
                'Value': order.executed.value,
                'Comm': order.executed.comm,
                'Reason': getattr(order.info, 'name', 'Signal') # 嘗試讀取訂單備註
            })
            self.order = None

    def execute_buy(self, pct, reason):
        """ 執行買入：使用剩餘現金的 pct% """
        if pct <= 0: return
        cash = self.broker.getcash()
        target_amount = cash * (pct / 100.0)
        size = int(target_amount / self.dataclose[0])
        if size > 0:
            # 傳入 info 讓我們知道是誰買的
            self.buy(size=size, info={'name': reason})

    def execute_sell(self, pct, reason):
        """ 執行賣出：使用目前持倉的 pct% """
        if pct <= 0: return
        pos_size = self.position.size
        if pos_size > 0:
            target_size = int(pos_size * (pct / 100.0))
            if target_size > 0:
                self.sell(size=target_size, info={'name': reason})

    def next(self):
        if self.order: return

        # 這裡的邏輯改為：檢查「穿越 (CrossOver)」瞬間
        # 防止每天符合條件就每天買，買到沒錢為止
        
        # =======================
        # 1. VIX 獨立邏輯
        # =======================
        if self.c['use_vix'] and self.vix:
            # 買入：VIX 向上突破 買入閥值 (代表恐慌發生)
            # 或者 VIX 向下穿越 買入閥值 (代表恐慌消退) -> 這裡依你需求設定，通常是恐慌時買
            # 我們設定為：當 VIX > 閥值 的瞬間 (代表恐慌飆升)
            if self.vix[0] > self.c['vix_buy_thres'] and self.vix[-1] <= self.c['vix_buy_thres']:
                self.execute_buy(self.c['vix_buy_pct'], f"VIX>{int(self.c['vix_buy_thres'])}")
            
            # 賣出：VIX 跌破 賣出閥值 (代表市場安逸)
            if self.vix[0] < self.c['vix_sell_thres'] and self.vix[-1] >= self.c['vix_sell_thres']:
                self.execute_sell(self.c['vix_sell_pct'], f"VIX<{int(self.c['vix_sell_thres'])}")

        # =======================
        # 2. EMA 獨立邏輯
        # =======================
        if self.c['use_ema']:
            # 買入：價格 黃金交叉 EMA
            if self.dataclose[0] > self.inds['ema'][0] and self.dataclose[-1] <= self.inds['ema'][-1]:
                self.execute_buy(self.c['ema_buy_pct'], "EMA金叉")
            
            # 賣出：價格 死亡交叉 EMA
            if self.dataclose[0] < self.inds['ema'][0] and self.dataclose[-1] >= self.inds['ema'][-1]:
                self.execute_sell(self.c['ema_sell_pct'], "EMA死叉")

        # =======================
        # 3. MACD 獨立邏輯
        # =======================
        if self.c['use_macd']:
            # 買入：MACD 金叉
            if self.inds['macd'].macd[0] > self.inds['macd'].signal[0] and self.inds['macd'].macd[-1] <= self.inds['macd'].signal[-1]:
                self.execute_buy(self.c['macd_buy_pct'], "MACD金叉")
            
            # 賣出：MACD 死叉
            if self.inds['macd'].macd[0] < self.inds['macd'].signal[0] and self.inds['macd'].macd[-1] >= self.inds['macd'].signal[-1]:
                self.execute_sell(self.c['macd_sell_pct'], "MACD死叉")
                
        # =======================
        # 4. RSI 獨立邏輯
        # =======================
        if self.c['use_rsi']:
            # 買入：RSI 跌破買點 (超賣)
            if self.inds['rsi'][0] < self.c['rsi_buy_val'] and self.inds['rsi'][-1] >= self.c['rsi_buy_val']:
                self.execute_buy(self.c['rsi_buy_pct'], "RSI超賣")
                
            # 賣出：RSI 突破賣點 (超買)
            if self.inds['rsi'][0] > self.c['rsi_sell_val'] and self.inds['rsi'][-1] <= self.c['rsi_sell_val']:
                self.execute_sell(self.c['rsi_sell_pct'], "RSI超買")

class PandasDataPlus(bt.feeds.PandasData):
    lines = ('vix',)
    params = (('vix', -1),)

# ==========================================
# 4. 側邊欄：參數設定區 (獨立設定)
# ==========================================
st.sidebar.header("🎛️ 獨立因子設定室")

with st.sidebar.expander("1. 基礎設定", expanded=True):
    symbol = st.text_input("股票代碼", "NVDA")
    init_cash = st.number_input("初始本金", value=100000.0)
    comm_rate = st.number_input("手續費率 (%)", value=0.1425) / 100.0

# VIX 設定
with st.sidebar.expander("2. VIX 恐慌指標", expanded=True):
    use_vix = st.checkbox("啟用 VIX", True)
    c1, c2 = st.columns(2)
    vix_buy_thres = c1.number_input("VIX > 多少買", value=26.0)
    vix_buy_pct = c2.number_input("買入資金 % (VIX)", value=30.0) # 獨立資金比例
    
    c3, c4 = st.columns(2)
    vix_sell_thres = c3.number_input("VIX < 多少賣", value=13.0)
    vix_sell_pct = c4.number_input("賣出持倉 % (VIX)", value=50.0)

# EMA 設定
with st.sidebar.expander("3. EMA 趨勢指標", expanded=False):
    use_ema = st.checkbox("啟用 EMA", True)
    ema_len = st.number_input("EMA 週期", value=20)
    c1, c2 = st.columns(2)
    ema_buy_pct = c1.number_input("站上 EMA 買入資金 %", value=30.0)
    ema_sell_pct = c2.number_input("跌破 EMA 賣出持倉 %", value=100.0)

# MACD 設定
with st.sidebar.expander("4. MACD 動能指標", expanded=False):
    use_macd = st.checkbox("啟用 MACD", False)
    macd_fast = st.number_input("快線", 12)
    macd_slow = st.number_input("慢線", 26)
    macd_sig = st.number_input("訊號", 9)
    c1, c2 = st.columns(2)
    macd_buy_pct = c1.number_input("金叉買入資金 %", value=20.0)
    macd_sell_pct = c2.number_input("死叉賣出持倉 %", value=50.0)

# RSI 設定
with st.sidebar.expander("5. RSI 震盪指標", expanded=False):
    use_rsi = st.checkbox("啟用 RSI", True)
    rsi_len = st.number_input("RSI 週期", 14)
    c1, c2 = st.columns(2)
    rsi_buy_val = c1.number_input("低於多少買", 30)
    rsi_buy_pct = c2.number_input("買入資金 % (RSI)", 20.0)
    c3, c4 = st.columns(2)
    rsi_sell_val = c3.number_input("高於多少賣", 70)
    rsi_sell_pct = c4.number_input("賣出持倉 % (RSI)", 50.0)

config = {
    'use_vix': use_vix, 'vix_buy_thres': vix_buy_thres, 'vix_buy_pct': vix_buy_pct, 
    'vix_sell_thres': vix_sell_thres, 'vix_sell_pct': vix_sell_pct,
    'use_ema': use_ema, 'ema_len': ema_len, 'ema_buy_pct': ema_buy_pct, 'ema_sell_pct': ema_sell_pct,
    'use_macd': use_macd, 'macd_fast': macd_fast, 'macd_slow': macd_slow, 'macd_sig': macd_sig,
    'macd_buy_pct': macd_buy_pct, 'macd_sell_pct': macd_sell_pct,
    'use_rsi': use_rsi, 'rsi_len': rsi_len, 'rsi_buy_val': rsi_buy_val, 'rsi_buy_pct': rsi_buy_pct,
    'rsi_sell_val': rsi_sell_val, 'rsi_sell_pct': rsi_sell_pct
}

start_date = st.sidebar.date_input("開始日期", datetime.date(2023, 1, 1))
btn_run = st.sidebar.button("🚀 執行獨立回測", type="primary")

# ==========================================
# 5. 執行
# ==========================================
if btn_run:
    with st.spinner("正在運算..."):
        # 下載
        df = yf.download(symbol, start=start_date, end=datetime.date.today(), progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        vix_df = yf.download("^VIX", start=start_date, end=datetime.date.today(), progress=False)
        if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.get_level_values(0)
        df['vix'] = vix_df['Close'].reindex(df.index).ffill()
        
        if df.empty:
            st.error("無數據")
            st.stop()

        cerebro = bt.Cerebro()
        cerebro.adddata(PandasDataPlus(dataname=df))
        cerebro.addstrategy(IndependentStrategy, config=config)
        cerebro.broker.setcash(init_cash)
        cerebro.broker.setcommission(commission=comm_rate)
        
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
        results = cerebro.run()
        strat = results[0]
        
        final_val = cerebro.broker.getvalue()
        roi = (final_val - init_cash) / init_cash * 100
        t_ret = strat.analyzers.timereturn.get_analysis()
        equity_curve = pd.Series(t_ret).fillna(0)
        equity_curve = (1 + equity_curve).cumprod() * init_cash
        bh_ret = df['Close'].pct_change().fillna(0)
        bh_curve = (1 + bh_ret).cumprod() * init_cash
        bh_roi = (bh_curve.iloc[-1] - init_cash) / init_cash * 100
        
        trade_log = pd.DataFrame(strat.trade_list)

    # UI
    st.title(f"🛡️ {symbol} 獨立因子回測報告")
    
    # 績效
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最終權益", f"${final_val:,.0f}", f"{roi:.2f}%")
    c2.metric("Buy & Hold", f"${bh_curve.iloc[-1]:,.0f}", f"{bh_roi:.2f}%")
    c3.metric("Alpha", f"{roi - bh_roi:.2f}%")
    c4.metric("交易次數", len(trade_log) if not trade_log.empty else 0)

    # 曲線
    st.subheader("📈 資金成長")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name='策略', line=dict(color='#00e676', width=2)))
    fig.add_trace(go.Scatter(x=bh_curve.index, y=bh_curve.values, mode='lines', name='B&H', line=dict(color='#555555', dash='dash')))
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
    st.plotly_chart(fig, use_container_width=True)

    # K線
    st.subheader("🕯️ K線與買賣點")
    kline_data = [{"time": i.strftime('%Y-%m-%d'), "open": r['Open'], "high": r['High'], "low": r['Low'], "close": r['Close']} for i, r in df.iterrows()]
    series_main = [{"type": 'Candlestick', "data": kline_data, "options": {"upColor": '#089981', "downColor": '#f23645', "borderVisible": False}}]
    
    if not trade_log.empty:
        markers = []
        for _, t in trade_log.iterrows():
            # 根據 Reason 顯示不同顏色或文字
            txt = "B" if t['Type']=='Buy' else "S"
            if "VIX" in str(t['Reason']): txt = "V"
            if "EMA" in str(t['Reason']): txt = "E"
            
            markers.append({
                "time": t['Date'].strftime('%Y-%m-%d'),
                "position": "belowBar" if t['Type']=='Buy' else "aboveBar",
                "color": "#089981" if t['Type']=='Buy' else "#f23645",
                "shape": "arrowUp" if t['Type']=='Buy' else "arrowDown",
                "text": txt
            })
        series_main[0]["markers"] = markers

    chart_opts = {"layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"}, "height": 450}
    renderLightweightCharts([{"chart": chart_opts, "series": series_main}], key="v17_chart")

    # 明細
    if not trade_log.empty:
        st.subheader("📋 交易日記 (含觸發原因)")
        trade_log['Date'] = trade_log['Date'].dt.strftime('%Y-%m-%d')
        trade_log['Value'] = trade_log['Value'].abs().map('{:.0f}'.format)
        trade_log['Comm'] = trade_log['Comm'].map('{:.2f}'.format)
        st.dataframe(trade_log, use_container_width=True)
