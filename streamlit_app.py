import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import backtrader as bt
import datetime
import plotly.graph_objects as go
from streamlit_lightweight_charts import renderLightweightCharts
import collections.abc

# 1. 兼容性修復 (針對 Backtrader 與新版 Python)
collections.Iterable = collections.abc.Iterable

# 2. 頁面設定 (必須是第一行 Streamlit 指令)
st.set_page_config(page_title="無限回測 v15 (修復版)", layout="wide")

# 3. CSS 強制去除黑框與優化 (UI Fix)
st.markdown("""
<style>
    /* 隱藏 Streamlit 預設上方的 Header (黑框) */
    header {visibility: hidden;}
    /* 隱藏右上角的選單 (如果需要的話可以打開) */
    #MainMenu {visibility: hidden;}
    /* 去除頂部留白，讓畫面往上貼 */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* 深色模式優化 */
    .stApp {background-color: #0e1117;}
    
    /* 輸入框數據顏色 */
    input {font-weight: bold; color: #00e676 !important;}
    
    /* 儀表板卡片 */
    div[data-testid="stMetric"] {
        background-color: #262730; 
        border: 1px solid #464b5f; 
        border-radius: 5px;
    }
    div[data-testid="stMetricLabel"] {color: #babcbf;}
    div[data-testid="stMetricValue"] {color: #ffffff;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. Backtrader 策略 (邏輯補完)
# ==========================================
class LogicStrategy(bt.Strategy):
    params = (('config', {}),)

    def __init__(self):
        # 初始化父類 (重要修復)
        super().__init__()
        
        self.dataclose = self.datas[0].close
        self.c = self.params.config
        
        # ⚠️ 關鍵修復：明確初始化 order 變數，防止 AttributeError
        self.order = None 
        self.trade_list = []
        self.inds = {}
        
        # 1. 綁定 VIX (如果有)
        self.vix = self.datas[0].vix if hasattr(self.datas[0], 'vix') else None

        # 2. 指標初始化
        # SMA
        if self.c['use_sma']:
            self.inds['sma'] = bt.indicators.SMA(self.datas[0], period=int(self.c['sma_len']))
        
        # EMA (新增)
        if self.c['use_ema']:
            self.inds['ema'] = bt.indicators.EMA(self.datas[0], period=int(self.c['ema_len']))

        # RSI
        if self.c['use_rsi']:
            self.inds['rsi'] = bt.indicators.RSI(self.datas[0], period=int(self.c['rsi_len']))
            
        # MACD
        if self.c['use_macd']:
            self.inds['macd'] = bt.indicators.MACD(self.datas[0], 
                                                   period_me1=int(self.c['macd_fast']), 
                                                   period_me2=int(self.c['macd_slow']), 
                                                   period_signal=int(self.c['macd_sig']))

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.trade_list.append({
                    'Type': 'Buy', 'Date': bt.num2date(order.executed.dt),
                    'Price': order.executed.price, 'Size': order.executed.size,
                    'Comm': order.executed.comm, 'Value': order.executed.value
                })
            elif order.issell():
                self.trade_list.append({
                    'Type': 'Sell', 'Date': bt.num2date(order.executed.dt),
                    'Price': order.executed.price, 'Size': order.executed.size,
                    'Comm': order.executed.comm, 'Value': order.executed.value
                })
            self.order = None # 重置訂單狀態

    def next(self):
        # 如果有未完成的訂單，則不執行
        if self.order:
            return

        # --- A. 買入邏輯檢查 (Buy Logic) ---
        can_buy = True
        
        # 1. 宏觀 VIX 買入濾網
        if self.c['use_vix'] and self.vix:
            if self.c['vix_buy_logic'] == '高於閥值不買 (避險)':
                if self.vix[0] > self.c['vix_buy_thres']: can_buy = False
            elif self.c['vix_buy_logic'] == '低於閥值不買 (抄底)':
                if self.vix[0] < self.c['vix_buy_thres']: can_buy = False

        # 2. 趨勢濾網 (SMA/EMA)
        if can_buy and self.c['use_sma']:
            if self.dataclose[0] < self.inds['sma'][0]: can_buy = False
        if can_buy and self.c['use_ema']:
            if self.dataclose[0] < self.inds['ema'][0]: can_buy = False

        # 3. 觸發訊號 (Triggers)
        buy_sig = False
        if can_buy:
            triggers = []
            if self.c['use_rsi']:
                triggers.append(self.inds['rsi'][0] < self.c['rsi_buy'])
            if self.c['use_macd']:
                triggers.append(self.inds['macd'].macd[0] > self.inds['macd'].signal[0]) # 金叉
            
            # AND 邏輯：如果有啟用指標，必須全過
            if triggers and all(triggers):
                buy_sig = True
            # 如果沒啟用任何 Trigger 指標 (例如只開 SMA)，則 SMA 過了就買
            elif not triggers and can_buy and (self.c['use_sma'] or self.c['use_ema']):
                buy_sig = True

        # --- B. 賣出邏輯檢查 (Sell Logic) ---
        sell_sig = False
        
        if self.position:
            # 1. 宏觀 VIX 賣出 (強制逃命)
            if self.c['use_vix'] and self.vix and self.c['vix_sell_active']:
                if self.vix[0] > self.c['vix_sell_thres']:
                    sell_sig = True # 恐慌逃命

            # 2. RSI 賣出 (超買)
            if self.c['use_rsi'] and self.inds['rsi'][0] > self.c['rsi_sell']:
                sell_sig = True

            # 3. SMA 賣出 (跌破均線)
            if self.c['use_sma'] and self.dataclose[0] < self.inds['sma'][0]:
                sell_sig = True
                
            # 4. EMA 賣出 (跌破均線)
            if self.c['use_ema'] and self.dataclose[0] < self.inds['ema'][0]:
                sell_sig = True

            # 5. MACD 賣出 (死叉: 快線跌破訊號線)
            if self.c['use_macd'] and self.inds['macd'].macd[0] < self.inds['macd'].signal[0]:
                sell_sig = True

        # --- C. 執行 ---
        if not self.position and buy_sig:
            cash = self.broker.getcash()
            target_cash = 0
            if self.c['stake_type'] == '固定金額 (Fixed Cash)':
                target_cash = self.c['stake_val']
            else:
                target_cash = cash * (self.c['stake_val'] / 100.0)
            
            size = int(target_cash / self.dataclose[0])
            if size > 0:
                self.order = self.buy(size=size) # 紀錄 order
            
        elif self.position and sell_sig:
            self.order = self.close() # 紀錄 order

# 資料格式擴充
class PandasDataPlus(bt.feeds.PandasData):
    lines = ('vix',)
    params = (('vix', -1),)

# ==========================================
# 5. 側邊欄：全參數設定
# ==========================================
st.sidebar.header("🎛️ 策略指揮中心")

with st.sidebar.expander("1. 資金與標的", expanded=True):
    symbol = st.text_input("股票代碼", "NVDA")
    init_cash = st.number_input("初始本金", value=100000.0)
    comm_rate = st.number_input("手續費率 (%)", value=0.1425, format="%.6f") / 100.0
    
    stake_type = st.radio("投入方式", ["資金百分比 (%)", "固定金額 (Fixed Cash)"])
    stake_val = st.number_input("投入數值", value=100.0 if stake_type=="資金百分比 (%)" else 50000.0)

with st.sidebar.expander("2. 宏觀 VIX 設定 (買/賣)", expanded=True):
    use_vix = st.checkbox("啟用 VIX 監控", True)
    
    st.caption("🟢 買入濾網")
    vix_buy_logic = st.selectbox("買入邏輯", ["高於閥值不買 (避險)", "低於閥值不買 (抄底)"])
    vix_buy_thres = st.number_input("VIX 買入閥值", value=30.0)
    
    st.caption("🔴 賣出觸發 (新增)")
    vix_sell_active = st.checkbox("啟用 VIX 強制賣出", False)
    vix_sell_thres = st.number_input("VIX 賣出閥值 (高於此值逃命)", value=40.0)

with st.sidebar.expander("3. 技術指標參數 (買/賣)", expanded=True):
    # SMA
    st.markdown("---")
    use_sma = st.checkbox("SMA (簡單均線)", True)
    sma_len = st.number_input("SMA 週期", value=20)
    
    # EMA
    st.markdown("---")
    use_ema = st.checkbox("EMA (指數均線)", False)
    ema_len = st.number_input("EMA 週期", value=20)
    
    # RSI
    st.markdown("---")
    use_rsi = st.checkbox("RSI (相對強弱)", True)
    c1, c2, c3 = st.columns(3)
    rsi_len = c1.number_input("RSI 週期", value=14)
    rsi_buy = c2.number_input("買入 <", value=30)
    rsi_sell = c3.number_input("賣出 >", value=70)
    
    # MACD
    st.markdown("---")
    use_macd = st.checkbox("MACD (動能)", False)
    m1, m2, m3 = st.columns(3)
    macd_fast = m1.number_input("快線", 12)
    macd_slow = m2.number_input("慢線", 26)
    macd_sig = m3.number_input("訊號", 9)
    st.caption("邏輯: 金叉買入，死叉賣出")

config = {
    'stake_type': stake_type, 'stake_val': stake_val,
    'use_vix': use_vix, 'vix_buy_logic': vix_buy_logic, 'vix_buy_thres': vix_buy_thres,
    'vix_sell_active': vix_sell_active, 'vix_sell_thres': vix_sell_thres,
    'use_sma': use_sma, 'sma_len': sma_len,
    'use_ema': use_ema, 'ema_len': ema_len,
    'use_rsi': use_rsi, 'rsi_len': rsi_len, 'rsi_buy': rsi_buy, 'rsi_sell': rsi_sell,
    'use_macd': use_macd, 'macd_fast': macd_fast, 'macd_slow': macd_slow, 'macd_sig': macd_sig
}

start_date = st.sidebar.date_input("開始日期", datetime.date(2023, 1, 1))
btn_run = st.sidebar.button("🚀 執行修復版回測", type="primary")

# ==========================================
# 6. 主程式執行
# ==========================================
if btn_run:
    with st.spinner("正在修復並運算..."):
        # 下載
        df = yf.download(symbol, start=start_date, end=datetime.date.today(), progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        vix_df = yf.download("^VIX", start=start_date, end=datetime.date.today(), progress=False)
        if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.get_level_values(0)
        df['vix'] = vix_df['Close'].reindex(df.index).ffill()
        
        if df.empty:
            st.error("無數據")
            st.stop()

        # 回測引擎
        cerebro = bt.Cerebro()
        cerebro.adddata(PandasDataPlus(dataname=df))
        cerebro.addstrategy(LogicStrategy, config=config)
        cerebro.broker.setcash(init_cash)
        cerebro.broker.setcommission(commission=comm_rate)
        
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
        results = cerebro.run()
        strat = results[0]
        
        # 績效計算
        final_val = cerebro.broker.getvalue()
        roi = (final_val - init_cash) / init_cash * 100
        
        t_ret = strat.analyzers.timereturn.get_analysis()
        equity_curve = pd.Series(t_ret).fillna(0)
        equity_curve = (1 + equity_curve).cumprod() * init_cash
        
        bh_ret = df['Close'].pct_change().fillna(0)
        bh_curve = (1 + bh_ret).cumprod() * init_cash
        bh_roi = (bh_curve.iloc[-1] - init_cash) / init_cash * 100
        
        trade_log = pd.DataFrame(strat.trade_list)

    # UI 呈現
    st.title(f"🛡️ {symbol} 回測戰報 (v15)")
    
    # 1. 儀表板
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最終權益", f"${final_val:,.0f}", f"{roi:.2f}%")
    c2.metric("Buy & Hold", f"${bh_curve.iloc[-1]:,.0f}", f"{bh_roi:.2f}%")
    c3.metric("Alpha", f"{roi - bh_roi:.2f}%")
    c4.metric("交易次數", len(trade_log) if not trade_log.empty else 0)

    # 2. 資金曲線
    st.subheader("📈 資金成長")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name='策略', line=dict(color='#00e676', width=2)))
    fig.add_trace(go.Scatter(x=bh_curve.index, y=bh_curve.values, mode='lines', name='B&H', line=dict(color='#555555', dash='dash')))
    
    if not trade_log.empty:
        buys = trade_log[trade_log['Type'] == 'Buy']
        sells = trade_log[trade_log['Type'] == 'Sell']
        fig.add_trace(go.Scatter(x=buys['Date'], y=equity_curve.loc[buys['Date']], mode='markers', name='買入', marker=dict(color='yellow', symbol='triangle-up', size=8)))
        fig.add_trace(go.Scatter(x=sells['Date'], y=equity_curve.loc[sells['Date']], mode='markers', name='賣出', marker=dict(color='red', symbol='triangle-down', size=8)))

    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # 3. K線圖
    st.subheader("🕯️ K 線訊號")
    kline_data = [{"time": i.strftime('%Y-%m-%d'), "open": r['Open'], "high": r['High'], "low": r['Low'], "close": r['Close']} for i, r in df.iterrows()]
    
    series_main = [{
        "type": 'Candlestick',
        "data": kline_data,
        "options": {"upColor": '#089981', "downColor": '#f23645', "borderVisible": False}
    }]
    
    # 疊加指標
    if config['use_sma']:
        sma_vals = ta.sma(df['Close'], length=int(config['sma_len']))
        d = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in sma_vals.items() if not pd.isna(v)]
        series_main.append({"type": "Line", "data": d, "options": {"color": "yellow", "lineWidth": 2, "title": "SMA"}})
        
    if config['use_ema']:
        ema_vals = ta.ema(df['Close'], length=int(config['ema_len']))
        d = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in ema_vals.items() if not pd.isna(v)]
        series_main.append({"type": "Line", "data": d, "options": {"color": "orange", "lineWidth": 2, "title": "EMA"}})

    if not trade_log.empty:
        markers = []
        for _, t in trade_log.iterrows():
            markers.append({
                "time": t['Date'].strftime('%Y-%m-%d'),
                "position": "belowBar" if t['Type']=='Buy' else "aboveBar",
                "color": "#089981" if t['Type']=='Buy' else "#f23645",
                "shape": "arrowUp" if t['Type']=='Buy' else "arrowDown",
                "text": "B" if t['Type']=='Buy' else "S"
            })
        series_main[0]["markers"] = markers

    chart_opts = {
        "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"},
        "grid": {"vertLines": {"color": "#2a2e39"}, "horzLines": {"color": "#2a2e39"}},
        "height": 450
    }
    charts = [{"chart": chart_opts, "series": series_main}]

    # VIX 副圖
    if config['use_vix']:
        vix_d = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in df['vix'].items()]
        charts.append({
            "chart": {**chart_opts, "height": 150},
            "series": [{"type": "Line", "data": vix_d, "options": {"color": "#ef5350", "title": "VIX"}}]
        })
        
    renderLightweightCharts(charts, key="v15_chart")

    # 4. 交易明細
    if not trade_log.empty:
        st.subheader("📋 交易日記")
        trade_log['Date'] = trade_log['Date'].dt.strftime('%Y-%m-%d')
        trade_log['Value'] = trade_log.get('Value', 0).fillna(0).abs().map('{:.0f}'.format)
        st.dataframe(trade_log, use_container_width=True)
