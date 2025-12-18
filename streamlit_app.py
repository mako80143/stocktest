import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import backtrader as bt
import datetime
import plotly.graph_objects as go
from streamlit_lightweight_charts import renderLightweightCharts
import collections.abc

# 兼容性修復
collections.Iterable = collections.abc.Iterable

# --- 1. 頁面全黑化設定 ---
st.set_page_config(page_title="黑夜戰情室 v13", layout="wide")
st.markdown("""
<style>
    /* 強制深色背景與文字優化 */
    .stApp {background-color: #0e1117;}
    .block-container {padding-top: 1rem;}
    
    /* 儀表板卡片樣式 */
    div[data-testid="stMetric"] {
        background-color: #262730;
        border: 1px solid #464b5f;
        padding: 10px;
        border-radius: 5px;
        color: white;
    }
    div[data-testid="stMetricLabel"] {color: #babcbf;}
    div[data-testid="stMetricValue"] {color: #ffffff;}
    
    /* 側邊欄優化 */
    section[data-testid="stSidebar"] {background-color: #262730;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Backtrader 策略核心 (優先順序邏輯)
# ==========================================
class PriorityStrategy(bt.Strategy):
    params = (('config', {}),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.c = self.params.config
        
        # 綁定 VIX 數據 (如果有的話)
        # 注意：我們稍後會把 VIX 併入主數據的額外欄位，方便存取
        self.vix = self.datas[0].vix if hasattr(self.datas[0], 'vix') else None
        
        self.trade_list = []
        self.inds = {}

        # --- 初始化指標 (根據參數) ---
        # 趨勢
        if self.c['use_sma']:
            self.inds['sma'] = bt.indicators.SMA(self.datas[0], period=self.c['sma_len'])
        
        # 震盪
        if self.c['use_rsi']:
            self.inds['rsi'] = bt.indicators.RSI(self.datas[0], period=14)
            
        # MACD
        if self.c['use_macd']:
            self.inds['macd'] = bt.indicators.MACD(self.datas[0])

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.trade_list.append({
                    'Type': 'Buy', 'Date': bt.num2date(order.executed.dt),
                    'Price': order.executed.price, 'Size': order.executed.size,
                    'Cost': order.executed.value, 'Comm': order.executed.comm
                })
            elif order.issell():
                self.trade_list.append({
                    'Type': 'Sell', 'Date': bt.num2date(order.executed.dt),
                    'Price': order.executed.price, 'Size': order.executed.size,
                    'Value': order.executed.value, 'Comm': order.executed.comm
                })

    def next(self):
        if self.order: return

        # =========================================
        # 🧠 核心邏輯：漏斗式篩選 (Priority Funnel)
        # =========================================
        
        # 預設狀態
        can_buy = True
        sell_signal = False
        
        # --- 第 1 關：宏觀濾網 (VIX) ---
        # 許多策略是：VIX 太高不買(怕崩盤) 或是 VIX 高才買(恐慌抄底)
        # 這裡根據使用者設定
        if self.c['use_vix']:
            current_vix = self.datas[0].vix[0]
            if self.c['vix_mode'] == '避險模式 (VIX高不買)':
                if current_vix > self.c['vix_thres']: can_buy = False
            elif self.c['vix_mode'] == '抄底模式 (VIX高才買)':
                if current_vix < self.c['vix_thres']: can_buy = False

        # --- 第 2 關：趨勢濾網 (Trend) ---
        if can_buy and self.c['use_sma']:
            # 只有股價 > SMA 才允許做多 (多頭排列)
            if self.dataclose[0] < self.inds['sma'][0]:
                can_buy = False

        # --- 第 3 關：進場訊號 (Trigger) ---
        buy_signal = False
        
        # 只有前面兩關都通過，才檢查進場指標
        if can_buy:
            triggers = []
            if self.c['use_rsi']:
                triggers.append(self.inds['rsi'][0] < self.c['rsi_buy'])
            if self.c['use_macd']:
                triggers.append(self.inds['macd'].macd[0] > self.inds['macd'].signal[0])
            
            # 判斷邏輯：所有啟用的 Trigger 都要符合 (AND)
            if triggers and all(triggers):
                buy_signal = True

        # --- 第 4 關：出場訊號 (Exit) ---
        # 出場通常比較寬鬆，只要指標過熱或跌破均線就跑
        if self.position:
            exits = []
            if self.c['use_rsi']:
                exits.append(self.inds['rsi'][0] > self.c['rsi_sell'])
            if self.c['use_sma']:
                exits.append(self.dataclose[0] < self.inds['sma'][0])
            
            if any(exits): sell_signal = True

        # =========================
        # ⚡ 執行交易
        # =========================
        if not self.position and buy_signal:
            # 資金管理：投入設定的百分比
            cash = self.broker.getcash()
            size = int((cash * self.c['trade_pct']) / self.dataclose[0])
            if size > 0: self.buy(size=size)
            
        elif self.position and sell_signal:
            self.close()

# 擴充 PandasData 以支援 VIX 欄位
class PandasDataPlus(bt.feeds.PandasData):
    lines = ('vix',)
    params = (('vix', -1),) # 自動對應 DataFrame 中的 'vix' 欄位

# ==========================================
# 3. 側邊欄：參數設定實驗室
# ==========================================
st.sidebar.header("🎛️ 策略指揮中心")

with st.sidebar.expander("1. 標的與資金", expanded=True):
    symbol = st.text_input("股票代碼", "NVDA")
    start_date = st.date_input("開始", datetime.date(2023, 1, 1))
    init_cash = st.number_input("初始本金", 100000)
    trade_pct = st.slider("每次投入資金 %", 10, 100, 50) / 100.0
    comm_rate = st.number_input("手續費率 (%)", 0.1425) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ 決策優先順序")

# Layer 1: VIX
use_vix = st.sidebar.checkbox("1. 優先開啟 VIX 濾網", True)
vix_mode = "避險模式 (VIX高不買)"
vix_thres = 30
if use_vix:
    vix_mode = st.sidebar.selectbox("VIX 邏輯", ["避險模式 (VIX高不買)", "抄底模式 (VIX高才買)"])
    vix_thres = st.sidebar.slider(f"VIX 閥值 ({vix_mode[:2]})", 10, 80, 30)

# Layer 2: Trend
use_sma = st.sidebar.checkbox("2. 開啟 SMA 趨勢濾網", True)
sma_len = 20
if use_sma:
    sma_len = st.sidebar.number_input("SMA 均線週期 (只在價格之上買)", 20)

# Layer 3: Trigger
use_rsi = st.sidebar.checkbox("3. 開啟 RSI 進出場訊號", True)
rsi_buy, rsi_sell = 30, 70
if use_rsi:
    c1, c2 = st.sidebar.columns(2)
    rsi_buy = c1.number_input("RSI 買點 <", 30)
    rsi_sell = c2.number_input("RSI 賣點 >", 70)

use_macd = st.sidebar.checkbox("4. 開啟 MACD 金叉買入", False)

config = {
    'trade_pct': trade_pct,
    'use_vix': use_vix, 'vix_mode': vix_mode, 'vix_thres': vix_thres,
    'use_sma': use_sma, 'sma_len': sma_len,
    'use_rsi': use_rsi, 'rsi_buy': rsi_buy, 'rsi_sell': rsi_sell,
    'use_macd': use_macd
}

btn_run = st.sidebar.button("🚀 執行黑夜回測", type="primary")

# ==========================================
# 4. 主程式邏輯
# ==========================================
if btn_run:
    with st.spinner("正在下載數據與運算..."):
        # 1. 抓取數據
        df = yf.download(symbol, start=start_date, end=datetime.date.today(), progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # 抓取 VIX
        df_vix = yf.download("^VIX", start=start_date, end=datetime.date.today(), progress=False)
        if isinstance(df_vix.columns, pd.MultiIndex): df_vix.columns = df_vix.columns.get_level_values(0)
        
        # 合併 VIX 到主資料表 (關鍵步驟：對齊索引)
        df['vix'] = df_vix['Close'].reindex(df.index).ffill() # 缺值補前值

        if df.empty:
            st.error("無數據")
            st.stop()

        # 2. Backtrader 執行
        cerebro = bt.Cerebro()
        # 使用自訂的 PandasDataPlus 類別來讀取 vix 欄位
        data_feed = PandasDataPlus(dataname=df)
        cerebro.adddata(data_feed)
        
        cerebro.addstrategy(PriorityStrategy, config=config)
        cerebro.broker.setcash(init_cash)
        cerebro.broker.setcommission(commission=comm_rate)
        
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        
        results = cerebro.run()
        strat = results[0]
        
        # 3. 績效計算
        final_val = cerebro.broker.getvalue()
        roi = (final_val - init_cash) / init_cash * 100
        
        t_ret = strat.analyzers.timereturn.get_analysis()
        equity_curve = pd.Series(t_ret).fillna(0)
        equity_curve = (1 + equity_curve).cumprod() * init_cash
        
        # B&H
        bh_ret = df['Close'].pct_change().fillna(0)
        bh_curve = (1 + bh_ret).cumprod() * init_cash
        bh_roi = (bh_curve.iloc[-1] - init_cash) / init_cash * 100

        # 交易明細
        trade_log = pd.DataFrame(strat.trade_list)

    # ==========================================
    # 5. 黑夜版 UI 呈現
    # ==========================================
    st.title(f"🌑 {symbol} 策略戰報")
    
    # A. 儀表板
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最終權益", f"${final_val:,.0f}", f"{roi:.2f}%")
    c2.metric("Buy & Hold", f"${bh_curve.iloc[-1]:,.0f}", f"{bh_roi:.2f}%")
    c3.metric("Alpha 超額", f"{roi - bh_roi:.2f}%", help="策略 - 大盤")
    c4.metric("交易次數", len(trade_log) if not trade_log.empty else 0)

    # B. 資金曲線 (Plotly Dark Template)
    st.subheader("📈 獲利曲線")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name='我的策略', line=dict(color='#00e676', width=2)))
    fig.add_trace(go.Scatter(x=bh_curve.index, y=bh_curve.values, mode='lines', name='傻瓜持有', line=dict(color='#555555', dash='dash')))
    
    # 標記買賣點
    if not trade_log.empty:
        buys = trade_log[trade_log['Type'] == 'Buy']
        sells = trade_log[trade_log['Type'] == 'Sell']
        fig.add_trace(go.Scatter(x=buys['Date'], y=equity_curve.loc[buys['Date']], mode='markers', name='買入', marker=dict(color='yellow', symbol='triangle-up', size=10)))
        fig.add_trace(go.Scatter(x=sells['Date'], y=equity_curve.loc[sells['Date']], mode='markers', name='賣出', marker=dict(color='red', symbol='triangle-down', size=10)))

    # 設定全黑主題
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', # 透明背景融入網頁
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.02, x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # C. K線圖 (Lightweight Charts Dark Mode)
    st.subheader("🕯️ 交易訊號詳情")
    
    # 預計算指標供繪圖
    df['SMA'] = ta.sma(df['Close'], length=config['sma_len']) if config['use_sma'] else None
    
    kline_data = [{"time": i.strftime('%Y-%m-%d'), "open": r['Open'], "high": r['High'], "low": r['Low'], "close": r['Close']} for i, r in df.iterrows()]
    
    # 設定圖表 (全黑配色)
    chart_options = {
        "layout": {
            "background": {"type": "solid", "color": "#131722"}, # TradingView 深色背景
            "textColor": "#d1d4dc",
        },
        "grid": {
            "vertLines": {"color": "rgba(42, 46, 57, 0.5)"},
            "horzLines": {"color": "rgba(42, 46, 57, 0.5)"},
        },
        "height": 500
    }
    
    series_main = [{
        "type": 'Candlestick',
        "data": kline_data,
        "options": {"upColor": '#089981', "downColor": '#f23645', "borderVisible": False}
    }]
    
    if config['use_sma']:
        sma_d = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in df['SMA'].items() if not pd.isna(v)]
        series_main.append({"type": "Line", "data": sma_d, "options": {"color": "yellow", "lineWidth": 2, "title": "SMA Trend"}})

    # 交易標記
    markers = []
    if not trade_log.empty:
        for _, t in trade_log.iterrows():
            markers.append({
                "time": t['Date'].strftime('%Y-%m-%d'),
                "position": "belowBar" if t['Type']=='Buy' else "aboveBar",
                "color": "#089981" if t['Type']=='Buy' else "#f23645",
                "shape": "arrowUp" if t['Type']=='Buy' else "arrowDown",
                "text": "B" if t['Type']=='Buy' else "S"
            })
    series_main[0]["markers"] = markers
    
    charts = [{"chart": chart_options, "series": series_main}]
    
    # 副圖：VIX
    if config['use_vix']:
        vix_d = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in df['vix'].items()]
        charts.append({
            "chart": {"height": 150, "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"}}, 
            "series": [{"type": "Line", "data": vix_d, "options": {"color": "#ef5350", "title": "VIX Filter"}}]
        })
    
    # 副圖：RSI
    if config['use_rsi']:
        df['RSI'] = ta.rsi(df['Close'], length=14)
        rsi_d = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in df['RSI'].items() if not pd.isna(v)]
        charts.append({
            "chart": {"height": 150, "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"}}, 
            "series": [{"type": "Line", "data": rsi_d, "options": {"color": "#b2ebf2", "title": "RSI"}}]
        })

    renderLightweightCharts(charts, key="dark_chart")

    # D. 交易明細
    if not trade_log.empty:
        st.subheader("📋 交易日記")
        # 美化表格
        trade_log['Date'] = trade_log['Date'].dt.strftime('%Y-%m-%d')
        trade_log['Value'] = trade_log.get('Value', trade_log.get('Cost', 0)).fillna(0).abs().round(0)
        trade_log['Comm'] = trade_log['Comm'].round(2)
        st.dataframe(trade_log.style.applymap(lambda x: 'color: #089981' if x == 'Buy' else 'color: #f23645', subset=['Type']), use_container_width=True)
    else:
        st.warning("⚠️ 條件太嚴格，無交易產生。請嘗試調低 VIX 閥值或放寬 RSI。")

else:
    st.info("👈 請在左側設定參數，開始黑夜回測。")
