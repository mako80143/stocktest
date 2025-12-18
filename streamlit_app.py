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

# 2. 頁面與黑夜模式設定
st.set_page_config(page_title="VIX 戰法 v16", layout="wide")
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
# 3. Backtrader 策略核心 (VIX 主導 + 指標輔助)
# ==========================================
class VixStrategy(bt.Strategy):
    params = (('config', {}),)

    def __init__(self):
        super().__init__()
        self.dataclose = self.datas[0].close
        self.c = self.params.config
        self.order = None
        self.trade_list = []
        
        # 綁定 VIX
        self.vix = self.datas[0].vix if hasattr(self.datas[0], 'vix') else None

        # --- 指標初始化 ---
        self.inds = {}
        
        # EMA (新增設定)
        if self.c['use_ema']:
            self.inds['ema'] = bt.indicators.EMA(self.datas[0], period=int(self.c['ema_len']))
            
        # MACD (新增設定)
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
            self.order = None

    def next(self):
        if self.order: return

        # =========================
        # 🟢 買入條件檢查 (AND)
        # 邏輯：VIX > 26 (恐慌) 且 技術面轉強/未轉弱
        # =========================
        buy_signal = True # 預設為 True，有一項不過就變 False
        
        # 1. VIX 條件 (必須符合)
        if self.vix:
            if self.vix[0] < self.c['vix_buy_thres']: # 如果 VIX 不夠高 (例如 < 26)
                buy_signal = False
        
        # 2. EMA 條件 (第二確認)
        if self.c['use_ema']:
            # 價格必須在 EMA 之上才買 (多頭確認)
            if self.dataclose[0] < self.inds['ema'][0]:
                buy_signal = False
                
        # 3. MACD 條件 (第二確認)
        if self.c['use_macd']:
            # 快線必須 > 訊號線 (金叉狀態)
            if self.inds['macd'].macd[0] < self.inds['macd'].signal[0]:
                buy_signal = False

        # =========================
        # 🔴 賣出條件檢查 (OR)
        # 邏輯：VIX < 13 (安逸) 或 技術面轉弱
        # =========================
        sell_signal = False
        
        if self.position:
            # 1. VIX 賣出 (安逸過頭)
            if self.vix and self.vix[0] < self.c['vix_sell_thres']:
                sell_signal = True # VIX < 13 賣出
            
            # 2. EMA 賣出 (跌破均線)
            if self.c['use_ema'] and self.dataclose[0] < self.inds['ema'][0]:
                sell_signal = True
                
            # 3. MACD 賣出 (死叉)
            if self.c['use_macd'] and self.inds['macd'].macd[0] < self.inds['macd'].signal[0]:
                sell_signal = True

        # =========================
        # ⚡ 執行交易
        # =========================
        
        # 買入 (使用設定的資金 %)
        if not self.position and buy_signal:
            cash = self.broker.getcash()
            # 計算投入金額
            target_cash = cash * (self.c['stake_pct'] / 100.0)
            size = int(target_cash / self.dataclose[0])
            
            if size > 0:
                self.order = self.buy(size=size)
        
        # 賣出 (清倉)
        elif self.position and sell_signal:
            self.order = self.close() # close() 預設就是全賣

class PandasDataPlus(bt.feeds.PandasData):
    lines = ('vix',)
    params = (('vix', -1),)

# ==========================================
# 4. 側邊欄：VIX 戰法設定
# ==========================================
st.sidebar.header("🎛️ VIX 戰法控制台")

with st.sidebar.expander("1. 資金與標的", expanded=True):
    symbol = st.text_input("股票代碼", "NVDA")
    init_cash = st.number_input("初始本金", value=100000.0)
    comm_rate = st.number_input("手續費率 (%)", value=0.1425, step=0.0001) / 100.0
    stake_pct = st.number_input("買入投入資金 %", value=100.0, max_value=100.0, help="100代表全倉買進")

with st.sidebar.expander("2. VIX 核心參數", expanded=True):
    st.info("買入邏輯：VIX > 買入閥值 (恐慌)\n賣出邏輯：VIX < 賣出閥值 (安逸)")
    vix_buy_thres = st.number_input("VIX 買入閥值 (>)", value=26.0, step=1.0)
    vix_sell_thres = st.number_input("VIX 賣出閥值 (<)", value=13.0, step=1.0)

with st.sidebar.expander("3. 第二條件 (技術指標)", expanded=True):
    st.caption("勾選後，必須同時符合 VIX 與指標才會買進")
    
    # EMA
    use_ema = st.checkbox("EMA 均線", True)
    ema_len = st.number_input("EMA 週期", value=20)
    
    # MACD
    st.divider()
    use_macd = st.checkbox("MACD 動能", False)
    c1, c2, c3 = st.columns(3)
    macd_fast = c1.number_input("快", 12)
    macd_slow = c2.number_input("慢", 26)
    macd_sig = c3.number_input("訊號", 9)

config = {
    'stake_pct': stake_pct,
    'vix_buy_thres': vix_buy_thres, 'vix_sell_thres': vix_sell_thres,
    'use_ema': use_ema, 'ema_len': ema_len,
    'use_macd': use_macd, 'macd_fast': macd_fast, 'macd_slow': macd_slow, 'macd_sig': macd_sig
}

start_date = st.sidebar.date_input("開始日期", datetime.date(2022, 1, 1))
btn_run = st.sidebar.button("🚀 執行回測", type="primary")

# ==========================================
# 5. 主程式
# ==========================================
if btn_run:
    with st.spinner("正在計算 VIX 與指標策略..."):
        # 下載數據
        df = yf.download(symbol, start=start_date, end=datetime.date.today(), progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        vix_df = yf.download("^VIX", start=start_date, end=datetime.date.today(), progress=False)
        if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.get_level_values(0)
        df['vix'] = vix_df['Close'].reindex(df.index).ffill() # 對齊

        if df.empty:
            st.error("無數據")
            st.stop()

        # 執行 Backtrader
        cerebro = bt.Cerebro()
        cerebro.adddata(PandasDataPlus(dataname=df))
        cerebro.addstrategy(VixStrategy, config=config)
        cerebro.broker.setcash(init_cash)
        cerebro.broker.setcommission(commission=comm_rate)
        
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
        results = cerebro.run()
        strat = results[0]
        
        # 績效處理
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
    st.title(f"🛡️ {symbol} VIX 戰法報告")

    # 1. 績效看板
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最終權益", f"${final_val:,.0f}", f"{roi:.2f}%")
    c2.metric("Buy & Hold", f"${bh_curve.iloc[-1]:,.0f}", f"{bh_roi:.2f}%")
    c3.metric("Alpha", f"{roi - bh_roi:.2f}%")
    c4.metric("交易次數", len(trade_log) if not trade_log.empty else 0)

    # 2. 獲利曲線
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

    # 3. K線圖 + VIX
    st.subheader("🕯️ K線與指標")
    
    # 預算 EMA 供繪圖
    if config['use_ema']:
        df['EMA'] = ta.ema(df['Close'], length=int(config['ema_len']))

    kline_data = [{"time": i.strftime('%Y-%m-%d'), "open": r['Open'], "high": r['High'], "low": r['Low'], "close": r['Close']} for i, r in df.iterrows()]
    
    series_main = [{
        "type": 'Candlestick',
        "data": kline_data,
        "options": {"upColor": '#089981', "downColor": '#f23645', "borderVisible": False}
    }]
    
    if config['use_ema']:
        ema_d = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in df['EMA'].items() if not pd.isna(v)]
        series_main.append({"type": "Line", "data": ema_d, "options": {"color": "orange", "lineWidth": 2, "title": "EMA"}})

    # 標記
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
    vix_d = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in df['vix'].items()]
    # 畫出 26 和 13 的參考線 (雖 LWC 不支援固定水平線，但我們可以 visually 看數值)
    charts.append({
        "chart": {**chart_opts, "height": 150},
        "series": [{"type": "Line", "data": vix_d, "options": {"color": "#ef5350", "title": "VIX"}}]
    })

    renderLightweightCharts(charts, key="v16_chart")

    # 4. 明細
    if not trade_log.empty:
        st.subheader("📋 交易日記")
        trade_log['Date'] = trade_log['Date'].dt.strftime('%Y-%m-%d')
        trade_log['Value'] = trade_log['Value'].abs().map('{:.0f}'.format)
        trade_log['Comm'] = trade_log['Comm'].map('{:.2f}'.format)
        st.dataframe(trade_log, use_container_width=True)
    else:
        st.warning("在此設定下未觸發任何交易 (VIX 條件可能太嚴格)。")
