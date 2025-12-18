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
st.set_page_config(page_title="無限回測系統 v14", layout="wide")
st.markdown("""
<style>
    .stApp {background-color: #0e1117;}
    .block-container {padding-top: 1rem;}
    /* 輸入框優化 */
    input {font-weight: bold; color: #00e676 !important;}
    /* 儀表板樣式 */
    div[data-testid="stMetric"] {background-color: #262730; border: 1px solid #464b5f; border-radius: 5px;}
    div[data-testid="stMetricLabel"] {color: #babcbf;}
    div[data-testid="stMetricValue"] {color: #ffffff;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Backtrader 策略 (參數無限制版)
# ==========================================
class UnconstrainedStrategy(bt.Strategy):
    params = (('config', {}),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.c = self.params.config
        self.trade_list = []
        self.inds = {}
        
        # 1. VIX 數據 (如果有的話)
        self.vix = self.datas[0].vix if hasattr(self.datas[0], 'vix') else None

        # 2. 動態指標初始化 (讀取使用者輸入的任意數值)
        # SMA
        if self.c['use_sma']:
            self.inds['sma'] = bt.indicators.SMA(self.datas[0], period=int(self.c['sma_len']))
        
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

    def next(self):
        if self.order: return

        # --- A. 優先順序邏輯 ---
        can_trade = True
        
        # 1. 宏觀濾網 (VIX)
        if self.c['use_vix'] and self.vix:
            if self.c['vix_logic'] == '高於閥值不買 (避險)':
                if self.vix[0] > self.c['vix_thres']: can_trade = False
            elif self.c['vix_logic'] == '低於閥值不買 (抄底)':
                if self.vix[0] < self.c['vix_thres']: can_trade = False

        # 2. 趨勢濾網 (SMA)
        if can_trade and self.c['use_sma']:
            if self.dataclose[0] < self.inds['sma'][0]:
                can_trade = False

        # --- B. 進出場訊號 ---
        buy_sig = False
        sell_sig = False
        
        if can_trade:
            triggers = []
            if self.c['use_rsi']:
                triggers.append(self.inds['rsi'][0] < self.c['rsi_buy'])
            if self.c['use_macd']:
                triggers.append(self.inds['macd'].macd[0] > self.inds['macd'].signal[0])
            
            # AND 邏輯：有開啟的指標都必須符合
            if triggers and all(triggers):
                buy_sig = True

        # 出場 (OR 邏輯：任一條件滿足即賣)
        if self.position:
            if self.c['use_rsi'] and self.inds['rsi'][0] > self.c['rsi_sell']:
                sell_sig = True
            if self.c['use_sma'] and self.dataclose[0] < self.inds['sma'][0]:
                sell_sig = True

        # --- C. 執行交易 (資金無限制) ---
        if not self.position and buy_sig:
            cash = self.broker.getcash()
            
            # 判斷是「固定金額」還是「百分比」
            target_cash = 0
            if self.c['stake_type'] == '固定金額 (Fixed Cash)':
                target_cash = self.c['stake_val']
            else: # 百分比
                target_cash = cash * (self.c['stake_val'] / 100.0)
            
            # 計算股數 (不設限，除非現金真的不夠)
            size = int(target_cash / self.dataclose[0])
            
            # Backtrader 內建檢查：如果 size * price > cash，它會自動拒單
            # 但我們這裡不做額外限制，完全照你輸入的算
            if size > 0: self.buy(size=size)
            
        elif self.position and sell_sig:
            self.close()

# 用於傳遞 VIX 的資料格式
class PandasDataPlus(bt.feeds.PandasData):
    lines = ('vix',)
    params = (('vix', -1),)

# ==========================================
# 3. 側邊欄：自由輸入區
# ==========================================
st.sidebar.header("🎛️ 參數自由設定")

with st.sidebar.expander("1. 資金與手續費 (無限制)", expanded=True):
    symbol = st.text_input("股票代碼", "NVDA")
    init_cash = st.number_input("初始本金", value=100000.0, step=1000.0)
    
    # 手續費：開放高精度小數點
    comm_rate = st.number_input("手續費率 (%)", value=0.1425, format="%.6f", step=0.0001) / 100.0
    
    # 投入金額設定
    stake_type = st.radio("投入方式", ["資金百分比 (%)", "固定金額 (Fixed Cash)"])
    if stake_type == "資金百分比 (%)":
        stake_val = st.number_input("每次買入佔現金 %", value=100.0, step=10.0, help="可以設 100% 全倉梭哈")
    else:
        stake_val = st.number_input("每次買入金額 ($)", value=50000.0, step=1000.0)

with st.sidebar.expander("2. 宏觀 VIX 設定", expanded=True):
    use_vix = st.checkbox("啟用 VIX 濾網", True)
    vix_logic = st.selectbox("邏輯", ["高於閥值不買 (避險)", "低於閥值不買 (抄底)"])
    # 這裡不用 slider，改用 number_input，你想填 15.5 或 80 都可以
    vix_thres = st.number_input("VIX 閥值", value=30.0, step=1.0)

with st.sidebar.expander("3. 技術指標參數 (無限制)", expanded=True):
    # SMA
    use_sma = st.checkbox("SMA 均線", True)
    sma_len = st.number_input("SMA 週期", value=20, min_value=1, step=1)
    
    # RSI
    use_rsi = st.checkbox("RSI 指標", True)
    c1, c2, c3 = st.columns(3)
    rsi_len = c1.number_input("RSI 週期", value=14, min_value=2)
    rsi_buy = c2.number_input("買入 <", value=30)
    rsi_sell = c3.number_input("賣出 >", value=70)
    
    # MACD
    use_macd = st.checkbox("MACD 指標", False)
    m1, m2, m3 = st.columns(3)
    macd_fast = m1.number_input("快線", value=12)
    macd_slow = m2.number_input("慢線", value=26)
    macd_sig = m3.number_input("訊號", value=9)

# 打包參數
config = {
    'stake_type': stake_type, 'stake_val': stake_val,
    'use_vix': use_vix, 'vix_logic': vix_logic, 'vix_thres': vix_thres,
    'use_sma': use_sma, 'sma_len': sma_len,
    'use_rsi': use_rsi, 'rsi_len': rsi_len, 'rsi_buy': rsi_buy, 'rsi_sell': rsi_sell,
    'use_macd': use_macd, 'macd_fast': macd_fast, 'macd_slow': macd_slow, 'macd_sig': macd_sig
}

start_date = st.sidebar.date_input("開始日期", datetime.date(2023, 1, 1))
btn_run = st.sidebar.button("🚀 執行回測", type="primary")

# ==========================================
# 4. 主程式邏輯
# ==========================================
if btn_run:
    with st.spinner("正在運算..."):
        # 下載數據
        df = yf.download(symbol, start=start_date, end=datetime.date.today(), progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # 下載 VIX 並合併
        vix_df = yf.download("^VIX", start=start_date, end=datetime.date.today(), progress=False)
        if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.get_level_values(0)
        df['vix'] = vix_df['Close'].reindex(df.index).ffill()
        
        if df.empty:
            st.error("無數據")
            st.stop()

        # Backtrader 設置
        cerebro = bt.Cerebro()
        cerebro.adddata(PandasDataPlus(dataname=df))
        cerebro.addstrategy(UnconstrainedStrategy, config=config)
        cerebro.broker.setcash(init_cash)
        cerebro.broker.setcommission(commission=comm_rate)
        
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
        results = cerebro.run()
        strat = results[0]
        
        # 數據計算
        final_val = cerebro.broker.getvalue()
        roi = (final_val - init_cash) / init_cash * 100
        
        t_ret = strat.analyzers.timereturn.get_analysis()
        equity_curve = pd.Series(t_ret).fillna(0)
        equity_curve = (1 + equity_curve).cumprod() * init_cash
        
        # B&H
        bh_ret = df['Close'].pct_change().fillna(0)
        bh_curve = (1 + bh_ret).cumprod() * init_cash
        bh_roi = (bh_curve.iloc[-1] - init_cash) / init_cash * 100
        
        trade_log = pd.DataFrame(strat.trade_list)

    # ==========================================
    # 5. 黑夜版 UI 呈現
    # ==========================================
    st.title(f"🌑 {symbol} 無限回測戰報")
    
    # A. 績效儀表板
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最終權益", f"${final_val:,.0f}", f"{roi:.2f}%")
    c2.metric("Buy & Hold", f"${bh_curve.iloc[-1]:,.0f}", f"{bh_roi:.2f}%")
    c3.metric("Alpha", f"{roi - bh_roi:.2f}%")
    c4.metric("交易次數", len(trade_log) if not trade_log.empty else 0)

    # B. 獲利曲線 (Plotly Dark)
    st.subheader("📈 資金成長")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name='策略', line=dict(color='#00e676', width=2)))
    fig.add_trace(go.Scatter(x=bh_curve.index, y=bh_curve.values, mode='lines', name='B&H', line=dict(color='#555555', dash='dash')))
    
    if not trade_log.empty:
        buys = trade_log[trade_log['Type'] == 'Buy']
        sells = trade_log[trade_log['Type'] == 'Sell']
        fig.add_trace(go.Scatter(x=buys['Date'], y=equity_curve.loc[buys['Date']], mode='markers', name='買入', marker=dict(color='yellow', symbol='triangle-up', size=8)))
        fig.add_trace(go.Scatter(x=sells['Date'], y=equity_curve.loc[sells['Date']], mode='markers', name='賣出', marker=dict(color='red', symbol='triangle-down', size=8)))

    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # C. K線圖 (LWC Dark)
    st.subheader("🕯️ K 線與信號")
    kline_data = [{"time": i.strftime('%Y-%m-%d'), "open": r['Open'], "high": r['High'], "low": r['Low'], "close": r['Close']} for i, r in df.iterrows()]
    
    series_main = [{
        "type": 'Candlestick',
        "data": kline_data,
        "options": {"upColor": '#089981', "downColor": '#f23645', "borderVisible": False}
    }]
    
    if config['use_sma']:
        sma_vals = ta.sma(df['Close'], length=int(config['sma_len']))
        sma_d = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in sma_vals.items() if not pd.isna(v)]
        series_main.append({"type": "Line", "data": sma_d, "options": {"color": "yellow", "lineWidth": 2}})
    
    # 買賣標記
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

    # 圖表設定
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
        
    renderLightweightCharts(charts, key="final_chart")

    # D. 交易明細
    if not trade_log.empty:
        st.subheader("📋 交易日記 (含手續費)")
        trade_log['Date'] = trade_log['Date'].dt.strftime('%Y-%m-%d')
        trade_log['Comm'] = trade_log['Comm'].map('{:.2f}'.format)
        trade_log['Value'] = trade_log['Value'].abs().map('{:.0f}'.format)
        st.dataframe(trade_log, use_container_width=True)
