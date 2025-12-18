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

# 1. 兼容性修復
collections.Iterable = collections.abc.Iterable

# 2. 頁面設定
st.set_page_config(page_title="真實資金流回測 v18", layout="wide")
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
# 3. Backtrader 策略 (資金流監控版)
# ==========================================
class CashFlowStrategy(bt.Strategy):
    params = (('config', {}),)

    def __init__(self):
        super().__init__()
        self.dataclose = self.datas[0].close
        self.c = self.params.config
        self.order = None
        
        # 紀錄表
        self.trade_list = []      # 成功交易
        self.skipped_list = []    # 資金不足被略過的交易
        self.cash_history = []    # 每日現金水位
        self.value_history = []   # 每日總資產
        
        # 綁定 VIX
        self.vix = self.datas[0].vix if hasattr(self.datas[0], 'vix') else None

        # 指標
        self.inds = {}
        if self.c['use_ema']:
            self.inds['ema'] = bt.indicators.EMA(self.datas[0], period=int(self.c['ema_len']))
        if self.c['use_macd']:
            self.inds['macd'] = bt.indicators.MACD(self.datas[0], 
                                                   period_me1=int(self.c['macd_fast']), 
                                                   period_me2=int(self.c['macd_slow']), 
                                                   period_signal=int(self.c['macd_sig']))
        if self.c['use_rsi']:
            self.inds['rsi'] = bt.indicators.RSI(self.datas[0], period=int(self.c['rsi_len']))

    def notify_order(self, order):
        if order.status in [order.Completed]:
            action = 'Buy' if order.isbuy() else 'Sell'
            self.trade_list.append({
                'Date': bt.num2date(order.executed.dt),
                'Type': action,
                'Price': order.executed.price,
                'Size': order.executed.size,
                'Value': order.executed.value,
                'Comm': order.executed.comm,
                'Cash_Left': self.broker.getcash(), # 交易後剩多少錢
                'Reason': getattr(order.info, 'name', 'Signal')
            })
            self.order = None

    def attempt_buy(self, pct, reason):
        """ 嘗試買入：檢查資金是否足夠 """
        if pct <= 0: return
        
        cash = self.broker.getcash()
        target_amount = cash * (pct / 100.0)
        
        # 估算手續費緩衝 (避免買滿後付不出手續費被拒單)
        # 假設手續費最高 0.2%
        target_amount = target_amount * 0.998 
        
        size = int(target_amount / self.dataclose[0])
        
        if size > 0:
            self.buy(size=size, info={'name': reason})
        else:
            # 紀錄：想買但沒錢
            self.skipped_list.append({
                'Date': self.datas[0].datetime.date(0),
                'Type': 'No Cash',
                'Reason': reason,
                'Current_Cash': cash,
                'Price': self.dataclose[0]
            })

    def attempt_sell(self, pct, reason):
        """ 嘗試賣出 """
        if pct <= 0: return
        pos_size = self.position.size
        if pos_size > 0:
            target_size = int(pos_size * (pct / 100.0))
            if target_size > 0:
                self.sell(size=target_size, info={'name': reason})

    def next(self):
        # 紀錄每日資產狀態 (這是驗證資金流的關鍵)
        self.cash_history.append(self.broker.getcash())
        self.value_history.append(self.broker.getvalue())

        if self.order: return

        # =======================
        # 1. VIX 邏輯
        # =======================
        if self.c['use_vix'] and self.vix:
            # 買入訊號
            if self.vix[0] > self.c['vix_buy_thres'] and self.vix[-1] <= self.c['vix_buy_thres']:
                self.attempt_buy(self.c['vix_buy_pct'], f"VIX>{int(self.c['vix_buy_thres'])}")
            
            # 賣出訊號
            if self.vix[0] < self.c['vix_sell_thres'] and self.vix[-1] >= self.c['vix_sell_thres']:
                self.attempt_sell(self.c['vix_sell_pct'], f"VIX<{int(self.c['vix_sell_thres'])}")

        # =======================
        # 2. EMA 邏輯
        # =======================
        if self.c['use_ema']:
            # 金叉買
            if self.dataclose[0] > self.inds['ema'][0] and self.dataclose[-1] <= self.inds['ema'][-1]:
                self.attempt_buy(self.c['ema_buy_pct'], "EMA金叉")
            # 死叉賣
            if self.dataclose[0] < self.inds['ema'][0] and self.dataclose[-1] >= self.inds['ema'][-1]:
                self.attempt_sell(self.c['ema_sell_pct'], "EMA死叉")

        # =======================
        # 3. MACD 邏輯
        # =======================
        if self.c['use_macd']:
            # 金叉買
            if self.inds['macd'].macd[0] > self.inds['macd'].signal[0] and self.inds['macd'].macd[-1] <= self.inds['macd'].signal[-1]:
                self.attempt_buy(self.c['macd_buy_pct'], "MACD金叉")
            # 死叉賣
            if self.inds['macd'].macd[0] < self.inds['macd'].signal[0] and self.inds['macd'].macd[-1] >= self.inds['macd'].signal[-1]:
                self.attempt_sell(self.c['macd_sell_pct'], "MACD死叉")
                
        # =======================
        # 4. RSI 邏輯
        # =======================
        if self.c['use_rsi']:
            # 超賣買
            if self.inds['rsi'][0] < self.c['rsi_buy_val'] and self.inds['rsi'][-1] >= self.c['rsi_buy_val']:
                self.attempt_buy(self.c['rsi_buy_pct'], "RSI超賣")
            # 超買賣
            if self.inds['rsi'][0] > self.c['rsi_sell_val'] and self.inds['rsi'][-1] <= self.c['rsi_sell_val']:
                self.attempt_sell(self.c['rsi_sell_pct'], "RSI超買")

class PandasDataPlus(bt.feeds.PandasData):
    lines = ('vix',)
    params = (('vix', -1),)

# ==========================================
# 4. 側邊欄：獨立設定
# ==========================================
st.sidebar.header("🎛️ 資金流回測系統")

with st.sidebar.expander("1. 初始設定", expanded=True):
    symbol = st.text_input("股票代碼", "NVDA")
    init_cash = st.number_input("初始本金", value=100000.0)
    comm_rate = st.number_input("手續費率 (%)", value=0.1425) / 100.0

# VIX
with st.sidebar.expander("2. VIX 設定", expanded=True):
    use_vix = st.checkbox("啟用 VIX", True)
    c1, c2 = st.columns(2)
    vix_buy_thres = c1.number_input("買入閥值 (>)", value=26.0)
    vix_buy_pct = c2.number_input("買入資金 %", value=100.0, help="設100就是全梭，沒錢就不能再買其他指標")
    
    c3, c4 = st.columns(2)
    vix_sell_thres = c3.number_input("賣出閥值 (<)", value=13.0)
    vix_sell_pct = c4.number_input("賣出持倉 %", value=100.0)

# EMA
with st.sidebar.expander("3. EMA 設定", expanded=False):
    use_ema = st.checkbox("啟用 EMA", True)
    ema_len = st.number_input("週期", value=20)
    c1, c2 = st.columns(2)
    ema_buy_pct = c1.number_input("EMA 買入 %", value=30.0)
    ema_sell_pct = c2.number_input("EMA 賣出 %", value=50.0)

# MACD
with st.sidebar.expander("4. MACD 設定", expanded=False):
    use_macd = st.checkbox("啟用 MACD", False)
    m1, m2, m3 = st.columns(3)
    macd_fast = m1.number_input("快", 12); macd_slow = m2.number_input("慢", 26); macd_sig = m3.number_input("訊", 9)
    c1, c2 = st.columns(2)
    macd_buy_pct = c1.number_input("MACD 買入 %", value=30.0)
    macd_sell_pct = c2.number_input("MACD 賣出 %", value=50.0)

# RSI
with st.sidebar.expander("5. RSI 設定", expanded=False):
    use_rsi = st.checkbox("啟用 RSI", False)
    rsi_len = st.number_input("週期", 14)
    c1, c2 = st.columns(2)
    rsi_buy_val = c1.number_input("< 多少買", 30)
    rsi_buy_pct = c2.number_input("RSI 買入 %", 30.0)
    c3, c4 = st.columns(2)
    rsi_sell_val = c3.number_input("> 多少賣", 70)
    rsi_sell_pct = c4.number_input("RSI 賣出 %", 50.0)

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
btn_run = st.sidebar.button("🚀 執行真實資金回測", type="primary")

# ==========================================
# 5. 執行
# ==========================================
if btn_run:
    with st.spinner("運算中 (含資金流模擬)..."):
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
        cerebro.addstrategy(CashFlowStrategy, config=config)
        cerebro.broker.setcash(init_cash)
        cerebro.broker.setcommission(commission=comm_rate)
        
        results = cerebro.run()
        strat = results[0]
        
        final_val = cerebro.broker.getvalue()
        roi = (final_val - init_cash) / init_cash * 100
        
        # 提取每日現金與總資產
        # 注意：backtrader 的長度可能跟原始 df 有落差 (因為指標預熱 period)，這裡做對齊
        idx = df.index[-len(strat.value_history):]
        equity_curve = pd.Series(strat.value_history, index=idx)
        cash_curve = pd.Series(strat.cash_history, index=idx)
        
        bh_ret = df['Close'].pct_change().fillna(0)
        bh_curve = (1 + bh_ret).cumprod() * init_cash
        bh_roi = (bh_curve.iloc[-1] - init_cash) / init_cash * 100
        
        trade_log = pd.DataFrame(strat.trade_list)
        skipped_log = pd.DataFrame(strat.skipped_list)

    # UI 呈現
    st.title(f"🛡️ {symbol} 資金流戰報")
    
    # 1. 績效
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最終權益", f"${final_val:,.0f}", f"{roi:.2f}%")
    c2.metric("剩餘現金", f"${cash_curve.iloc[-1]:,.0f}", help="還沒買股票的閒錢")
    c3.metric("Buy & Hold", f"${bh_curve.iloc[-1]:,.0f}", f"{bh_roi:.2f}%")
    c4.metric("交易次數", f"{len(trade_log)} (失敗: {len(skipped_log)})")

    # 2. 資金結構圖 (重要新增)
    st.subheader("💰 資金結構分析 (現金 vs 持倉)")
    
    # 雙軸圖：總權益 + 現金水位
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 總權益 (綠線)
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name='總資產 (Equity)', line=dict(color='#00e676', width=2)), secondary_y=False)
    # 現金 (灰填滿)
    fig.add_trace(go.Scatter(x=cash_curve.index, y=cash_curve.values, mode='lines', name='現金水位 (Cash)', fill='tozeroy', line=dict(color='rgba(255, 255, 255, 0.2)', width=1)), secondary_y=False)
    
    # B&H 對照
    fig.add_trace(go.Scatter(x=bh_curve.index, y=bh_curve.values, mode='lines', name='Buy & Hold', line=dict(color='#555555', dash='dash')), secondary_y=False)

    # 標記
    if not trade_log.empty:
        buys = trade_log[trade_log['Type'] == 'Buy']
        sells = trade_log[trade_log['Type'] == 'Sell']
        fig.add_trace(go.Scatter(x=buys['Date'], y=equity_curve.loc[buys['Date']], mode='markers', name='買入', marker=dict(color='yellow', symbol='triangle-up', size=8)), secondary_y=False)
        fig.add_trace(go.Scatter(x=sells['Date'], y=equity_curve.loc[sells['Date']], mode='markers', name='賣出', marker=dict(color='red', symbol='triangle-down', size=8)), secondary_y=False)

    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=450, title="資產成長與現金消耗圖")
    st.plotly_chart(fig, use_container_width=True)

    # 3. K線圖
    st.subheader("🕯️ 交易訊號")
    kline_data = [{"time": i.strftime('%Y-%m-%d'), "open": r['Open'], "high": r['High'], "low": r['Low'], "close": r['Close']} for i, r in df.iterrows()]
    series_main = [{"type": 'Candlestick', "data": kline_data, "options": {"upColor": '#089981', "downColor": '#f23645', "borderVisible": False}}]
    
    if config['use_ema']:
        df['EMA'] = ta.ema(df['Close'], length=int(config['ema_len']))
        ema_d = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in df['EMA'].items() if not pd.isna(v)]
        series_main.append({"type": "Line", "data": ema_d, "options": {"color": "orange", "lineWidth": 2}})

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

    chart_opts = {"layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"}, "height": 450}
    renderLightweightCharts([{"chart": chart_opts, "series": series_main}], key="v18_chart")

    # 4. 日誌區 (成功 vs 失敗)
    c_log1, c_log2 = st.columns(2)
    with c_log1:
        st.subheader("✅ 成功交易")
        if not trade_log.empty:
            trade_log['Date'] = trade_log['Date'].dt.strftime('%Y-%m-%d')
            trade_log['Value'] = trade_log['Value'].abs().map('{:.0f}'.format)
            trade_log['Cash_Left'] = trade_log['Cash_Left'].map('{:.0f}'.format)
            st.dataframe(trade_log[['Date', 'Type', 'Price', 'Size', 'Value', 'Cash_Left', 'Reason']], use_container_width=True)
        else:
            st.info("無交易")

    with c_log2:
        st.subheader("🚫 資金不足 (Skipped)")
        if not skipped_log.empty:
            st.caption("以下訊號觸發時，因現金不足而未執行：")
            skipped_log['Date'] = skipped_log['Date'].astype(str)
            skipped_log['Current_Cash'] = skipped_log['Current_Cash'].map('{:.0f}'.format)
            skipped_log['Price'] = skipped_log['Price'].map('{:.2f}'.format)
            st.dataframe(skipped_log[['Date', 'Reason', 'Current_Cash', 'Price']], use_container_width=True)
        else:
            st.success("資金充裕，所有訊號皆成功執行！")
