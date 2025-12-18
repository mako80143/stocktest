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
st.set_page_config(page_title="資產詳解回測 v19", layout="wide")
st.markdown("""
<style>
    header {visibility: hidden;}
    .block-container {padding-top: 0rem !important;}
    .stApp {background-color: #0e1117;}
    input {font-weight: bold; color: #00e676 !important;}
    /* 儀表板樣式 */
    div[data-testid="stMetric"] {background-color: #262730; border: 1px solid #464b5f; border-radius: 5px;}
    div[data-testid="stMetricLabel"] {color: #babcbf;}
    div[data-testid="stMetricValue"] {color: #ffffff;}
    
    /* 分隔線優化 */
    hr {margin-top: 0.5rem; margin-bottom: 0.5rem; border-color: #464b5f;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. Backtrader 策略 (加入損益計算)
# ==========================================
class DetailedStrategy(bt.Strategy):
    params = (('config', {}),)

    def __init__(self):
        super().__init__()
        self.dataclose = self.datas[0].close
        self.c = self.params.config
        self.order = None
        
        # 綁定 VIX
        self.vix = self.datas[0].vix if hasattr(self.datas[0], 'vix') else None
        
        # 紀錄表
        self.trade_list = []
        self.cash_history = []
        self.value_history = []
        
        # 損益統計
        self.total_realized_pnl = 0.0 # 累計已實現損益 (含手續費)

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
                'Reason': getattr(order.info, 'name', 'Signal')
            })
            self.order = None

    # 新增：監聽「平倉」事件，計算已實現損益
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        # pnlcomm 代表扣除手續費後的淨損益
        self.total_realized_pnl += trade.pnlcomm 

    def attempt_buy(self, pct, reason):
        if pct <= 0: return
        cash = self.broker.getcash()
        target_amount = cash * (pct / 100.0)
        target_amount = target_amount * 0.998 # 手續費緩衝
        size = int(target_amount / self.dataclose[0])
        if size > 0:
            self.order = self.buy(size=size, info={'name': reason})

    def attempt_sell(self, pct, reason):
        if pct <= 0: return
        pos_size = self.position.size
        if pos_size > 0:
            target_size = int(pos_size * (pct / 100.0))
            if target_size > 0:
                self.order = self.sell(size=target_size, info={'name': reason})

    def next(self):
        self.cash_history.append(self.broker.getcash())
        self.value_history.append(self.broker.getvalue())
        if self.order: return

        # === 邏輯區 (獨立觸發) ===
        
        # 1. VIX
        if self.c['use_vix'] and self.vix:
            if self.vix[0] > self.c['vix_buy_thres'] and self.vix[-1] <= self.c['vix_buy_thres']:
                self.attempt_buy(self.c['vix_buy_pct'], f"VIX>{int(self.c['vix_buy_thres'])}")
            if self.vix[0] < self.c['vix_sell_thres'] and self.vix[-1] >= self.c['vix_sell_thres']:
                self.attempt_sell(self.c['vix_sell_pct'], f"VIX<{int(self.c['vix_sell_thres'])}")

        # 2. EMA
        if self.c['use_ema']:
            if self.dataclose[0] > self.inds['ema'][0] and self.dataclose[-1] <= self.inds['ema'][-1]:
                self.attempt_buy(self.c['ema_buy_pct'], "EMA金叉")
            if self.dataclose[0] < self.inds['ema'][0] and self.dataclose[-1] >= self.inds['ema'][-1]:
                self.attempt_sell(self.c['ema_sell_pct'], "EMA死叉")

        # 3. MACD
        if self.c['use_macd']:
            if self.inds['macd'].macd[0] > self.inds['macd'].signal[0] and self.inds['macd'].macd[-1] <= self.inds['macd'].signal[-1]:
                self.attempt_buy(self.c['macd_buy_pct'], "MACD金叉")
            if self.inds['macd'].macd[0] < self.inds['macd'].signal[0] and self.inds['macd'].macd[-1] >= self.inds['macd'].signal[-1]:
                self.attempt_sell(self.c['macd_sell_pct'], "MACD死叉")

        # 4. RSI
        if self.c['use_rsi']:
            if self.inds['rsi'][0] < self.c['rsi_buy_val'] and self.inds['rsi'][-1] >= self.c['rsi_buy_val']:
                self.attempt_buy(self.c['rsi_buy_pct'], "RSI超賣")
            if self.inds['rsi'][0] > self.c['rsi_sell_val'] and self.inds['rsi'][-1] <= self.c['rsi_sell_val']:
                self.attempt_sell(self.c['rsi_sell_pct'], "RSI超買")

class PandasDataPlus(bt.feeds.PandasData):
    lines = ('vix',)
    params = (('vix', -1),)

# ==========================================
# 4. 側邊欄設定
# ==========================================
st.sidebar.header("🎛️ 參數設定")

with st.sidebar.expander("1. 資金與手續費", expanded=True):
    symbol = st.text_input("股票代碼", "NVDA")
    init_cash = st.number_input("初始本金", value=100000.0)
    comm_rate = st.number_input("手續費率 (%)", value=0.1425) / 100.0

# VIX
with st.sidebar.expander("2. VIX 設定", expanded=True):
    use_vix = st.checkbox("啟用 VIX", True)
    c1, c2 = st.columns(2)
    vix_buy_thres = c1.number_input("買入閥值 (>)", value=26.0)
    vix_buy_pct = c2.number_input("買入資金 %", value=100.0)
    c3, c4 = st.columns(2)
    vix_sell_thres = c3.number_input("賣出閥值 (<)", value=15.0)
    vix_sell_pct = c4.number_input("賣出持倉 %", value=100.0)

# 其他指標 (摺疊)
with st.sidebar.expander("3. 其他指標 (EMA/MACD/RSI)", expanded=False):
    use_ema = st.checkbox("啟用 EMA", True); ema_len = st.number_input("EMA 週期", 20); ema_buy_pct = st.number_input("EMA 買 %", 30.0); ema_sell_pct = st.number_input("EMA 賣 %", 50.0)
    st.divider()
    use_macd = st.checkbox("啟用 MACD", False); macd_buy_pct = st.number_input("MACD 買 %", 30.0); macd_sell_pct = st.number_input("MACD 賣 %", 50.0)
    macd_fast = 12; macd_slow = 26; macd_sig = 9
    st.divider()
    use_rsi = st.checkbox("啟用 RSI", False); rsi_len=14; rsi_buy_val=30; rsi_sell_val=70; rsi_buy_pct=30.0; rsi_sell_pct=50.0

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
btn_run = st.sidebar.button("🚀 執行詳細回測", type="primary")

# ==========================================
# 5. 主程式
# ==========================================
if btn_run:
    with st.spinner("計算損益結構中..."):
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
        cerebro.addstrategy(DetailedStrategy, config=config)
        cerebro.broker.setcash(init_cash)
        cerebro.broker.setcommission(commission=comm_rate)
        
        results = cerebro.run()
        strat = results[0]
        
        # 基礎計算
        final_equity = cerebro.broker.getvalue() # 總權益 (現金+市值)
        final_cash = cerebro.broker.getcash()    # 剩餘現金
        final_market_value = final_equity - final_cash # 持倉市值
        
        # 損益計算
        realized_pnl = strat.total_realized_pnl # 已實現 (包含手續費)
        
        # 未實現損益 = 持倉市值 - 持倉成本
        # 持倉成本較難精確獲取，我們可以用：總權益 - 初始本金 - 已實現損益
        # 推導： (Init + Realized + Unrealized) = Equity
        # 所以： Unrealized = Equity - Init - Realized
        unrealized_pnl = final_equity - init_cash - realized_pnl
        
        roi = (final_equity - init_cash) / init_cash * 100
        
        # 曲線與日誌
        idx = df.index[-len(strat.value_history):]
        equity_curve = pd.Series(strat.value_history, index=idx)
        cash_curve = pd.Series(strat.cash_history, index=idx)
        trade_log = pd.DataFrame(strat.trade_list)

    # UI 呈現
    st.title(f"🧾 {symbol} 資產負債詳解 (v19)")

    # 1. 資產總覽 (Row 1)
    st.subheader("💰 資產總覽")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1. 最終權益 (Total Equity)", f"${final_equity:,.0f}", f"{roi:.2f}%", help="= 現金 + 股票市值")
    c2.metric("2. 口袋現金 (Cash)", f"${final_cash:,.0f}", help="還沒買股票的錢")
    c3.metric("3. 股票市值 (Market Value)", f"${final_market_value:,.0f}", help="目前持倉值多少錢")
    
    # 顯示目前持倉股數
    pos_size = strat.position.size
    pos_price = strat.position.price
    c4.metric("目前持股", f"{pos_size} 股", f"均價 ${pos_price:.2f}" if pos_size>0 else "空手")

    st.markdown("---")

    # 2. 損益詳情 (Row 2)
    st.subheader("⚖️ 損益拆解")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("已實現損益 (Realized P&L)", f"${realized_pnl:,.0f}", help="真正賣出入袋的獲利 (扣除手續費後)")
    d2.metric("未實現損益 (Unrealized P&L)", f"${unrealized_pnl:,.0f}", help="目前持股的帳面浮動盈虧")
    
    # 簡單驗證算式
    check_val = init_cash + realized_pnl + unrealized_pnl
    d3.metric("驗證算式 (Init+R+U)", f"${check_val:,.0f}", help="應該要等於最終權益")
    d4.metric("交易總次數", len(trade_log))

    # 3. 資金結構圖
    st.markdown("---")
    st.subheader("📊 資金與市值消長")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 堆疊面積圖：現金 vs 市值
    fig.add_trace(go.Scatter(
        x=cash_curve.index, y=cash_curve.values, mode='lines', name='現金 (Cash)', 
        stackgroup='one', line=dict(width=0, color='rgba(200, 200, 200, 0.5)')
    ), secondary_y=False)
    
    # 市值 = 總權益 - 現金 (用算出來的)
    market_val_curve = equity_curve - cash_curve
    fig.add_trace(go.Scatter(
        x=market_val_curve.index, y=market_val_curve.values, mode='lines', name='股票市值 (Market Val)', 
        stackgroup='one', line=dict(width=0, color='rgba(0, 230, 118, 0.6)')
    ), secondary_y=False)
    
    # 總權益線
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name='總權益', line=dict(color='white', width=2)), secondary_y=False)

    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=450, title="資產配置變化圖 (灰色=現金, 綠色=股票)")
    st.plotly_chart(fig, use_container_width=True)

    # 4. K線圖
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

    chart_opts = {"layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"}, "height": 450}
    renderLightweightCharts([{"chart": chart_opts, "series": series_main}], key="v19_chart")

    # 5. 交易明細
    if not trade_log.empty:
        st.subheader("📋 交易日記")
        trade_log['Date'] = trade_log['Date'].dt.strftime('%Y-%m-%d')
        trade_log['Value'] = trade_log['Value'].abs().map('{:.0f}'.format)
        trade_log['Comm'] = trade_log['Comm'].map('{:.2f}'.format)
        st.dataframe(trade_log, use_container_width=True)
