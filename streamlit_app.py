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
st.set_page_config(page_title="VIX 強力修復版 v21", layout="wide")
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
# 3. 數據下載與清洗函數 (關鍵修復)
# ==========================================
def get_clean_data(symbol, start, end):
    # 下載主數據
    df = yf.download(symbol, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    # ⚠️ 強制移除時區，避免對不齊
    df.index = df.index.tz_localize(None) 
    
    # 下載 VIX
    vix_df = yf.download("^VIX", start=start, end=end, progress=False)
    if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.get_level_values(0)
    # ⚠️ 強制移除時區
    vix_df.index = vix_df.index.tz_localize(None)
    
    # 合併數據 (Left Join 確保以股票交易日為主)
    # 使用 merge 而不是 reindex，更穩健
    df['vix'] = vix_df['Close']
    
    # 補值 (如果某天股票有開盤但 VIX 沒數據，用前一天的補)
    df['vix'] = df['vix'].ffill()
    
    return df

# ==========================================
# 4. Backtrader 策略
# ==========================================
class RobustStrategy(bt.Strategy):
    params = (('config', {}),)

    def __init__(self):
        super().__init__()
        self.dataclose = self.datas[0].close
        self.c = self.params.config
        self.order = None
        self.trade_list = []
        self.skipped_list = [] # 紀錄失敗原因
        
        # 綁定數據
        self.vix = self.datas[0].vix
        
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

    def attempt_buy(self, pct, reason):
        if pct <= 0: return
        cash = self.broker.getcash()
        
        # 如果現金太少 (<100)，直接不執行，避免報錯
        if cash < 100: 
            self.skipped_list.append({'Date': self.datas[0].datetime.date(0), 'Reason': f"{reason} (沒錢)"})
            return

        target_amount = cash * (pct / 100.0) * 0.998
        size = int(target_amount / self.dataclose[0])
        
        if size > 0:
            self.order = self.buy(size=size, info={'name': reason})
        else:
            self.skipped_list.append({'Date': self.datas[0].datetime.date(0), 'Reason': f"{reason} (股價太高買不起)"})

    def attempt_sell(self, pct, reason):
        if pct <= 0: return
        pos_size = self.position.size
        if pos_size > 0:
            target_size = int(pos_size * (pct / 100.0))
            if target_size > 0:
                self.order = self.sell(size=target_size, info={'name': reason})

    def next(self):
        if self.order: return

        # === 1. VIX 邏輯 (狀態檢查 State Check) ===
        # 只要 VIX 高於設定值，且還沒滿倉(透過資金比例控制)，就嘗試買入
        # 為了避免每天都買，Backtrader 預設只有在「有現金」時才會真的成交
        if self.c['use_vix']:
            # 買入：只要 VIX 高於閥值
            if self.vix[0] > self.c['vix_buy_thres']:
                # 這裡加一個小濾網：如果昨天也大於閥值，就不重複觸發 (CrossOver)，除非你想要連續買
                # 但為了確保"有買到"，我們改成：只要大於閥值 且 昨天小於閥值 (標準突破)
                if self.vix[-1] <= self.c['vix_buy_thres']:
                    self.attempt_buy(self.c['vix_buy_pct'], f"VIX>{int(self.c['vix_buy_thres'])}")
            
            # 賣出：只要 VIX 低於閥值
            if self.vix[0] < self.c['vix_sell_thres']:
                if self.vix[-1] >= self.c['vix_sell_thres']:
                    self.attempt_sell(self.c['vix_sell_pct'], f"VIX<{int(self.c['vix_sell_thres'])}")

        # === 2. EMA 邏輯 ===
        if self.c['use_ema']:
            if self.dataclose[0] > self.inds['ema'][0] and self.dataclose[-1] <= self.inds['ema'][-1]:
                self.attempt_buy(self.c['ema_buy_pct'], "EMA金叉")
            if self.dataclose[0] < self.inds['ema'][0] and self.dataclose[-1] >= self.inds['ema'][-1]:
                self.attempt_sell(self.c['ema_sell_pct'], "EMA死叉")

        # === 3. MACD 邏輯 ===
        if self.c['use_macd']:
            if self.inds['macd'].macd[0] > self.inds['macd'].signal[0] and self.inds['macd'].macd[-1] <= self.inds['macd'].signal[-1]:
                self.attempt_buy(self.c['macd_buy_pct'], "MACD金叉")
            if self.inds['macd'].macd[0] < self.inds['macd'].signal[0] and self.inds['macd'].macd[-1] >= self.inds['macd'].signal[-1]:
                self.attempt_sell(self.c['macd_sell_pct'], "MACD死叉")

        # === 4. RSI 邏輯 ===
        if self.c['use_rsi']:
            if self.inds['rsi'][0] < self.c['rsi_buy_val'] and self.inds['rsi'][-1] >= self.c['rsi_buy_val']:
                self.attempt_buy(self.c['rsi_buy_pct'], "RSI超賣")
            if self.inds['rsi'][0] > self.c['rsi_sell_val'] and self.inds['rsi'][-1] <= self.c['rsi_sell_val']:
                self.attempt_sell(self.c['rsi_sell_pct'], "RSI超買")

class PandasDataPlus(bt.feeds.PandasData):
    lines = ('vix',)
    params = (('vix', -1),)

# ==========================================
# 5. 側邊欄設定
# ==========================================
st.sidebar.header("🎛️ 參數設定")

with st.sidebar.expander("1. 資金與手續費", expanded=True):
    symbol = st.text_input("股票代碼", "NVDA")
    init_cash = st.number_input("初始本金", value=100000.0)
    comm_rate = st.number_input("手續費率 (%)", value=0.1425) / 100.0

# VIX
with st.sidebar.expander("2. VIX 設定 (必填)", expanded=True):
    use_vix = st.checkbox("啟用 VIX", True)
    c1, c2 = st.columns(2)
    vix_buy_thres = c1.number_input("買入閥值 (>)", value=30.0)
    vix_buy_pct = c2.number_input("買入資金 %", value=100.0)
    c3, c4 = st.columns(2)
    vix_sell_thres = c3.number_input("賣出閥值 (<)", value=15.0)
    vix_sell_pct = c4.number_input("賣出持倉 %", value=100.0)

# 其他指標
with st.sidebar.expander("3. 其他指標", expanded=False):
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

start_date = st.sidebar.date_input("開始日期", datetime.date(2022, 1, 1))
btn_run = st.sidebar.button("🚀 執行修復版回測", type="primary")

# ==========================================
# 6. 主程式
# ==========================================
if btn_run:
    with st.spinner("數據下載與校正中..."):
        # 1. 獲取清洗後的數據
        df = get_clean_data(symbol, start_date, datetime.date.today())
        
        if df.empty:
            st.error("❌ 無數據，請檢查股票代碼。")
            st.stop()
            
        # 2. 強制計算 Buy & Hold 曲線 (獨立於策略)
        # 假設第一天開盤就買
        initial_close = df['Close'].iloc[0]
        bh_shares = init_cash / initial_close
        bh_curve = df['Close'] * bh_shares
        
        # 3. 數據檢測：VIX 是否有超過閥值？
        vix_max = df['vix'].max()
        if config['use_vix'] and vix_max < config['vix_buy_thres']:
            st.warning(f"⚠️ **VIX 警告：** 此期間 VIX 最高只有 **{vix_max:.2f}**，未達到您設定的 **{config['vix_buy_thres']}**，因此不會觸發 VIX 買入。")
        
        # 4. 執行 Backtrader
        cerebro = bt.Cerebro()
        cerebro.adddata(PandasDataPlus(dataname=df))
        cerebro.addstrategy(RobustStrategy, config=config)
        cerebro.broker.setcash(init_cash)
        cerebro.broker.setcommission(commission=comm_rate)
        
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
        results = cerebro.run()
        strat = results[0]
        
        # 5. 整理結果
        t_ret = strat.analyzers.timereturn.get_analysis()
        equity_curve = pd.Series(t_ret).fillna(0)
        equity_curve = (1 + equity_curve).cumprod() * init_cash
        
        # 最終數值
        final_val = cerebro.broker.getvalue()
        roi = (final_val - init_cash) / init_cash * 100
        bh_final = bh_curve.iloc[-1]
        bh_roi = (bh_final - init_cash) / init_cash * 100
        
        trade_log = pd.DataFrame(strat.trade_list)
        skipped_log = pd.DataFrame(strat.skipped_list)

    # UI 呈現
    st.title(f"🛡️ {symbol} 戰報 (v21 數據修復)")

    # 1. VIX 數據表 (讓使用者眼見為憑)
    if config['use_vix']:
        with st.expander("📊 查看 VIX 觸發紀錄 (檢查數據是否存在)", expanded=False):
            high_vix_days = df[df['vix'] > config['vix_buy_thres']][['vix']]
            if not high_vix_days.empty:
                st.success(f"✅ 共有 {len(high_vix_days)} 天 VIX 高於 {config['vix_buy_thres']}：")
                # 格式化日期
                high_vix_days.index = high_vix_days.index.strftime('%Y-%m-%d')
                st.dataframe(high_vix_days.tail(10)) # 只顯示最後10筆
            else:
                st.error(f"❌ 沒有任何一天 VIX 高於 {config['vix_buy_thres']}！")

    # 2. 績效看板
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("策略最終權益", f"${final_val:,.0f}", f"{roi:.2f}%")
    c2.metric("Buy & Hold", f"${bh_final:,.0f}", f"{bh_roi:.2f}%")
    c3.metric("Alpha", f"{roi - bh_roi:.2f}%")
    c4.metric("交易次數", len(trade_log))

    # 3. 資金曲線圖
    st.subheader("📈 資金成長")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name='策略', line=dict(color='#00e676', width=2)))
    # 使用我們自己算好的 bh_curve
    fig.add_trace(go.Scatter(x=bh_curve.index, y=bh_curve.values, mode='lines', name='B&H', line=dict(color='#555555', dash='dash')))
    
    if not trade_log.empty:
        buys = trade_log[trade_log['Type'] == 'Buy']
        sells = trade_log[trade_log['Type'] == 'Sell']
        fig.add_trace(go.Scatter(x=buys['Date'], y=equity_curve.loc[buys['Date']], mode='markers', name='買入', marker=dict(color='yellow', symbol='triangle-up', size=8)))
        fig.add_trace(go.Scatter(x=sells['Date'], y=equity_curve.loc[sells['Date']], mode='markers', name='賣出', marker=dict(color='red', symbol='triangle-down', size=8)))

    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
    st.plotly_chart(fig, use_container_width=True)

    # 4. K線圖
    st.subheader("🕯️ K線與指標")
    kline_data = [{"time": i.strftime('%Y-%m-%d'), "open": r['Open'], "high": r['High'], "low": r['Low'], "close": r['Close']} for i, r in df.iterrows()]
    series_main = [{"type": 'Candlestick', "data": kline_data, "options": {"upColor": '#089981', "downColor": '#f23645', "borderVisible": False}}]
    
    if config['use_ema']:
        df['EMA'] = ta.ema(df['Close'], length=int(config['ema_len']))
        ema_d = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in df['EMA'].items() if not pd.isna(v)]
        series_main.append({"type": "Line", "data": ema_d, "options": {"color": "orange", "lineWidth": 2}})
    
    # 標記
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
    renderLightweightCharts([{"chart": chart_opts, "series": series_main}], key="v21_chart")

    # 5. 明細
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
        st.subheader("🚫 未成交紀錄 (Skipped)")
        if not skipped_log.empty:
            skipped_log['Date'] = skipped_log['Date'].astype(str)
            st.dataframe(skipped_log, use_container_width=True)
        else:
            st.info("無失敗紀錄")
