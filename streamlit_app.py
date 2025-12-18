import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import backtrader as bt
import datetime
from streamlit_lightweight_charts import renderLightweightCharts
import collections.abc
import numpy as np
from scipy.signal import argrelextrema
import gc
import warnings

# --- 1. 系統設定 ---
warnings.filterwarnings("ignore")
collections.Iterable = collections.abc.Iterable

st.set_page_config(page_title="VIX 戰情室 (無限版 v27)", layout="wide")

# CSS 優化：全黑化與滿版設定
st.markdown("""
<style>
    header {visibility: hidden;}
    .block-container {padding-top: 0rem !important; padding-bottom: 1rem !important;}
    .stApp {background-color: #0e1117;}
    /* 讓輸入框更明顯 */
    input {font-weight: bold; color: #00e676 !important;}
    /* 儀表板 */
    div[data-testid="stMetric"] {
        background-color: #1e2130; 
        border: 1px solid #2e3440; 
        border-radius: 8px; 
        padding: 15px;
    }
    div[data-testid="stMetricLabel"] {color: #a0aab9;}
    div[data-testid="stMetricValue"] {color: #ffffff; font-family: 'Roboto Mono', monospace;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 數據下載 (快取)
# ==========================================
@st.cache_data(ttl=3600)
def get_data(symbol, start):
    end = datetime.date.today()
    try:
        # 下載主數據
        df = yf.download(symbol, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty: return pd.DataFrame()

        df.index = df.index.tz_localize(None)
        
        # 下載 VIX
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

# ==========================================
# 3. 數學運算：上帝視角
# ==========================================
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
            shares = cash / price
            cash = 0
        elif i in max_idx and shares > 0:
            cash = shares * price
            shares = 0
            
        val = (shares * price) if shares > 0 else cash
        god_curve.append({"time": df.index[i].strftime('%Y-%m-%d'), "value": val})
        
    return god_curve

# ==========================================
# 4. Backtrader 策略
# ==========================================
class UnconstrainedStrategy(bt.Strategy):
    params = (('config', {}),)

    def __init__(self):
        super().__init__()
        self.dataclose = self.datas[0].close
        self.c = self.params.config
        self.vix = self.datas[0].vix if hasattr(self.datas[0], 'vix') else None
        
        self.trade_list = []
        self.inds = {}
        
        # 動態指標生成
        if self.c.get('use_ema'): 
            self.inds['ema'] = bt.indicators.EMA(self.datas[0], period=int(self.c['ema_len']))
        
        if self.c.get('use_macd'): 
            self.inds['macd'] = bt.indicators.MACD(self.datas[0], 
                                                   period_me1=int(self.c['macd_fast']), 
                                                   period_me2=int(self.c['macd_slow']), 
                                                   period_signal=int(self.c['macd_sig']))
                                                   
        if self.c.get('use_rsi'): 
            self.inds['rsi'] = bt.indicators.RSI(self.datas[0], period=int(self.c['rsi_len']))

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.trade_list.append({
                'Date': bt.num2date(order.executed.dt),
                'Type': 'Buy' if order.isbuy() else 'Sell',
                'Price': order.executed.price,
                'Value': order.executed.value,
                'Reason': getattr(order.info, 'name', 'Signal')
            })

    def attempt_buy(self, pct, reason):
        cash = self.broker.getcash()
        if cash < 100: return
        
        # 不限制 pct 上限，如果使用者輸入 200%，這裡就會算出 2倍資金
        # 但因為我們沒有開融資(Margin)，Backtrader 會自動買到現金用完為止 (Max Cash)
        target = cash * (pct / 100.0) * 0.998
        size = int(target / self.dataclose[0])
        
        if size > 0: self.buy(size=size, info={'name': reason})

    def attempt_sell(self, pct, reason):
        size = self.position.size
        if size > 0:
            target = int(size * (pct / 100.0))
            if target > 0: self.sell(size=target, info={'name': reason})

    def next(self):
        # 1. VIX
        if self.c.get('use_vix') and self.vix:
            if self.vix[0] > self.c['vix_b_thres']:
                # 這裡不做 CrossOver 檢查，改為只要大於就嘗試買
                # 這樣如果你設定買入 20%，VIX 連續三天高檔，他就會連續買三天 (加碼)
                self.attempt_buy(self.c['vix_b_pct'], f"VIX>{self.c['vix_b_thres']}")
            
            if self.vix[0] < self.c['vix_s_thres']:
                self.attempt_sell(self.c['vix_s_pct'], f"VIX<{self.c['vix_s_thres']}")

        # 2. EMA
        if self.c.get('use_ema'):
            # 只要價格在 EMA 上方就嘗試買 (若沒錢就不會成交，若有錢就加碼)
            if self.dataclose[0] > self.inds['ema'][0]:
                self.attempt_buy(self.c['ema_b_pct'], "Price>EMA")
            elif self.dataclose[0] < self.inds['ema'][0]:
                self.attempt_sell(self.c['ema_s_pct'], "Price<EMA")

        # 3. MACD
        if self.c.get('use_macd'):
            if self.inds['macd'].macd[0] > self.inds['macd'].signal[0]:
                self.attempt_buy(self.c['macd_b_pct'], "MACD Gold")
            elif self.inds['macd'].macd[0] < self.inds['macd'].signal[0]:
                self.attempt_sell(self.c['macd_s_pct'], "MACD Dead")

        # 4. RSI
        if self.c.get('use_rsi'):
            if self.inds['rsi'][0] < self.c['rsi_b_val']:
                self.attempt_buy(self.c['rsi_b_pct'], f"RSI<{self.c['rsi_b_val']}")
            elif self.inds['rsi'][0] > self.c['rsi_s_val']:
                self.attempt_sell(self.c['rsi_s_pct'], f"RSI>{self.c['rsi_s_val']}")

class PandasDataPlus(bt.feeds.PandasData):
    lines = ('vix',)
    params = (('vix', -1),)

# ==========================================
# 5. 無限自由輸入區
# ==========================================
st.sidebar.header("🎛️ 無限參數設定")

symbol = st.sidebar.text_input("股票代碼", "NVDA")
start_date = st.sidebar.date_input("開始日期", datetime.date(2023, 1, 1))
# 這裡使用 number_input 並且不設 min/max，讓你可以隨意輸入
init_cash = st.sidebar.number_input("初始本金", value=100000.0, step=1000.0)

with st.sidebar.expander("1. VIX 策略", expanded=True):
    use_vix = st.checkbox("啟用 VIX", True)
    c1, c2 = st.columns(2)
    # step=0.1 允許小數點，無 min/max 限制
    vix_b_thres = c1.number_input("VIX 買入閥值 (>)", value=26.0, step=0.1, format="%.2f") 
    vix_b_pct = c2.number_input("VIX 買入資金 %", value=100.0, step=10.0) 
    
    c3, c4 = st.columns(2)
    vix_s_thres = c3.number_input("VIX 賣出閥值 (<)", value=14.0, step=0.1, format="%.2f")
    vix_s_pct = c4.number_input("VIX 賣出持倉 %", value=100.0, step=10.0)

with st.sidebar.expander("2. 技術指標 (無限制)", expanded=False):
    # EMA
    use_ema = st.checkbox("啟用 EMA", True)
    ema_len = st.number_input("EMA 週期", value=20, min_value=1) # 週期不能負數
    c1, c2 = st.columns(2)
    ema_b_pct = c1.number_input("EMA 買 %", value=30.0)
    ema_s_pct = c2.number_input("EMA 賣 %", value=50.0)
    
    st.divider()
    # MACD
    use_macd = st.checkbox("啟用 MACD", False)
    m1, m2, m3 = st.columns(3)
    macd_fast = m1.number_input("快線", value=12)
    macd_slow = m2.number_input("慢線", value=26)
    macd_sig = m3.number_input("訊號", value=9)
    c3, c4 = st.columns(2)
    macd_b_pct = c3.number_input("MACD 買 %", value=30.0)
    macd_s_pct = c4.number_input("MACD 賣 %", value=50.0)

    st.divider()
    # RSI
    use_rsi = st.checkbox("啟用 RSI", False)
    rsi_len = st.number_input("RSI 週期", value=14, min_value=2)
    c5, c6 = st.columns(2)
    rsi_b_val = c5.number_input("RSI 買入 <", value=30.0)
    rsi_b_pct = c6.number_input("RSI 買入 %", value=30.0)
    c7, c8 = st.columns(2)
    rsi_s_val = c7.number_input("RSI 賣出 >", value=70.0)
    rsi_s_pct = c8.number_input("RSI 賣出 %", value=50.0)

config = {
    'use_vix': use_vix, 'vix_b_thres': vix_b_thres, 'vix_b_pct': vix_b_pct, 'vix_s_thres': vix_s_thres, 'vix_s_pct': vix_s_pct,
    'use_ema': use_ema, 'ema_len': ema_len, 'ema_b_pct': ema_b_pct, 'ema_s_pct': ema_s_pct,
    'use_macd': use_macd, 'macd_fast': macd_fast, 'macd_slow': macd_slow, 'macd_sig': macd_sig, 'macd_b_pct': macd_b_pct, 'macd_s_pct': macd_s_pct,
    'use_rsi': use_rsi, 'rsi_len': rsi_len, 'rsi_b_val': rsi_b_val, 'rsi_b_pct': rsi_b_pct, 'rsi_s_val': rsi_s_val, 'rsi_s_pct': rsi_s_pct
}

btn = st.sidebar.button("🔥 執行回測", type="primary")

# ==========================================
# 6. 主程式
# ==========================================
if btn:
    gc.collect()
    with st.spinner("運算中..."):
        df = get_data(symbol, start_date)
        if df.empty:
            st.error("無數據"); st.stop()

        # 1. 上帝視角數據
        god_data = calculate_god_mode(df, init_cash)
        god_final = god_data[-1]['value'] if god_data else init_cash

        # 2. Buy & Hold 數據
        initial_price = df['Close'].iloc[0]
        bh_series = (df['Close'] / initial_price) * init_cash
        bh_data = [{"time": t.strftime('%Y-%m-%d'), "value": v} for t, v in bh_series.items()]
        bh_final = bh_series.iloc[-1]

        # 3. 執行回測
        cerebro = bt.Cerebro()
        cerebro.adddata(PandasDataPlus(dataname=df))
        cerebro.addstrategy(UnconstrainedStrategy, config=config)
        cerebro.broker.setcash(init_cash)
        cerebro.broker.setcommission(commission=0.001425)
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
        
        results = cerebro.run()
        strat = results[0]
        final_val = cerebro.broker.getvalue()
        
        # 4. 策略數據
        t_ret = strat.analyzers.timereturn.get_analysis()
        eq_series = pd.Series(t_ret).fillna(0)
        eq_series = (1 + eq_series).cumprod() * init_cash
        eq_data = [{"time": t.strftime('%Y-%m-%d'), "value": v} for t, v in eq_series.items()]
        
        trade_log = pd.DataFrame(strat.trade_list)

    # === UI 顯示 ===
    st.title(f"🚀 {symbol} 無限自由戰情室")

    # A. 績效
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("😇 上帝視角", f"${god_final:,.0f}")
    c2.metric("😈 我的策略", f"${final_val:,.0f}")
    c3.metric("😴 Buy & Hold", f"${bh_final:,.0f}")
    c4.metric("交易次數", len(trade_log))

    # B. 曲線圖 (Canvas)
    st.subheader("📈 資金大對決")
    equity_chart = {
        "chart": {"height": 400, "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"}},
        "series": [
            {"type": "Line", "data": god_data, "options": {"color": "#FFD700", "lineWidth": 2, "title": "上帝視角"}},
            {"type": "Line", "data": eq_data, "options": {"color": "#00E676", "lineWidth": 2, "title": "我的策略"}},
            {"type": "Line", "data": bh_data, "options": {"color": "#787B86", "lineWidth": 1, "lineStyle": 2, "title": "B&H"}}
        ]
    }
    renderLightweightCharts([equity_chart], key="eq_chart")

    # C. K線與訊號 (Canvas)
    st.subheader("🕯️ K 線與買賣點")
    kline_data = [{"time": i.strftime('%Y-%m-%d'), "open": r['Open'], "high": r['High'], "low": r['Low'], "close": r['Close']} for i, r in df.iterrows()]
    series_main = [{"type": 'Candlestick', "data": kline_data, "options": {"upColor": '#089981', "downColor": '#f23645', "borderVisible": False}}]
    
    # 疊加 EMA
    if config['use_ema']:
        ema_vals = ta.ema(df['Close'], length=int(config['ema_len']))
        ema_d = [{"time": i.strftime('%Y-%m-%d'), "value": v} for i, v in ema_vals.items() if not pd.isna(v)]
        series_main.append({"type": "Line", "data": ema_d, "options": {"color": "#FFA726", "lineWidth": 2, "title": "EMA"}})

    # 標記
    if not trade_log.empty:
        markers = []
        for _, t in trade_log.iterrows():
            txt = "B" if t['Type']=='Buy' else "S"
            if "VIX" in str(t['Reason']): txt = "V"
            markers.append({
                "time": t['Date'].strftime('%Y-%m-%d'),
                "position": "belowBar" if t['Type']=='Buy' else "aboveBar",
                "color": "#00E676" if t['Type']=='Buy' else "#FF5252",
                "shape": "arrowUp" if t['Type']=='Buy' else "arrowDown",
                "text": txt
            })
        series_main[0]["markers"] = markers

    renderLightweightCharts([{"chart": {"height": 450, "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"}}, "series": series_main}], key="k_chart")
    
    # VIX 副圖
    vix_data = [{"time": i.strftime('%Y-%m-%d'), "value": v} for i, v in df['vix'].items()]
    renderLightweightCharts([{"chart": {"height": 150, "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"}}, "series": [{"type": "Line", "data": vix_data, "options": {"color": "#EF5350", "title": "VIX"}}]}], key="v_chart")

    # D. 日誌
    if not trade_log.empty:
        st.subheader("📋 交易日記")
        trade_log['Date'] = trade_log['Date'].dt.strftime('%Y-%m-%d')
        trade_log['Value'] = trade_log['Value'].abs().map('{:.0f}'.format)
        st.dataframe(trade_log, use_container_width=True)
