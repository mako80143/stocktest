import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta  # 用於圖表數據計算
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

st.set_page_config(page_title="DMS 2.0 策略實驗室", layout="wide")

# CSS 優化 (保持您原本的極簡暗黑風格)
st.markdown("""
<style>
    header {visibility: hidden;}
    .block-container {padding-top: 0rem !important; padding-bottom: 1rem !important;}
    .stApp {background-color: #0e1117;}
    input {font-weight: bold; color: #00e676 !important;}
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
# 2. 數據下載
# ==========================================
@st.cache_data(ttl=3600)
def get_data(symbol, start):
    end = datetime.date.today()
    try:
        # 下載數據
        df = yf.download(symbol, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty: return pd.DataFrame()

        df.index = df.index.tz_localize(None)
        return df
    except:
        return pd.DataFrame()

# ==========================================
# 3. 數學運算：上帝視角 (God Mode)
# ==========================================
def calculate_god_mode(df, init_cash):
    data = df['Close'].values
    # 尋找局部極值
    min_idx = argrelextrema(data, np.less, order=5)[0]
    max_idx = argrelextrema(data, np.greater, order=5)[0]
    
    cash = init_cash
    shares = 0
    god_curve = []
    
    for i in range(len(df)):
        price = data[i]
        
        # 簡單的波段低買高賣模擬
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
# 4. Backtrader 策略：DMS 2.0 核心
# ==========================================
class DMSStrategy(bt.Strategy):
    params = (('config', {}),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.c = self.params.config
        
        self.trade_list = []
        self.cash_history = []
        self.value_history = []
        
        # --- DMS 關鍵指標定義 ---
        # 1. 移動平均
        self.ma20 = bt.indicators.SMA(self.datas[0], period=self.c['ma_support'])
        self.ma50 = bt.indicators.SMA(self.datas[0], period=self.c['ma_trend'])
        
        # 2. ROC 動能 (Rate of Change)
        self.roc = bt.indicators.ROC(self.datas[0], period=self.c['roc_period'])
        self.roc_ma = bt.indicators.SMA(self.roc, period=self.c['roc_ma_period'])
        
        # 3. ADX 趨勢強度
        # Backtrader 的 ADX 內建 DI+ 和 DI-
        self.adx = bt.indicators.ADX(self.datas[0], period=self.c['adx_period'])
        self.di_plus = bt.indicators.PlusDI(self.datas[0], period=self.c['adx_period'])
        self.di_minus = bt.indicators.MinusDI(self.datas[0], period=self.c['adx_period'])

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.trade_list.append({
                'Date': bt.num2date(order.executed.dt),
                'Type': 'Buy' if order.isbuy() else 'Sell',
                'Price': order.executed.price,
                'Value': order.executed.value,
                'Reason': getattr(order.info, 'name', 'Signal')
            })

    def next(self):
        # 記錄資金曲線
        self.cash_history.append(self.broker.getcash())
        self.value_history.append(self.broker.getvalue())
        
        # 避免數據不足時報錯
        if len(self) < 100: return

        # 取出參數
        adx_thres = self.c['adx_threshold']
        adx_strong_thres = self.c['adx_strong']

        # --- 策略邏輯 ---
        
        # 狀態判斷
        is_uptrend = (self.dataclose[0] > self.ma50[0])           # 收盤在 MA50 之上
        is_adx_strong = (self.adx[0] > adx_thres)                 # ADX 大於門檻 (20)
        is_bullish_di = (self.di_plus[0] > self.di_minus[0])      # DI+ > DI-
        is_momentum_acc = (self.roc[0] > self.roc_ma[0])          # ROC 上穿均線 (加速中)
        
        # 1. 進場條件 (Buy Signal)
        # 邏輯：趨勢向上 + 動能強勁 + 加速確認
        if not self.position:
            if is_uptrend and is_adx_strong and is_bullish_di and is_momentum_acc:
                # 全倉買入 (保留一點緩衝)
                cash = self.broker.getcash()
                size = int((cash * 0.99) / self.dataclose[0])
                if size > 0:
                    self.buy(size=size, info={'name': 'DMS_Entry'})

        # 2. 出場條件 (Sell Signal)
        else:
            # A. 止損/趨勢破壞：跌破 MA20
            trend_broken = (self.dataclose[0] < self.ma20[0])
            
            # B. 動能衰竭：ADX 高檔轉折向下 且 ROC 減速
            # ADX 前一天 > 25 (強趨勢) 且 今天 < 前一天
            adx_turning_down = (self.adx[-1] > adx_strong_thres) and (self.adx[0] < self.adx[-1])
            momentum_lost = (self.roc[0] < self.roc_ma[0])
            
            exhaustion = adx_turning_down and momentum_lost
            
            if trend_broken:
                self.close(info={'name': 'Stop_MA20'})
            elif exhaustion:
                self.close(info={'name': 'Exhaustion'})

# ==========================================
# 5. 控制台 (Sidebar)
# ==========================================
st.sidebar.header("🦁 DMS 2.0 策略實驗室")
symbol = st.sidebar.text_input("股票代碼", "NVDA")
start_date = st.sidebar.date_input("回測開始", datetime.date(2023, 1, 1))
init_cash = st.sidebar.number_input("初始本金", value=10000.0, step=1000.0)

st.sidebar.subheader("⚙️ 策略參數設定")
ma_support = st.sidebar.number_input("防守線 (MA Short)", 20)
ma_trend = st.sidebar.number_input("趨勢線 (MA Trend)", 50)
roc_period = st.sidebar.number_input("動能週期 (ROC)", 12)
roc_ma_period = st.sidebar.number_input("動能平滑 (ROC MA)", 6)
adx_period = st.sidebar.number_input("強度週期 (ADX)", 14)
adx_threshold = st.sidebar.number_input("趨勢門檻 (ADX Min)", 20)
adx_strong = st.sidebar.number_input("強趨勢門檻 (ADX High)", 25)

config = {
    'ma_support': ma_support,
    'ma_trend': ma_trend,
    'roc_period': roc_period,
    'roc_ma_period': roc_ma_period,
    'adx_period': adx_period,
    'adx_threshold': adx_threshold,
    'adx_strong': adx_strong
}

btn = st.sidebar.button("🔥 執行 DMS 回測", type="primary")

# ==========================================
# 6. 主程式執行
# ==========================================
if btn:
    gc.collect()
    with st.spinner(f"正在模擬 {symbol} 的 DMS 策略表現..."):
        df = get_data(symbol, start_date)
        if df.empty: st.error("無數據，請檢查代碼"); st.stop()

        # A. 計算上帝視角 & B&H
        god_data = calculate_god_mode(df, init_cash)
        god_final = god_data[-1]['value'] if god_data else init_cash
        
        initial_price = df['Close'].iloc[0]
        bh_series = (df['Close'] / initial_price) * init_cash
        bh_data = [{"time": t.strftime('%Y-%m-%d'), "value": v} for t, v in bh_series.items()]
        bh_final = bh_series.iloc[-1]

        # B. 執行 Backtrader 回測
        cerebro = bt.Cerebro()
        cerebro.adddata(bt.feeds.PandasData(dataname=df))
        cerebro.addstrategy(DMSStrategy, config=config)
        cerebro.broker.setcash(init_cash)
        # 設定手續費 (模擬美股券商少量費用或滑價)
        cerebro.broker.setcommission(commission=0.001) 
        
        results = cerebro.run()
        strat = results[0]
        final_val = cerebro.broker.getvalue()
        
        # C. 處理圖表數據
        dates = df.index[-len(strat.value_history):]
        
        # 權益曲線
        eq_data = [{"time": d.strftime('%Y-%m-%d'), "value": v} for d, v in zip(dates, strat.value_history)]
        # 現金水位
        cash_data = [{"time": d.strftime('%Y-%m-%d'), "value": v} for d, v in zip(dates, strat.cash_history)]
        
        trade_log = pd.DataFrame(strat.trade_list)

    # === UI 顯示 ===
    st.title(f"🚀 {symbol} DMS 策略戰報")

    # 1. 績效看板
    ret_strat = ((final_val - init_cash) / init_cash) * 100
    ret_bh = ((bh_final - init_cash) / init_cash) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("😇 上帝視角", f"${god_final:,.0f}")
    c2.metric("🦁 DMS 策略", f"${final_val:,.0f}", delta=f"{ret_strat:.1f}%")
    c3.metric("😴 Buy & Hold", f"${bh_final:,.0f}", delta=f"{ret_bh:.1f}%")
    c4.metric("目前倉位", "空手 (現金)" if strat.cash_history[-1] > final_val*0.9 else "持有股票")

    # 2. 資產成長曲線 (Benchmark)
    st.subheader("📈 總資產對決")
    equity_chart = {
        "chart": {"height": 350, "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"}},
        "series": [
            {"type": "Line", "data": god_data, "options": {"color": "#FFD700", "lineWidth": 1, "lineStyle": 2, "title": "上帝視角"}},
            {"type": "Line", "data": eq_data, "options": {"color": "#00E676", "lineWidth": 3, "title": "DMS 策略"}},
            {"type": "Line", "data": bh_data, "options": {"color": "#787B86", "lineWidth": 1, "title": "B&H"}}
        ]
    }
    renderLightweightCharts([equity_chart], key="eq_chart")

    # 3. 現金透視 (Druckenmiller 風格：看不懂就縮手)
    st.subheader("💰 資金控管 (現金水位)")
    cash_chart = {
        "chart": {"height": 200, "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"}},
        "series": [
            {"type": "Area", "data": eq_data, "options": {"lineColor": "rgba(0, 0, 0, 0)", "topColor": "rgba(0, 230, 118, 0.1)", "bottomColor": "rgba(0, 230, 118, 0.1)", "title": "總資產底色"}},
            {"type": "Area", "data": cash_data, "options": {"lineColor": "#2962FF", "topColor": "rgba(41, 98, 255, 0.4)", "bottomColor": "rgba(41, 98, 255, 0.0)", "title": "持有現金 (Cash)"}}
        ]
    }
    renderLightweightCharts([cash_chart], key="cash_chart")

    # 4. K線與交易點位
    st.subheader("🕯️ 交易訊號還原")
    
    # 計算繪圖用的 MA (使用 pandas_ta 方便繪圖，與回測邏輯分離但數值一致)
    df['MA20_Plot'] = ta.sma(df['Close'], length=config['ma_support'])
    df['MA50_Plot'] = ta.sma(df['Close'], length=config['ma_trend'])
    
    kline_data = [{"time": i.strftime('%Y-%m-%d'), "open": r['Open'], "high": r['High'], "low": r['Low'], "close": r['Close']} for i, r in df.iterrows()]
    ma20_data = [{"time": i.strftime('%Y-%m-%d'), "value": v} for i, v in df['MA20_Plot'].items() if not pd.isna(v)]
    ma50_data = [{"time": i.strftime('%Y-%m-%d'), "value": v} for i, v in df['MA50_Plot'].items() if not pd.isna(v)]

    series_main = [
        {"type": 'Candlestick', "data": kline_data, "options": {"upColor": '#089981', "downColor": '#f23645', "borderVisible": False}},
        {"type": "Line", "data": ma20_data, "options": {"color": "#FF5252", "lineWidth": 1, "title": "MA20 (防守)"}},
        {"type": "Line", "data": ma50_data, "options": {"color": "#2962FF", "lineWidth": 2, "title": "MA50 (趨勢)"}}
    ]

    # 標記買賣點
    if not trade_log.empty:
        markers = []
        for _, t in trade_log.iterrows():
            is_buy = t['Type'] == 'Buy'
            markers.append({
                "time": t['Date'].strftime('%Y-%m-%d'),
                "position": "belowBar" if is_buy else "aboveBar",
                "color": "#00E676" if is_buy else "#FF5252",
                "shape": "arrowUp" if is_buy else "arrowDown",
                "text": f"{t['Reason']}"
            })
        series_main[0]["markers"] = markers

    renderLightweightCharts([{"chart": {"height": 500, "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"}}, "series": series_main}], key="k_chart")

    # 5. 交易明細表
    if not trade_log.empty:
        st.subheader("📋 交易日記")
        
        # 格式化表格
        display_log = trade_log.copy()
        display_log['Date'] = display_log['Date'].dt.strftime('%Y-%m-%d')
        display_log['Price'] = display_log['Price'].map('${:,.2f}'.format)
        display_log['Value'] = display_log['Value'].abs().map('${:,.0f}'.format)
        
        # 使用顏色標記買賣
        def highlight_row(row):
            return ['background-color: rgba(0, 230, 118, 0.1)'] * len(row) if row['Type'] == 'Buy' else ['background-color: rgba(255, 82, 82, 0.1)'] * len(row)
        
        st.dataframe(display_log.style.apply(highlight_row, axis=1), use_container_width=True)
    else:
        st.info("這段期間沒有觸發任何交易訊號。")
