import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import backtrader as bt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# --- 頁面設定 ---
st.set_page_config(page_title="股票回測與分析平台", layout="wide")

# ==========================================
# 1. Backtrader 策略類別 (核心邏輯)
# ==========================================
class GenericStrategy(bt.Strategy):
    """
    這是一個通用策略，可以根據使用者選擇的指標動態調整。
    目前範例：雙均線交叉 (Golden Cross)
    """
    params = (
        ('fast_period', 10),
        ('slow_period', 20),
        ('indicator_type', 'SMA'), # SMA 或 RSI
        ('rsi_period', 14),
        ('rsi_upper', 70),
        ('rsi_lower', 30),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # 建立指標
        if self.params.indicator_type == 'SMA':
            self.fast_ma = bt.indicators.SimpleMovingAverage(
                self.datas[0], period=self.params.fast_period)
            self.slow_ma = bt.indicators.SimpleMovingAverage(
                self.datas[0], period=self.params.slow_period)
            self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        
        elif self.params.indicator_type == 'RSI':
            self.rsi = bt.indicators.RSI(self.datas[0], period=self.params.rsi_period)

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        # 可以在這裡加入 st.write 來輸出日誌，但在回測中大量輸出會影響效能
        pass

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'買入執行: {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'賣出執行: {order.executed.price:.2f}')
            self.bar_executed = len(self)

        self.order = None

    def next(self):
        if self.order:
            return

        # --- 策略邏輯 ---
        if self.params.indicator_type == 'SMA':
            # 黃金交叉買入
            if not self.position:
                if self.crossover > 0:
                    self.buy()
            # 死亡交叉賣出
            elif self.crossover < 0:
                self.close()
        
        elif self.params.indicator_type == 'RSI':
            if not self.position:
                if self.rsi < self.params.rsi_lower:
                    self.buy()
            elif self.rsi > self.params.rsi_upper:
                self.close()

# ==========================================
# 2. 輔助功能：繪圖與數據下載
# ==========================================
def plot_candlestick(df, symbol, fast_ma=None, slow_ma=None, indicator_type='SMA'):
    """使用 Plotly 繪製互動式 K 線圖"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'{symbol} 價格走勢', '成交量'), 
                        row_width=[0.2, 0.7])

    # K線圖
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)

    # 疊加指標
    if indicator_type == 'SMA' and fast_ma is not None and slow_ma is not None:
        fig.add_trace(go.Scatter(x=df.index, y=df[f'SMA_{fast_ma}'], line=dict(color='orange', width=1), name=f'MA {fast_ma}'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df[f'SMA_{slow_ma}'], line=dict(color='blue', width=1), name=f'MA {slow_ma}'), row=1, col=1)

    # 成交量
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], showlegend=False, marker_color='rgba(128,128,128,0.5)'), row=2, col=1)

    # 介面優化
    fig.update_layout(
        title=f"{symbol} 技術分析圖表",
        yaxis_title='價格',
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

# ==========================================
# 3. Streamlit 主程式介面
# ==========================================
st.title("📈 智慧股票回測系統 v1.0")
st.markdown("輸入股票代碼與策略參數，立即查看回測績效與互動圖表。")

# --- 側邊欄：參數設定 ---
st.sidebar.header("1. 數據設定")
symbol = st.sidebar.text_input("股票代碼 (Yahoo Finance)", value="2330.TW")
start_date = st.sidebar.date_input("開始日期", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("結束日期", datetime.date.today())

st.sidebar.header("2. 資金與手續費")
cash = st.sidebar.number_input("初始資金", value=100000, step=10000)
commission = st.sidebar.number_input("手續費率 (例如 0.001425)", value=0.001425, step=0.0001, format="%.6f")

st.sidebar.header("3. 策略參數")
strategy_type = st.sidebar.selectbox("選擇策略指標", ["SMA (雙均線)", "RSI (相對強弱)"])

fast_ma_len = 0
slow_ma_len = 0
rsi_len = 14

if strategy_type == "SMA (雙均線)":
    fast_ma_len = st.sidebar.slider("短均線 (Fast MA)", 5, 50, 10)
    slow_ma_len = st.sidebar.slider("長均線 (Slow MA)", 10, 200, 20)
elif strategy_type == "RSI (相對強弱)":
    rsi_len = st.sidebar.slider("RSI 週期", 5, 30, 14)

# --- 主按鈕 ---
if st.button("🚀 開始回測"):
    with st.spinner('正在下載數據並執行策略...'):
        # 1. 下載數據
        try:
            df = yf.download(symbol, start=start_date, end=end_date)
            if df.empty:
                st.error("❌ 找不到數據，請檢查股票代碼或日期範圍。")
                st.stop()
            
            # 處理 MultiIndex (yfinance 新版問題)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # 2. 計算指標 (為了繪圖用，Backtrader 內部會自己再算一次)
            if strategy_type == "SMA (雙均線)":
                df.ta.sma(length=fast_ma_len, append=True)
                df.ta.sma(length=slow_ma_len, append=True)
            
            # 3. 設定 Backtrader
            cerebro = bt.Cerebro()
            
            # 加入數據
            data = bt.feeds.PandasData(dataname=df)
            cerebro.adddata(data)
            
            # 加入策略
            if strategy_type == "SMA (雙均線)":
                cerebro.addstrategy(GenericStrategy, 
                                    fast_period=fast_ma_len, 
                                    slow_period=slow_ma_len, 
                                    indicator_type='SMA')
            else:
                cerebro.addstrategy(GenericStrategy, 
                                    indicator_type='RSI',
                                    rsi_period=rsi_len)

            # 設定資金與手續費
            cerebro.broker.setcash(cash)
            cerebro.broker.setcommission(commission=commission)

            # 4. 執行回測
            start_value = cerebro.broker.getvalue()
            cerebro.run()
            end_value = cerebro.broker.getvalue()
            
            # 5. 計算績效
            total_return = (end_value - start_value) / start_value * 100
            market_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
            
            # --- 顯示結果區域 ---
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("最終資產", f"${end_value:,.0f}")
            col2.metric("策略報酬率", f"{total_return:.2f}%", delta_color="normal")
            col3.metric("大盤(買入持有)報酬", f"{market_return:.2f}%")
            col4.metric("交易成本/手續費", "已扣除")

            # --- 繪製互動圖表 ---
            st.subheader("📊 技術分析互動圖表")
            st.info("💡 提示：您可以使用滑鼠滾輪縮放圖表，或選取特定區域放大。")
            
            indicator_code = 'SMA' if 'SMA' in strategy_type else 'RSI'
            fig = plot_candlestick(df, symbol, fast_ma_len, slow_ma_len, indicator_code)
            st.plotly_chart(fig, use_container_width=True)

            # --- 交易紀錄 (從 analyzer 提取會更精確，這邊先做簡單版) ---
            st.subheader("📝 原始數據預覽")
            st.dataframe(df.tail())

        except Exception as e:
            st.error(f"發生錯誤：{e}")
            st.code(e)
