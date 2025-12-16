import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import backtrader as bt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# --- 頁面設定 ---
st.set_page_config(page_title="專業級策略回測系統 v3", layout="wide")

# ==========================================
# 1. Backtrader 策略引擎 (核心邏輯)
# ==========================================
class AdvancedStrategy(bt.Strategy):
    """
    v3 策略：支援動態組裝、修正 VIX 恐慌買入、強制停損優先級
    """
    params = (
        ('strategy_params', {}), # 用字典傳入所有 UI 設定
        ('stop_loss_pct', 0.10),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.comm = None
        
        # 輔助數據 (VIX) - 只有當 datas 長度大於 1 時才讀取
        self.vix = self.datas[1].close if len(self.datas) > 1 else None

        # --- 指標預計算 (根據 UI 參數) ---
        p = self.params.strategy_params
        
        # 1. 移動平均線 (MA)
        self.ma_fast = bt.indicators.SimpleMovingAverage(self.datas[0], period=p.get('fast_len', 10))
        self.ma_slow = bt.indicators.SimpleMovingAverage(self.datas[0], period=p.get('slow_len', 50))
        self.crossover = bt.indicators.CrossOver(self.ma_fast, self.ma_slow)
        
        # 2. RSI
        self.rsi = bt.indicators.RSI(self.datas[0], period=p.get('rsi_len', 14))
        
        # 3. KD (Stochastic) - 引入更多指標
        self.stoch_k, self.stoch_d = bt.indicators.Stochastic(self.datas[0], period=p.get('kd_len', 14)).lines
        
        # --- 績效分析器 (用於繪製淨值曲線) ---
        bt.Cerebro.addanalyzer(self, bt.analyzers.TimeReturn, _name='timereturn')


    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buyprice = order.executed.price
            self.bar_executed = len(self)
        self.order = None

    def next(self):
        if self.order:
            return

        # 1. --- 風險管理：強制出場優先級最高 ---
        if self.position:
            # 停損 (Stop Loss) 檢查
            cost_price = self.position.price
            current_price = self.dataclose[0]
            pct_change = (current_price - cost_price) / cost_price
            if pct_change < -self.params.stop_loss_pct:
                self.close()
                self.log(f'🛑 停損出場: {pct_change:.2%}')
                return # 停損優先級最高，直接結束本次循環

        # 2. --- 進場條件檢查 ---
        if not self.position:
            p = self.params.strategy_params
            buy_signal = True 
            
            # --- A. 宏觀濾網 (Druckenmiller VIX 修正) ---
            if p.get('use_vix_filter', False) and self.vix is not None:
                # VIX 恐慌時 (VIX 高漲) 允許買入；VIX 平靜時 (VIX 低迷) 暫停買入
                if p.get('vix_logic', 'buy_on_panic') == 'avoid_flat' and self.vix[0] < p.get('vix_threshold', 30):
                    buy_signal = False # VIX 太低，市場過熱，不買
                elif p.get('vix_logic', 'buy_on_panic') == 'buy_on_panic' and self.vix[0] < p.get('vix_threshold', 30):
                    buy_signal = False # VIX 未達恐慌線，不買
            
            # --- B. 技術指標條件 (使用 AND 邏輯，需全部滿足) ---
            
            # MA 交叉
            if p.get('use_ma_cross', False):
                if not (p.get('ma_buy_crossover', True) and self.crossover > 0):
                    buy_signal = False
            
            # RSI 超賣
            if p.get('use_rsi_signal', False):
                if not (self.rsi < p.get('rsi_buy', 30)):
                    buy_signal = False
            
            # KD 超賣
            if p.get('use_kd_signal', False):
                 if not (self.stoch_k < p.get('kd_buy', 20)):
                    buy_signal = False
            
            # --- C. 執行買入 ---
            if buy_signal:
                self.buy()

        # 3. --- 出場條件檢查 (非停損情況下的策略出場) ---
        else:
            p = self.params.strategy_params
            sell_signal = False 

            # MA 交叉出場
            if p.get('use_ma_cross', False) and p.get('ma_sell_crossunder', True):
                if self.crossover < 0:
                    sell_signal = True
            
            # RSI 超買出場
            if p.get('use_rsi_signal', False):
                if self.rsi > p.get('rsi_sell', 70):
                    sell_signal = True
            
            # KD 超買出場
            if p.get('use_kd_signal', False):
                if self.stoch_k > p.get('kd_sell', 80):
                    sell_signal = True

            # 執行賣出
            if sell_signal:
                self.close()

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        # print(f'{dt.isoformat()}, {txt}') # 可以在此處將交易日誌存入 Streamlit 變量

# ==========================================
# 2. 輔助功能：繪圖 (Plotly 互動圖表)
# ==========================================
def plot_results(df_stock, symbol, df_bench, equity_curve):
    """繪製 K線、指標與淨值曲線"""
    
    # 計算大盤累積報酬 (基準)
    bench_cumulative = (1 + df_bench['Close'].pct_change()).fillna(1).cumprod()
    bench_cumulative = bench_cumulative / bench_cumulative.iloc[0] * 100000 
    
    # 建立子圖：K線, 淨值曲線
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        subplot_titles=(f'{symbol} 股價與指標', '策略 vs 大盤 淨值曲線 (資產總值)'), 
                        row_width=[0.5, 0.5])

    # --- 1. K線圖 ---
    fig.add_trace(go.Candlestick(x=df_stock.index,
                                 open=df_stock['Open'], high=df_stock['High'],
                                 low=df_stock['Low'], close=df_stock['Close'], name='K線'), row=1, col=1)

    # 疊加 MA (如果存在)
    if f'SMA_{st.session_state.fast_len}' in df_stock.columns:
        fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock[f'SMA_{st.session_state.fast_len}'], line=dict(color='orange', width=1), name=f'MA {st.session_state.fast_len}'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock[f'SMA_{st.session_state.slow_len}'], line=dict(color='blue', width=1), name=f'MA {st.session_state.slow_len}'), row=1, col=1)

    # --- 2. 淨值曲線 ---
    # 策略淨值 (使用 TimeReturn Analyzer 提取的數據)
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, 
                             line=dict(color='red', width=2), name='策略淨值'), row=2, col=1)
    
    # 大盤淨值
    fig.add_trace(go.Scatter(x=bench_cumulative.index, y=bench_cumulative.values, 
                             line=dict(color='gray', width=2, dash='dash'), name='大盤基準 (Buy & Hold)'), row=2, col=1)

    fig.update_layout(height=800, margin=dict(l=50, r=50, t=50, b=50), xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text='價格', row=1, col=1)
    fig.update_yaxes(title_text='資產總值 ($)', row=2, col=1)
    return fig

# ==========================================
# 3. Streamlit 主介面
# ==========================================
st.title("🛡️ 專業級策略回測系統 v3")
st.markdown("自由組合多重指標、調整 VIX 宏觀濾網，並進行專業績效比較。")

# --- 初始化 Session State (用於儲存參數，使數據能傳給繪圖功能) ---
if 'fast_len' not in st.session_state:
    st.session_state.fast_len = 10
    st.session_state.slow_len = 50

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("1. 標的與資金")
    symbol = st.text_input("回測股票代碼", value="2330.TW")
    benchmark_symbol = st.text_input("對比大盤代碼", value="^TWII")
    start_date = st.date_input("開始日期", datetime.date(2020, 1, 1))
    end_date = st.date_input("結束日期", datetime.date.today())
    cash = st.number_input("初始資金", value=100000, step=10000)
    commission = st.number_input("手續費率", value=0.001425, step=0.0001, format="%.6f")

    st.divider()
    st.header("2. 策略組裝 (AND 邏輯)")

    # --- A. MA 條件 ---
    st.subheader("趨勢指標 (MA)")
    use_ma = st.checkbox("啟用 MA 交叉訊號", value=True)
    st.session_state.fast_len = st.number_input("短均線 (Fast)", 5, 50, 10, key="fast_len")
    st.session_state.slow_len = st.number_input("長均線 (Slow)", 10, 200, 50, key="slow_len")
    st.caption("進場: 短線向上穿過長線 | 出場: 短線向下穿過長線")

    # --- B. 震盪指標 (RSI) ---
    st.subheader("震盪指標 (RSI)")
    use_rsi = st.checkbox("啟用 RSI 超賣/超買訊號", value=False)
    rsi_len = st.slider("RSI 週期", 5, 30, 14)
    rsi_buy = st.slider("RSI 買入閾值 (<)", 10, 50, 30, help="RSI 低於此值時買入")
    rsi_sell = st.slider("RSI 賣出閾值 (>)", 50, 90, 70, help="RSI 高於此值時賣出")

    # --- C. 停損 ---
    st.subheader("風險管理")
    stop_loss = st.slider("🛑 強制停損百分比 (%)", 1.0, 30.0, 10.0) / 100.0


    st.divider()
    st.header("🌪️ 宏觀濾網 (Druckenmiller)")
    use_vix = st.checkbox("啟用 VIX 恐慌濾網", value=True)
    vix_logic = st.selectbox("VIX 執行邏輯", ["恐慌時買入 (Buy on Panic)", "平靜時避免買入 (Avoid Flat)"])
    vix_thres = st.slider("VIX 警戒值", 15.0, 50.0, 30.0, help="VIX 高於此值時，視為恐慌狀態")

# --- 執行按鈕 ---
if st.button("🚀 執行策略回測", type="primary"):
    status_text = st.empty()
    status_text.text("⏳ 正在下載數據...")

    # 將所有參數打包成字典，方便傳入 Backtrader
    strategy_params = {
        'use_ma_cross': use_ma,
        'fast_len': st.session_state.fast_len,
        'slow_len': st.session_state.slow_len,
        'use_rsi_signal': use_rsi,
        'rsi_len': rsi_len,
        'rsi_buy': rsi_buy,
        'rsi_sell': rsi_sell,
        'use_vix_filter': use_vix,
        'vix_logic': 'buy_on_panic' if vix_logic == "恐慌時買入 (Buy on Panic)" else 'avoid_flat',
        'vix_threshold': vix_thres,
    }

    try:
        # 1. 下載數據
        df_stock = yf.download(symbol, start=start_date, end=end_date)
        df_bench = yf.download(benchmark_symbol, start=start_date, end=end_date)
        df_vix = None
        
        # VIX 數據
        if use_vix:
            status_text.text("⏳ 正在下載宏觀數據 (VIX)...")
            df_vix = yf.download("^VIX", start=start_date, end=end_date)
            if df_vix.empty:
                st.warning("⚠️ VIX 數據下載失敗，濾網將被禁用。")
                use_vix = False

        # 處理 MultiIndex (yfinance 新版問題)
        for d in [df_stock, df_bench, df_vix]:
            if d is not None and isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)

        if df_stock.empty:
            st.error(f"❌ 找不到 {symbol} 的數據")
            st.stop()
        
        # 預先計算指標 (為了繪圖用)
        if use_ma:
            df_stock.ta.sma(length=st.session_state.fast_len, append=True)
            df_stock.ta.sma(length=st.session_state.slow_len, append=True)
        if use_rsi:
            df_stock.ta.rsi(length=rsi_len, append=True)
        
        # 2. Backtrader 設定
        cerebro = bt.Cerebro()
        cerebro.adddata(bt.feeds.PandasData(dataname=df_stock))
        
        # 加入 VIX 數據
        if use_vix and df_vix is not None and not df_vix.empty:
            cerebro.adddata(bt.feeds.PandasData(dataname=df_vix), name='VIX')
        
        # 設定策略
        cerebro.addstrategy(AdvancedStrategy, strategy_params=strategy_params, stop_loss_pct=stop_loss)
        cerebro.broker.setcash(cash)
        cerebro.broker.setcommission(commission=commission)
        
        # 加入分析器
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn') # 這是繪製淨值曲線的關鍵

        # 3. 執行回測
        status_text.text("⚡ 正在模擬交易與運算...")
        start_val = cerebro.broker.getvalue()
        results = cerebro.run()
        end_val = cerebro.broker.getvalue()
        strat = results[0] 

        # 4. 數據提取與計算
        # 淨值曲線 (包含未實現損益)
        return_analysis = strat.analyzers.timereturn.get_analysis()
        equity_curve_data = pd.Series(return_analysis).cumsum().apply(lambda x: cash * (1 + x))
        
        # 其他分析
        trade_analysis = strat.analyzers.trades.get_analysis()
        mdd = strat.analyzers.drawdown.get_analysis()['max']['drawdown']
        total_trades = trade_analysis.get('total', {}).get('total', 0)
        win_trades = trade_analysis.get('won', {}).get('total', 0)
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
        total_return = (end_val - start_val) / start_val * 100
        
        bench_return = (df_bench['Close'].iloc[-1] - df_bench['Close'].iloc[0]) / df_bench['Close'].iloc[0] * 100
        
        status_text.empty() # 清除狀態文字

        # --- 5. 顯示結果儀表板 ---
        st.subheader("🏆 回測績效報告")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("最終資產", f"${end_val:,.0f}", f"{end_val-start_val:,.0f}")
        col2.metric("策略總報酬", f"{total_return:.2f}%", delta_color="normal")
        col3.metric("大盤基準報酬", f"{bench_return:.2f}%", delta=f"{total_return - bench_return:.2f}% (超額)")
        col4.metric("最大回撤 (MDD)", f"{mdd:.2f}%")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("總交易次數", f"{total_trades} 次")
        col6.metric("勝率", f"{win_rate:.1f}%")
        col7.metric("VIX 濾網", vix_logic)
        col8.metric("停損設定", f"{stop_loss*100:.1f}%")

        # --- 6. 繪圖 (淨值曲線修正) ---
        st.subheader("📊 績效與股價走勢")
        st.plotly_chart(plot_results(df_stock, symbol, df_bench, equity_curve_data), use_container_width=True)

    except Exception as e:
        st.error(f"發生錯誤：{e}")
        st.exception(e)

else:
    st.info("👈 請在左側調整參數，並點擊「執行策略回測」開始。")
