import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import backtrader as bt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# --- 頁面設定 ---
st.set_page_config(page_title="專業級策略回測系統 v4 (穩定版)", layout="wide")

# ==========================================
# 1. Backtrader 策略引擎 (核心邏輯)
# ==========================================
class AdvancedStrategy(bt.Strategy):
    """
    v4 策略：支援動態組裝、優先級 (SL > VIX > Tech Indicators)。
    """
    params = (
        ('strategy_params', {}), 
        ('stop_loss_pct', 0.10),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.comm = None
        
        # 輔助數據 (VIX)
        self.vix = self.getdatabyname('VIX').close if len(self.datas) > 1 else None

        # --- 指標預計算 ---
        p = self.params.strategy_params
        
        # 1. 移動平均線 (MA) - 即使未啟用，也需初始化以避免 next() 報錯
        self.ma_fast = bt.indicators.SimpleMovingAverage(self.datas[0], period=p.get('fast_len', 10))
        self.ma_slow = bt.indicators.SimpleMovingAverage(self.datas[0], period=p.get('slow_len', 50))
        self.crossover = bt.indicators.CrossOver(self.ma_fast, self.ma_slow)
        
        # 2. RSI
        self.rsi = bt.indicators.RSI(self.datas[0], period=p.get('rsi_len', 14))

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
            cost_price = self.position.price
            current_price = self.dataclose[0]
            pct_change = (current_price - cost_price) / cost_price
            if pct_change < -self.params.stop_loss_pct:
                self.close()
                return # 停損優先級最高

        # 2. --- 進場條件檢查 (AND 邏輯) ---
        if not self.position:
            p = self.params.strategy_params
            buy_signal = True 
            
            # --- A. 宏觀濾網 (優先順序高於技術指標) ---
            if p.get('use_vix_filter', False) and self.vix is not None:
                vix_thres = p.get('vix_threshold', 30)
                vix_logic = p.get('vix_logic', '恐慌時買入 (Buy on Panic)')
                
                if vix_logic == '恐慌時買入 (Buy on Panic)':
                    if self.vix[0] < vix_thres: # VIX 不恐慌，阻止買入
                        buy_signal = False
                elif vix_logic == '平靜時避免買入 (Avoid Flat)':
                    if self.vix[0] < vix_thres: # VIX 平靜，阻止買入
                        buy_signal = False
            
            # --- B. 技術指標條件 (需全部滿足) ---
            
            # MA 交叉
            if p.get('use_ma_cross', False) and buy_signal: # 只有 VIX 沒擋住，才繼續檢查
                if not (self.crossover > 0):
                    buy_signal = False
            
            # RSI 超賣
            if p.get('use_rsi_signal', False) and buy_signal:
                if not (self.rsi < p.get('rsi_buy', 30)):
                    buy_signal = False
            
            # --- C. 執行買入 ---
            if buy_signal:
                # 這裡不需要 sizer，因為我們在主程式中使用 AllInSizer/FixedSize 或在 buy 裡指定 size
                self.buy()

        # 3. --- 出場條件檢查 (非停損) ---
        else:
            p = self.params.strategy_params
            sell_signal = False 

            # MA 交叉出場
            if p.get('use_ma_cross', False):
                if self.crossover < 0:
                    sell_signal = True
            
            # RSI 超買出場
            if p.get('use_rsi_signal', False):
                if self.rsi > p.get('rsi_sell', 70):
                    sell_signal = True

            if sell_signal:
                self.close()

# ==========================================
# 2. 輔助功能：繪圖與數據 (與 V3 相同，已包含 Session State 修正)
# ==========================================
# (此處省略 plot_results 函式，以簡化篇幅，請使用 V3 最終穩定版中的 plot_results 函式)
# ...

def plot_results(df_stock, symbol, df_bench, equity_curve):
    """繪製 K線、指標與淨值曲線 (使用 V3 最終穩定版中的函式)"""
    
    # 計算大盤累積報酬 (基準)
    bench_cumulative = (1 + df_bench['Close'].pct_change()).fillna(0).cumprod()
    bench_cumulative = bench_cumulative / bench_cumulative.iloc[0] * st.session_state.cash 
    
    # 建立子圖：K線, 淨值曲線
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        subplot_titles=(f'{symbol} 股價與指標', '策略 vs 大盤 淨值曲線 (資產總值)'), 
                        row_width=[0.5, 0.5])

    # --- 1. K線圖 ---
    fig.add_trace(go.Candlestick(x=df_stock.index,
                                 open=df_stock['Open'], high=df_stock['High'],
                                 low=df_stock['Low'], close=df_stock['Close'], name='K線'), row=1, col=1)

    # 疊加 MA (從 Session State 獲取參數)
    fast_len_val = st.session_state.fast_len 
    slow_len_val = st.session_state.slow_len

    if f'SMA_{fast_len_val}' in df_stock.columns:
        fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock[f'SMA_{fast_len_val}'], line=dict(color='orange', width=1), name=f'MA {fast_len_val}'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock[f'SMA_{slow_len_val}'], line=dict(color='blue', width=1), name=f'MA {slow_len_val}'), row=1, col=1)

    # --- 2. 淨值曲線 ---
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, 
                             line=dict(color='red', width=2), name='策略淨值'), row=2, col=1)
    
    # 大盤淨值
    fig.add_trace(go.Scatter(x=bench_cumulative.index, y=bench_cumulative.values, 
                             line=dict(color='gray', width=2, dash='dash'), name='大盤基準 (Buy & Hold)'), row=2, col=1)
    
    start_cash = equity_curve.values[0]
    fig.add_hline(y=start_cash, line_dash="dot", line_color="green", row=2, col=1, 
                  annotation_text=f"起始資金: ${start_cash:,.0f}", annotation_position="top left")

    fig.update_layout(height=800, margin=dict(l=50, r=50, t=50, b=50), xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text='價格', row=1, col=1)
    fig.update_yaxes(title_text='資產總值 ($)', row=2, col=1)
    return fig


# ==========================================
# 3. Streamlit 主介面
# ==========================================
st.title("🛡️ 專業級策略回測系統 v4 (穩定版)")
st.markdown("使用下拉選單自由組合指標，並以 VIX 宏觀濾網為最高優先級。")


# --- 側邊欄設定 ---
with st.sidebar:
    st.header("1. 標的與資金")
    symbol = st.text_input("回測股票代碼", value="2330.TW")
    benchmark_symbol = st.text_input("對比大盤代碼", value="^TWII")
    start_date = st.date_input("開始日期", datetime.date(2020, 1, 1))
    end_date = st.date_input("結束日期", datetime.date.today())
    
    st.session_state.cash = st.number_input("初始資金", value=100000, step=10000)
    st.session_state.position_size = st.number_input("每次交易固定金額 (倉位)", value=50000, step=10000)
    commission = st.number_input("手續費率", value=0.001425, step=0.0001, format="%.6f")

    st.divider()
    st.header("2. 策略組裝 (AND 邏輯)")

    # 💡 使用下拉菜單取代多個 checkbox
    selected_indicators = st.multiselect(
        "選擇啟用的進場指標 (AND 組合)",
        options=["MA 交叉 (趨勢)", "RSI 超賣/超買 (震盪)"],
        default=["MA 交叉 (趨勢)"]
    )
    
    use_ma = "MA 交叉 (趨勢)" in selected_indicators
    use_rsi = "RSI 超賣/超買 (震盪)" in selected_indicators

    # A. MA 條件 UI
    if use_ma:
        st.subheader("MA 參數")
        st.number_input("短均線 (Fast)", 5, 50, 10, key="fast_len") 
        st.number_input("長均線 (Slow)", 10, 200, 50, key="slow_len")

    # B. RSI 條件 UI
    if use_rsi:
        st.subheader("RSI 參數")
        st.slider("RSI 週期", 5, 30, 14, key="rsi_len")
        st.slider("RSI 買入閾值 (<)", 10, 50, 30, help="RSI 低於此值時買入", key="rsi_buy")
        st.slider("RSI 賣出閾值 (>)", 50, 90, 70, help="RSI 高於此值時賣出", key="rsi_sell")

    # C. 停損
    st.subheader("風險管理")
    stop_loss = st.slider("🛑 強制停損百分比 (%)", 1.0, 30.0, 10.0) / 100.0


    st.divider()
    st.header("🌪️ 宏觀濾網 (Druckenmiller - 最高優先)")
    use_vix = st.checkbox("啟用 VIX 恐慌濾網", value=True)
    vix_logic = st.selectbox("VIX 執行邏輯", ["恐慌時買入 (Buy on Panic)", "平靜時避免買入 (Avoid Flat)"])
    vix_thres = st.slider("VIX 警戒值", 15.0, 50.0, 30.0, help="VIX 高於此值時，視為恐慌狀態")

# --- 執行按鈕 ---
if st.button("🚀 執行策略回測", type="primary"):
    status_text = st.empty()
    status_text.text("⏳ 正在下載數據...")

    # 將所有參數打包成字典
    strategy_params = {
        'use_ma_cross': use_ma,
        # MA 參數
        'fast_len': st.session_state.get('fast_len', 10), 
        'slow_len': st.session_state.get('slow_len', 50),
        
        'use_rsi_signal': use_rsi,
        # RSI 參數
        'rsi_len': st.session_state.get('rsi_len', 14),
        'rsi_buy': st.session_state.get('rsi_buy', 30),
        'rsi_sell': st.session_state.get('rsi_sell', 70),

        'use_vix_filter': use_vix,
        'vix_logic': vix_logic,
        'vix_threshold': vix_thres,
    }

    try:
        # 1. 下載數據
        df_stock = yf.download(symbol, start=start_date, end=end_date)
        df_bench = yf.download(benchmark_symbol, start=start_date, end=end_date)
        
        # VIX 數據
        df_vix = yf.download("^VIX", start=start_date, end=end_date)
        if df_vix.empty or df_vix.iloc[-1]['Close'] is None:
            use_vix = False
            
        # 處理 MultiIndex
        for d in [df_stock, df_bench, df_vix]:
            if d is not None and isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)

        if df_stock.empty:
            st.error(f"❌ 找不到 {symbol} 的數據")
            st.stop()
        
        # 預先計算指標 (Plotly 用)
        if use_ma:
            df_stock.ta.sma(length=st.session_state.get('fast_len', 10), append=True)
            df_stock.ta.sma(length=st.session_state.get('slow_len', 50), append=True)
        
        # 2. Backtrader 設定
        cerebro = bt.Cerebro()
        cerebro.adddata(bt.feeds.PandasData(dataname=df_stock))
        
        # 加入 VIX 數據
        if use_vix and df_vix is not None and not df_vix.empty:
            cerebro.adddata(bt.feeds.PandasData(dataname=df_vix), name='VIX')
        
        # 設定策略
        cerebro.addstrategy(AdvancedStrategy, strategy_params=strategy_params, stop_loss_pct=stop_loss)
        cerebro.broker.setcash(st.session_state.cash)
        cerebro.broker.setcommission(commission=commission)
        
        # 引入 Sizer 確保每次交易金額穩定 (使用 FixedSize)
        # Sizer 會將 buy() 呼叫轉換為指定數量
        # 數量 = FixedSize * 股票價格
        # 我們將 Sizer 設為買入 $50000 股
        cerebro.addsizer(bt.sizers.FixedSize, size=int(st.session_state.position_size / df_stock['Close'].iloc[0]))


        # 加入分析器
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn') 

        # 3. 執行回測
        status_text.text("⚡ 正在模擬交易與運算...")
        results = cerebro.run()
        end_val = cerebro.broker.getvalue()
        strat = results[0] 

        # 4. 數據提取與計算 (與 V3 相同)
        return_analysis = strat.analyzers.timereturn.get_analysis()
        equity_curve_data = pd.Series(return_analysis).fillna(0).cumsum().apply(lambda x: st.session_state.cash * (1 + x))
        
        trade_analysis = strat.analyzers.trades.get_analysis()
        mdd = strat.analyzers.drawdown.get_analysis()['max']['drawdown']
        total_trades = trade_analysis.get('total', {}).get('total', 0)
        win_trades = trade_analysis.get('won', {}).get('total', 0)
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
        total_return = (end_val - st.session_state.cash) / st.session_state.cash * 100
        
        bench_return = (df_bench['Close'].iloc[-1] - df_bench['Close'].iloc[0]) / df_bench['Close'].iloc[0] * 100
        
        status_text.empty() 

        # 5. 顯示結果儀表板
        st.subheader("🏆 回測績效報告")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("最終資產", f"${end_val:,.0f}", f"{end_val-st.session_state.cash:,.0f}")
        col2.metric("策略總報酬", f"{total_return:.2f}%", delta_color="normal")
        col3.metric("大盤基準報酬", f"{bench_return:.2f}%", delta=f"{total_return - bench_return:.2f}% (超額)")
        col4.metric("最大回撤 (MDD)", f"{mdd:.2f}%")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("總交易次數", f"{total_trades} 次")
        col6.metric("勝率", f"{win_rate:.1f}%")
        col7.metric("VIX 邏輯", vix_logic)
        col8.metric("停損設定", f"{stop_loss*100:.1f}%")

        # 6. 繪圖
        st.subheader("📊 績效與股價走勢")
        st.plotly_chart(plot_results(df_stock, symbol, df_bench, equity_curve_data), use_container_width=True)

    except Exception as e:
        st.error(f"發生錯誤：{e}")
        st.exception(e)

else:
    st.info("👈 請在左側調整參數，並點擊「執行策略回測」開始。")
