import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import backtrader as bt
import datetime
from streamlit_lightweight_charts import renderLightweightCharts

# --- 1. 頁面設定 ---
st.set_page_config(page_title="全能量化戰情室 v5.1", layout="wide")

# CSS 微調：讓圖表更寬，減少留白
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 3rem;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Backtrader 策略核心 (支援多指標)
# ==========================================
class AllInOneStrategy(bt.Strategy):
    params = (('config', {}),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.cfg = self.params.config
        self.inds = {}

        # --- A. 動態建立指標 ---
        # 1. SMA (均線)
        if 'SMA' in self.cfg['indicators']:
            self.inds['sma_fast'] = bt.indicators.SMA(self.datas[0], period=self.cfg['sma_fast'])
            self.inds['sma_slow'] = bt.indicators.SMA(self.datas[0], period=self.cfg['sma_slow'])
        
        # 2. RSI
        if 'RSI' in self.cfg['indicators']:
            self.inds['rsi'] = bt.indicators.RSI(self.datas[0], period=self.cfg['rsi_len'])

        # 3. MACD
        if 'MACD' in self.cfg['indicators']:
            self.inds['macd'] = bt.indicators.MACD(self.datas[0], 
                                                   period_me1=self.cfg['macd_fast'], 
                                                   period_me2=self.cfg['macd_slow'], 
                                                   period_signal=self.cfg['macd_signal'])
        
        # 4. Bollinger Bands (布林)
        if 'BBands' in self.cfg['indicators']:
            self.inds['bbands'] = bt.indicators.BollingerBands(self.datas[0], period=self.cfg['bb_len'], devfactor=self.cfg['bb_dev'])

    def next(self):
        if self.order: return

        # 取得設定
        inds_on = self.cfg['indicators']
        
        # --- B. 停損邏輯 ---
        if self.position:
            pct_change = (self.dataclose[0] - self.position.price) / self.position.price
            if pct_change < -self.cfg['stop_loss']:
                self.close()
                return

        # --- C. 進出場訊號 (AND 邏輯) ---
        buy_signal = False
        sell_signal = False
        conditions = []

        # 1. SMA 邏輯
        if 'SMA' in inds_on:
            conditions.append(self.inds['sma_fast'][0] > self.inds['sma_slow'][0])
        
        # 2. RSI 邏輯
        if 'RSI' in inds_on:
            conditions.append(self.inds['rsi'][0] < self.cfg['rsi_buy'])

        # 3. MACD 邏輯 (MACD 線 > 訊號線)
        if 'MACD' in inds_on:
            conditions.append(self.inds['macd'].macd[0] > self.inds['macd'].signal[0])
            
        # 4. BBands 邏輯 (收盤價觸碰到下軌)
        if 'BBands' in inds_on:
            conditions.append(self.dataclose[0] < self.inds['bbands'].bot[0])

        # 綜合判斷
        if conditions and all(conditions):
            buy_signal = True
        
        # --- D. 執行 ---
        if not self.position and buy_signal:
            # 資金管理：固定金額買入
            size = int(self.cfg['trade_size'] / self.dataclose[0])
            if size > 0: self.buy(size=size)
            
        elif self.position:
            # 簡單出場：指標反轉就賣 (或是你可以加入更複雜的出場)
            exit_conds = []
            if 'SMA' in inds_on: exit_conds.append(self.inds['sma_fast'][0] < self.inds['sma_slow'][0])
            if 'RSI' in inds_on: exit_conds.append(self.inds['rsi'][0] > self.cfg['rsi_sell'])
            
            if any(exit_conds):
                self.close()

# ==========================================
# 3. UI 介面設計 (左側控制台)
# ==========================================
st.sidebar.header("🛠️ 策略控制台")

# --- 1. 數據源 ---
with st.sidebar.expander("1. 數據與比較", expanded=True):
    symbol = st.text_input("主代號 (回測)", "NVDA")
    benchmark_symbol = st.text_input("比較代號 (基準)", "SPY")
    start_date = st.date_input("開始", datetime.date(2022, 1, 1))
    end_date = st.date_input("結束", datetime.date.today())

# --- 2. 指標選擇與參數 (全開) ---
with st.sidebar.expander("2. 技術指標設定", expanded=True):
    # 多選選單
    all_indicators = ['SMA', 'RSI', 'MACD', 'BBands']
    selected_inds = st.multiselect("選擇啟用指標 (同時符合才買)", all_indicators, default=['SMA', 'RSI'])
    
    config = {'indicators': selected_inds}

    # 動態顯示參數 (只有被選中時才跳出來)
    if 'SMA' in selected_inds:
        st.caption("--- SMA 設定 ---")
        c1, c2 = st.columns(2)
        config['sma_fast'] = c1.number_input("快線", 5, 50, 20)
        config['sma_slow'] = c2.number_input("慢線", 20, 200, 60)
    
    if 'RSI' in selected_inds:
        st.caption("--- RSI 設定 ---")
        config['rsi_len'] = st.number_input("RSI 週期", 5, 30, 14)
        c1, c2 = st.columns(2)
        config['rsi_buy'] = c1.slider("RSI 買入 <", 10, 50, 30)
        config['rsi_sell'] = c2.slider("RSI 賣出 >", 50, 90, 70)
        
    if 'MACD' in selected_inds:
        st.caption("--- MACD 設定 ---")
        c1, c2, c3 = st.columns(3)
        config['macd_fast'] = c1.number_input("快", 12)
        config['macd_slow'] = c2.number_input("慢", 26)
        config['macd_signal'] = c3.number_input("訊號", 9)
        
    if 'BBands' in selected_inds:
        st.caption("--- 布林通道設定 ---")
        config['bb_len'] = st.number_input("週期", 20)
        config['bb_dev'] = st.number_input("標準差", 2.0)

# --- 3. 資金 ---
with st.sidebar.expander("3. 資金管理"):
    init_cash = st.number_input("初始本金", 100000)
    config['trade_size'] = st.number_input("每次投入", 50000)
    config['stop_loss'] = st.slider("停損 %", 1, 50, 10) / 100

run_btn = st.sidebar.button("🚀 執行全能回測", type="primary")

# ==========================================
# 4. 主程式邏輯
# ==========================================
st.title(f"📊 {symbol} vs {benchmark_symbol} 全能分析")

if run_btn:
    with st.spinner("正在下載數據與運算..."):
        # 1. 下載數據 (處理 MultiIndex)
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        df_bench = yf.download(benchmark_symbol, start=start_date, end=end_date, progress=False)
        
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if isinstance(df_bench.columns, pd.MultiIndex): df_bench.columns = df_bench.columns.get_level_values(0)
        
        if df.empty or df_bench.empty:
            st.error("❌ 抓不到數據，請檢查代號")
            st.stop()

        # 2. 執行 Backtrader
        cerebro = bt.Cerebro()
        cerebro.adddata(bt.feeds.PandasData(dataname=df))
        cerebro.addstrategy(AllInOneStrategy, config=config)
        cerebro.broker.setcash(init_cash)
        
        # 分析器
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
        results = cerebro.run()
        strat = results[0]
        
        # 3. 計算淨值曲線
        t_ret = strat.analyzers.timereturn.get_analysis()
        equity_curve = pd.Series(t_ret).fillna(0).cumsum().apply(lambda x: init_cash * (1 + x))
        
        # 計算基準 (Benchmark) 的曲線 (模擬同樣本金買入持有)
        bench_ret = df_bench['Close'].pct_change().fillna(0)
        bench_curve = (1 + bench_ret).cumprod() * init_cash
        # 讓基準曲線的日期與策略對齊 (只取交集部分)
        bench_curve = bench_curve.reindex(equity_curve.index, method='ffill')

        # ==========================================
        # 5. 視覺化：同步圖表 (重點功能)
        # ==========================================
        
        # A. 準備主圖數據 (K線)
        chart_kline_data = []
        for idx, row in df.iterrows():
            chart_kline_data.append({
                "time": idx.strftime('%Y-%m-%d'),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close'])
            })

        # B. 準備比較數據 (策略 vs 基準)
        # 這些是 LineSeries，要與 K 線分開但共用時間軸
        line_strategy_data = []
        line_benchmark_data = []
        
        for date, val in equity_curve.items():
            line_strategy_data.append({"time": date.strftime('%Y-%m-%d'), "value": float(val)})
            
        for date, val in bench_curve.items():
            if pd.notnull(val):
                line_benchmark_data.append({"time": date.strftime('%Y-%m-%d'), "value": float(val)})

        # C. 準備指標數據 (預先用 pandas_ta 計算以方便繪圖)
        # 為了效能，我們這裡只算出 user 選的指標來畫圖
        indicator_series = [] # 存放要畫的指標設定
        
        if 'RSI' in selected_inds:
            rsi_vals = ta.rsi(df['Close'], length=config['rsi_len'])
            rsi_data = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in rsi_vals.items() if pd.notnull(v)]
            indicator_series.append({
                "type": "Line",
                "data": rsi_data,
                "options": {"color": 'purple', "lineWidth": 2, "priceScaleId": 'right'},
                "pane": 1 # 放在第二個窗格
            })

        # D. 組合圖表設定 (重點：使用 list 來堆疊圖表)
        
        # --- 窗格 0: K線 + 比較曲線 ---
        # 這裡我們做一個技巧：把 K 線放在主軸，把獲利曲線設為 Overlay
        
        chart_options_main = {
            "layout": {"textColor": 'black', "background": {"type": 'solid', "color": 'white'}},
            "height": 400,
            "timeScale": {"rightOffset": 5, "timeVisible": True},
            "grid": {"vertLines": {"visible": False}, "horzLines": {"color": "#eee"}},
            "rightPriceScale": {"scaleMargins": {"top": 0.1, "bottom": 0.1}}, # K線價格軸
        }
        
        series_main = [
            {
                "type": 'Candlestick',
                "data": chart_kline_data,
                "options": {
                    "upColor": '#26a69a', "downColor": '#ef5350', 
                    "borderVisible": False, "wickUpColor": '#26a69a', "wickDownColor": '#ef5350'
                }
            }
        ]
        
        # --- 窗格 1: 獲利比較 (Strategy vs Benchmark) ---
        # 這是獨立的一個區塊，專門看錢
        chart_options_equity = {
            "layout": {"textColor": 'black', "background": {"type": 'solid', "color": 'white'}},
            "height": 250,
            "timeScale": {"timeVisible": True},
            "grid": {"vertLines": {"visible": False}},
        }
        
        series_equity = [
            {
                "type": 'Line',
                "data": line_strategy_data,
                "options": {"color": 'blue', "lineWidth": 2, "title": "我的策略資產"}
            },
            {
                "type": 'Line',
                "data": line_benchmark_data,
                "options": {"color": 'gray', "lineWidth": 2, "lineStyle": 2, "title": f"基準 ({benchmark_symbol})"}
            }
        ]

        # --- 顯示結果 ---
        st.subheader("🎯 互動式同步分析")
        st.info("💡 提示：所有圖表的時間軸已鎖定。在任一圖表移動滑鼠，十字線會同步顯示所有數值。")
        
        # 渲染圖表：傳入 List 就會變成垂直排列且同步的圖表組
        charts_to_render = [
            {"chart": chart_options_main, "series": series_main},
            {"chart": chart_options_equity, "series": series_equity}
        ]
        
        # 如果有指標，加入指標窗格
        if indicator_series:
             chart_options_ind = {
                "layout": {"textColor": 'black', "background": {"type": 'solid', "color": 'white'}},
                "height": 200,
            }
             # 把所有指標加進去
             charts_to_render.append({"chart": chart_options_ind, "series": indicator_series})

        renderLightweightCharts(charts_to_render, key="sync_charts")

        # --- 文字數據摘要 ---
        st.divider()
        ret_pct = (equity_curve.iloc[-1] - init_cash) / init_cash * 100
        bench_pct = (bench_curve.iloc[-1] - init_cash) / init_cash * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("策略最終績效", f"{ret_pct:.2f}%", f"${equity_curve.iloc[-1] - init_cash:,.0f}")
        c2.metric("大盤基準績效", f"{bench_pct:.2f}%", f"{ret_pct - bench_pct:.2f}% (超額)")
        c3.metric("目前持倉狀態", "持倉中" if strat.position.size > 0 else "空手")

else:
    st.info("👈 請在左側設定參數並開始")
