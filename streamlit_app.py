import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import backtrader as bt
import datetime
import plotly.graph_objects as go
from streamlit_lightweight_charts import renderLightweightCharts

# --- 1. 頁面全域設定 ---
st.set_page_config(page_title="量化戰情室 v5 Alpha", layout="wide", initial_sidebar_state="expanded")

# CSS 優化 (讓介面更緊湊)
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    h1 {font-size: 1.8rem;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 策略邏輯核心 (Backtrader)
# ==========================================
class DynamicStrategy(bt.Strategy):
    """
    動態策略：根據 UI 傳入的參數決定怎麼做
    """
    params = (('ui_config', {}),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.config = self.params.ui_config
        
        # --- 動態建立指標 ---
        self.inds = {}
        
        # 1. MA 均線
        if 'MA' in self.config.get('indicators', []):
            self.inds['ma_fast'] = bt.indicators.SMA(self.datas[0], period=self.config.get('ma_fast', 10))
            self.inds['ma_slow'] = bt.indicators.SMA(self.datas[0], period=self.config.get('ma_slow', 50))
        
        # 2. RSI
        if 'RSI' in self.config.get('indicators', []):
            self.inds['rsi'] = bt.indicators.RSI(self.datas[0], period=14)

    def next(self):
        if self.order: return

        # 取得參數
        indicators = self.config.get('indicators', [])
        stop_loss = self.config.get('stop_loss', 0.1)
        
        # --- A. 風險控管 (優先) ---
        if self.position:
            pct_change = (self.dataclose[0] - self.position.price) / self.position.price
            if pct_change < -stop_loss:
                self.close()
                return

        # --- B. 訊號判斷 ---
        buy_signal = False
        sell_signal = False
        
        # 邏輯組裝 (AND 邏輯：所有選中的指標都必須符合)
        conditions = []
        
        if 'MA' in indicators:
            # 多頭排列才買
            conditions.append(self.inds['ma_fast'][0] > self.inds['ma_slow'][0])
            
        if 'RSI' in indicators:
            # 超賣才買
            conditions.append(self.inds['rsi'][0] < self.config.get('rsi_buy', 30))
        
        # 判斷結果
        if conditions and all(conditions):
            buy_signal = True
            
        # --- C. 執行 ---
        if not self.position and buy_signal:
            # 資金管理：每次投入固定金額
            size = int(self.config.get('trade_size', 50000) / self.dataclose[0])
            if size > 0: self.buy(size=size)
            
        elif self.position:
            # 出場邏輯 (簡單版：指標反轉就賣)
            if 'MA' in indicators and self.inds['ma_fast'][0] < self.inds['ma_slow'][0]:
                sell_signal = True
            if 'RSI' in indicators and self.inds['rsi'][0] > self.config.get('rsi_sell', 70):
                sell_signal = True
                
            if sell_signal:
                self.close()

# ==========================================
# 3. UI: 側邊欄控制台
# ==========================================
st.sidebar.title("🎛️ 策略控制台")

# 區塊 1: 數據設定
with st.sidebar.expander("1. 數據源設定", expanded=True):
    symbol = st.text_input("股票代碼", "NVDA")
    start_date = st.date_input("開始", datetime.date(2022, 1, 1))
    end_date = st.date_input("結束", datetime.date.today())

# 區塊 2: 策略組裝 (重點)
with st.sidebar.expander("2. 策略組裝工廠", expanded=True):
    # 下拉選單多選
    selected_inds = st.multiselect("選擇進場指標 (且/AND)", ["MA", "RSI"], default=["MA"])
    
    ui_config = {'indicators': selected_inds}
    
    if "MA" in selected_inds:
        col1, col2 = st.columns(2)
        ui_config['ma_fast'] = col1.number_input("MA 快線", 5, 50, 20)
        ui_config['ma_slow'] = col2.number_input("MA 慢線", 20, 200, 60)
        
    if "RSI" in selected_inds:
        col1, col2 = st.columns(2)
        ui_config['rsi_buy'] = col1.number_input("RSI 買點 <", 10, 40, 30)
        ui_config['rsi_sell'] = col2.number_input("RSI 賣點 >", 60, 90, 70)

# 區塊 3: 資金與風控
with st.sidebar.expander("3. 資金管理"):
    init_cash = st.number_input("初始本金", 100000, step=10000)
    ui_config['trade_size'] = st.number_input("每次交易金額", 30000, step=5000)
    ui_config['stop_loss'] = st.slider("停損 %", 1, 30, 10) / 100

# 區塊 4: Google Sheet (模擬)
st.sidebar.divider()
st.sidebar.markdown("### ☁️ 財務規劃")
gs_btn = st.sidebar.button("📤 儲存此回測結果到 Google Sheet")
if gs_btn:
    st.sidebar.success("✅ (模擬) 已將數據寫入 Google Sheet！\n\n包含：策略參數、最終獲利、MDD。")
    # 這裡未來會放 gspread 的程式碼

run_btn = st.sidebar.button("🚀 開始回測", type="primary")

# ==========================================
# 4. 主程式邏輯
# ==========================================
st.title(f"📊 {symbol} 策略戰情室")

if run_btn:
    # --- 1. 抓資料 ---
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    
    if data.empty:
        st.error("❌ 抓不到資料，請檢查代碼")
        st.stop()

    # --- 2. 執行 Backtrader ---
    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=data))
    cerebro.addstrategy(DynamicStrategy, ui_config=ui_config)
    cerebro.broker.setcash(init_cash)
    
    # 加入分析器
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    
    results = cerebro.run()
    strat = results[0]
    
    # --- 3. 整理數據 ---
    # 淨值曲線
    t_ret = strat.analyzers.timereturn.get_analysis()
    equity_curve = pd.Series(t_ret).fillna(0).cumsum().apply(lambda x: init_cash * (1 + x))
    
    # 績效指標
    end_val = cerebro.broker.getvalue()
    ret_pct = (end_val - init_cash) / init_cash * 100
    mdd = strat.analyzers.drawdown.get_analysis()['max']['drawdown']

    # ==========================================
    # 5. 結果呈現 (分頁式設計)
    # ==========================================
    tab1, tab2, tab3 = st.tabs(["📈 K線與買賣點 (互動)", "💰 獲利曲線 (分析)", "📋 詳細數據"])

    with tab1:
        st.markdown("### TradingView 風格 K 線圖")
        # 準備 Lightweight Charts 數據
        chart_data = []
        for idx, row in data.iterrows():
            chart_data.append({
                "time": idx.strftime('%Y-%m-%d'),
                "open": row['Open'], "high": row['High'], "low": row['Low'], "close": row['Close']
            })
            
        # 設定圖表 (這裡就是 JS 的設定檔，被包裝成 Python dict)
        chartOptions = {
            "layout": {"textColor": 'black', "background": {"type": 'solid', "color": 'white'}},
            "timeScale": {"rightOffset": 5},
            "grid": {"vertLines": {"visible": False}, "horzLines": {"color": "#eee"}},
        }
        
        series = [{
            "type": 'Candlestick',
            "data": chart_data,
            "options": {
                "upColor": '#26a69a', "downColor": '#ef5350',
                "borderVisible": False, "wickUpColor": '#26a69a', "wickDownColor": '#ef5350'
            }
        }]
        
        # 渲染
        renderLightweightCharts([{"chart": chartOptions, "series": series}], height=500)
        st.caption("💡 提示：滾輪縮放，左鍵拖曳，體驗絲滑流暢的操作。")

    with tab2:
        st.markdown("### 淨值成長曲線 (Equity Curve)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, 
                                 mode='lines', name='我的策略', line=dict(color='blue', width=2)))
        fig.add_hline(y=init_cash, line_dash="dash", line_color="gray", annotation_text="本金")
        fig.update_layout(height=400, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### 核心績效")
        c1, c2, c3 = st.columns(3)
        c1.metric("最終資產", f"${end_val:,.0f}", delta=f"{ret_pct:.1f}%")
        c2.metric("最大回撤 (Risk)", f"{mdd:.2f}%", delta_color="inverse")
        c3.metric("使用策略", "+".join(selected_inds))
        
        st.info("未來功能：這裡將顯示詳細的「逐筆交易紀錄」表格 (Trade Log)。")

else:
    st.info("👈 請在左側設定參數，並點擊「開始回測」")
