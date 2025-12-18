import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import backtrader as bt
import datetime
import plotly.graph_objects as go
from streamlit_lightweight_charts import renderLightweightCharts
import collections.abc
import itertools # 用於窮舉組合

# 1. 兼容性修復
collections.Iterable = collections.abc.Iterable

# 2. 頁面設定 (全黑化 + 記憶體優化)
st.set_page_config(page_title="超級運算版 v22", layout="wide")
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
# 3. 數據下載 (快取優化 - 解決黑屏關鍵)
# ==========================================
@st.cache_data(ttl=3600) # 數據快取 1 小時，避免重複下載撐爆記憶體
def get_data(symbol, start_date):
    end_date = datetime.date.today()
    
    # 下載主數據
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    if df.empty: return df
    
    # 移除時區
    df.index = df.index.tz_localize(None)
    
    # 下載 VIX
    vix_df = yf.download("^VIX", start=start_date, end=end_date, progress=False)
    if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.get_level_values(0)
    vix_df.index = vix_df.index.tz_localize(None)
    
    # 合併
    df['vix'] = vix_df['Close'].reindex(df.index).ffill()
    
    return df

# ==========================================
# 4. Backtrader 策略
# ==========================================
class OptimizationStrategy(bt.Strategy):
    params = (('config', {}),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.c = self.params.config
        self.vix = self.datas[0].vix
        self.trade_list = []
        
        # 指標 (這裡只初始化有開啟的)
        self.inds = {}
        if self.c.get('use_ema', False):
            self.inds['ema'] = bt.indicators.EMA(self.datas[0], period=int(self.c.get('ema_len', 20)))
        if self.c.get('use_macd', False):
            self.inds['macd'] = bt.indicators.MACD(self.datas[0], 
                                                   period_me1=12, period_me2=26, period_signal=9)
        if self.c.get('use_rsi', False):
            self.inds['rsi'] = bt.indicators.RSI(self.datas[0], period=14)

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

    def attempt_buy(self, pct, reason):
        if pct <= 0: return
        cash = self.broker.getcash()
        if cash < 100: return # 沒錢不買
        target_amount = cash * (pct / 100.0) * 0.998
        size = int(target_amount / self.dataclose[0])
        if size > 0: self.buy(size=size, info={'name': reason})

    def attempt_sell(self, pct, reason):
        if pct <= 0: return
        pos_size = self.position.size
        if pos_size > 0:
            target_size = int(pos_size * (pct / 100.0))
            if target_size > 0: self.sell(size=target_size, info={'name': reason})

    def next(self):
        # 1. VIX 邏輯
        if self.c.get('use_vix', True):
            # 買：突破買入閥值
            if self.vix[0] > self.c['vix_buy_thres'] and self.vix[-1] <= self.c['vix_buy_thres']:
                self.attempt_buy(self.c['vix_buy_pct'], "VIX Buy")
            # 賣：跌破賣出閥值
            if self.vix[0] < self.c['vix_sell_thres'] and self.vix[-1] >= self.c['vix_sell_thres']:
                self.attempt_sell(self.c['vix_sell_pct'], "VIX Sell")
        
        # 2. EMA 邏輯
        if self.c.get('use_ema', False):
            if self.dataclose[0] > self.inds['ema'][0] and self.dataclose[-1] <= self.inds['ema'][-1]:
                self.attempt_buy(self.c['ema_buy_pct'], "EMA Buy")
            if self.dataclose[0] < self.inds['ema'][0] and self.dataclose[-1] >= self.inds['ema'][-1]:
                self.attempt_sell(self.c['ema_sell_pct'], "EMA Sell")

        # 3. MACD
        if self.c.get('use_macd', False):
             if self.inds['macd'].macd[0] > self.inds['macd'].signal[0] and self.inds['macd'].macd[-1] <= self.inds['macd'].signal[-1]:
                self.attempt_buy(self.c['macd_buy_pct'], "MACD Buy")
             if self.inds['macd'].macd[0] < self.inds['macd'].signal[0] and self.inds['macd'].macd[-1] >= self.inds['macd'].signal[-1]:
                self.attempt_sell(self.c['macd_sell_pct'], "MACD Sell")

class PandasDataPlus(bt.feeds.PandasData):
    lines = ('vix',)
    params = (('vix', -1),)

# ==========================================
# 5. 側邊欄與運算邏輯
# ==========================================
st.sidebar.header("🎛️ 系統控制台")

# 模式選擇
mode = st.sidebar.radio("請選擇模式", ["單次詳細回測 (Single Run)", "參數窮舉優化 (Optimization)"], index=0)

symbol = st.sidebar.text_input("股票代碼", "NVDA")
init_cash = 100000.0
comm_rate = 0.001425
start_date = st.sidebar.date_input("開始日期", datetime.date(2022, 1, 1))

# --- 參數設定區 ---
if mode == "單次詳細回測 (Single Run)":
    st.sidebar.subheader("參數設定")
    vix_buy_thres = st.sidebar.number_input("VIX 買入閥值", 26.0)
    vix_sell_thres = st.sidebar.number_input("VIX 賣出閥值", 14.0)
    vix_buy_pct = st.sidebar.number_input("買入資金 %", 100.0)
    vix_sell_pct = st.sidebar.number_input("賣出持倉 %", 100.0)
    
    # 這裡為了簡化，指標參數設為固定或簡單開關，重點在 VIX
    use_ema = st.sidebar.checkbox("啟用 EMA 輔助", True)
    
    config = {
        'use_vix': True, 'vix_buy_thres': vix_buy_thres, 'vix_buy_pct': vix_buy_pct,
        'vix_sell_thres': vix_sell_thres, 'vix_sell_pct': vix_sell_pct,
        'use_ema': use_ema, 'ema_len': 20, 'ema_buy_pct': 30, 'ema_sell_pct': 50,
        'use_macd': False, 'use_rsi': False
    }

else: # Optimization Mode
    st.sidebar.subheader("🚀 窮舉範圍設定")
    st.sidebar.info("系統將測試以下範圍內的所有組合")
    
    # 窮舉 VIX 買入閥值
    c1, c2, c3 = st.sidebar.columns(3)
    v_buy_start = c1.number_input("買入開始", 20, 40, 24)
    v_buy_end = c2.number_input("買入結束", 20, 50, 32)
    v_buy_step = c3.number_input("間隔", 1, 5, 2)
    
    # 窮舉 VIX 賣出閥值
    c4, c5, c6 = st.sidebar.columns(3)
    v_sell_start = c4.number_input("賣出開始", 10, 20, 12)
    v_sell_end = c5.number_input("賣出結束", 15, 30, 18)
    v_sell_step = c6.number_input("間隔", 1, 5, 2)
    
    # 資金比例固定，減少運算量
    vix_buy_pct_opt = st.sidebar.number_input("固定買入 %", 100.0)
    vix_sell_pct_opt = st.sidebar.number_input("固定賣出 %", 100.0)

btn_run = st.sidebar.button("🚀 開始執行")

# ==========================================
# 6. 主程式執行
# ==========================================
if btn_run:
    df = get_data(symbol, start_date) # 使用快取數據
    
    if df.empty:
        st.error("無數據")
        st.stop()

    # 計算 Buy & Hold (一次就好)
    initial_close = df['Close'].iloc[0]
    bh_final = (init_cash / initial_close) * df['Close'].iloc[-1]
    bh_roi = (bh_final - init_cash) / init_cash * 100

    # ---------------------------
    # 模式 A: 單次詳細回測
    # ---------------------------
    if mode == "單次詳細回測 (Single Run)":
        cerebro = bt.Cerebro()
        cerebro.adddata(PandasDataPlus(dataname=df))
        cerebro.addstrategy(OptimizationStrategy, config=config)
        cerebro.broker.setcash(init_cash)
        cerebro.broker.setcommission(commission=comm_rate)
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
        
        results = cerebro.run()
        strat = results[0]
        final_val = cerebro.broker.getvalue()
        roi = (final_val - init_cash) / init_cash * 100
        
        # 畫圖與數據 (同 v21)
        st.title(f"📊 {symbol} 單次戰報")
        col1, col2, col3 = st.columns(3)
        col1.metric("最終權益", f"${final_val:,.0f}", f"{roi:.2f}%")
        col2.metric("Buy & Hold", f"${bh_final:,.0f}", f"{bh_roi:.2f}%")
        col3.metric("交易次數", len(strat.trade_list))
        
        # 繪圖
        t_ret = strat.analyzers.timereturn.get_analysis()
        equity_curve = pd.Series(t_ret).fillna(0)
        equity_curve = (1 + equity_curve).cumprod() * init_cash
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name='策略', line=dict(color='#00e676')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'] * (init_cash/initial_close), mode='lines', name='B&H', line=dict(color='#555555', dash='dash')))
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        if strat.trade_list:
            st.dataframe(pd.DataFrame(strat.trade_list), use_container_width=True)

    # ---------------------------
    # 模式 B: 參數窮舉優化 (Optimization)
    # ---------------------------
    else:
        st.title(f"🧪 {symbol} 參數窮舉實驗室")
        
        # 產生所有參數組合
        buy_range = range(int(v_buy_start), int(v_buy_end) + 1, int(v_buy_step))
        sell_range = range(int(v_sell_start), int(v_sell_end) + 1, int(v_sell_step))
        combinations = list(itertools.product(buy_range, sell_range))
        
        total_runs = len(combinations)
        st.info(f"預計執行 **{total_runs}** 次回測運算... 請稍候")
        
        # 進度條
        progress_bar = st.progress(0)
        results_data = []
        
        # 開始迴圈測試
        for i, (b_thres, s_thres) in enumerate(combinations):
            # 建立每一次的設定
            opt_config = {
                'use_vix': True, 
                'vix_buy_thres': b_thres, 'vix_buy_pct': vix_buy_pct_opt,
                'vix_sell_thres': s_thres, 'vix_sell_pct': vix_sell_pct_opt,
                'use_ema': False, 'use_macd': False # 為了速度，優化時先只測 VIX
            }
            
            # 建立並執行回測
            cerebro = bt.Cerebro()
            cerebro.adddata(PandasDataPlus(dataname=df))
            cerebro.addstrategy(OptimizationStrategy, config=opt_config)
            cerebro.broker.setcash(init_cash)
            cerebro.broker.setcommission(commission=comm_rate)
            
            res = cerebro.run()
            final_v = cerebro.broker.getvalue()
            roi_v = (final_v - init_cash) / init_cash * 100
            trades_count = len(res[0].trade_list)
            
            results_data.append({
                "VIX 買入": b_thres,
                "VIX 賣出": s_thres,
                "最終權益": final_v,
                "報酬率 (%)": roi_v,
                "交易次數": trades_count
            })
            
            # 更新進度條
            progress_bar.progress((i + 1) / total_runs)
        
        # 整理結果
        res_df = pd.DataFrame(results_data)
        
        # 找出冠軍
        best_run = res_df.loc[res_df['最終權益'].idxmax()]
        
        st.success("✅ 運算完成！")
        
        # 顯示冠軍參數
        c1, c2, c3 = st.columns(3)
        c1.metric("🏆 最佳 ROI", f"{best_run['報酬率 (%)']:.2f}%")
        c2.metric("最佳買入閥值", int(best_run['VIX 買入']))
        c3.metric("最佳賣出閥值", int(best_run['VIX 賣出']))
        
        # 顯示熱力圖表 (Top 10)
        st.subheader("📋 最佳參數排行 (Top 10)")
        top_10 = res_df.sort_values(by="報酬率 (%)", ascending=False).head(10)
        
        # 使用 Pandas Style 上色
        st.dataframe(
            top_10.style.format({
                "最終權益": "${:,.0f}", 
                "報酬率 (%)": "{:.2f}%"
            }).background_gradient(subset=["報酬率 (%)"], cmap="Greens"),
            use_container_width=True
        )
        
        st.subheader("🧩 所有測試數據")
        st.dataframe(res_df, use_container_width=True)
