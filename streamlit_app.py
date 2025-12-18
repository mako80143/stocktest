import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import backtrader as bt
import datetime
import plotly.graph_objects as go
from streamlit_lightweight_charts import renderLightweightCharts
import collections.abc
import itertools
import numpy as np
from scipy.signal import argrelextrema # 數學極值庫
import gc # 垃圾回收 (防爆用)

# 1. 兼容性修復
collections.Iterable = collections.abc.Iterable

# 2. 頁面設定
st.set_page_config(page_title="上帝視角 v23", layout="wide")
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
# 3. 數學核心：上帝視角演算法
# ==========================================
def calculate_god_mode(df, init_cash):
    """
    計算理論最大獲利 (上帝視角)
    邏輯：在每個波段低點買進 100%，高點賣出 100%
    """
    # 複製數據以免影響正本
    data = df['Close'].values
    dates = df.index
    
    # 使用 Scipy 尋找局部極值
    # order=3 代表前後 3 天都比它高/低才算 (過濾太細碎的雜訊)
    n = 3 
    
    # 找出低點索引 (Valley)
    min_idx = argrelextrema(data, np.less, order=n)[0]
    # 找出高點索引 (Peak)
    max_idx = argrelextrema(data, np.greater, order=n)[0]
    
    # 合併並排序所有轉折點
    signals = []
    for idx in min_idx: signals.append((idx, 'Buy'))
    for idx in max_idx: signals.append((idx, 'Sell'))
    signals.sort(key=lambda x: x[0])
    
    # 開始模擬上帝交易
    cash = init_cash
    shares = 0
    equity_curve = []
    trade_log = []
    
    # 建立一個與 df 等長的資產陣列，預設為 NaN
    god_curve_series = pd.Series(index=df.index, dtype=float)
    god_curve_series.iloc[0] = init_cash
    
    current_val = init_cash
    
    for i in range(len(data)):
        # 檢查今天是不是轉折點
        # 注意：argrelextrema 是看前後 n 天，所以會有未來函數 (這就是上帝視角)
        
        # 簡單狀態機
        is_buy_point = i in min_idx
        is_sell_point = i in max_idx
        
        price = data[i]
        
        if is_buy_point and cash > 0: # 有錢且遇到低點 -> 買
            shares = cash / price
            cash = 0
            trade_log.append({'Date': dates[i], 'Type': 'God Buy', 'Price': price})
            
        elif is_sell_point and shares > 0: # 有貨且遇到高點 -> 賣
            cash = shares * price
            shares = 0
            trade_log.append({'Date': dates[i], 'Type': 'God Sell', 'Price': price})
            
        # 更新每日市值
        if shares > 0:
            current_val = shares * price
        else:
            current_val = cash
            
        god_curve_series.iloc[i] = current_val

    # 補齊空值
    god_curve_series = god_curve_series.ffill()
    return god_curve_series, trade_log

# ==========================================
# 4. 數據下載 (快取)
# ==========================================
@st.cache_data(ttl=3600)
def get_data(symbol, start):
    end = datetime.date.today()
    df = yf.download(symbol, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if df.empty: return df
    df.index = df.index.tz_localize(None)
    
    vix_df = yf.download("^VIX", start=start, end=end, progress=False)
    if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.get_level_values(0)
    vix_df.index = vix_df.index.tz_localize(None)
    
    df['vix'] = vix_df['Close'].reindex(df.index).ffill()
    return df

# ==========================================
# 5. Backtrader 策略
# ==========================================
class RobustStrategy(bt.Strategy):
    params = (('config', {}),)
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.c = self.params.config
        self.vix = self.datas[0].vix
        self.trade_list = []
        
        self.inds = {}
        if self.c.get('use_ema'): self.inds['ema'] = bt.indicators.EMA(self.datas[0], period=int(self.c.get('ema_len', 20)))

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
        target = cash * (pct / 100.0) * 0.998
        size = int(target / self.dataclose[0])
        if size > 0: self.buy(size=size, info={'name': reason})

    def attempt_sell(self, pct, reason):
        size = self.position.size
        if size > 0:
            target = int(size * (pct / 100.0))
            if target > 0: self.sell(size=target, info={'name': reason})

    def next(self):
        # VIX
        if self.c.get('use_vix'):
            if self.vix[0] > self.c['vix_buy_thres'] and self.vix[-1] <= self.c['vix_buy_thres']:
                self.attempt_buy(self.c['vix_buy_pct'], "VIX Buy")
            if self.vix[0] < self.c['vix_sell_thres'] and self.vix[-1] >= self.c['vix_sell_thres']:
                self.attempt_sell(self.c['vix_sell_pct'], "VIX Sell")
        
        # EMA
        if self.c.get('use_ema'):
            if self.dataclose[0] > self.inds['ema'][0] and self.dataclose[-1] <= self.inds['ema'][-1]:
                self.attempt_buy(self.c['ema_buy_pct'], "EMA Buy")
            if self.dataclose[0] < self.inds['ema'][0] and self.dataclose[-1] >= self.inds['ema'][-1]:
                self.attempt_sell(self.c['ema_sell_pct'], "EMA Sell")

class PandasDataPlus(bt.feeds.PandasData):
    lines = ('vix',)
    params = (('vix', -1),)

# ==========================================
# 6. 介面與控制
# ==========================================
st.sidebar.header("🎛️ 系統控制")
mode = st.sidebar.radio("模式", ["單次詳細分析", "參數窮舉 (Optimization)"])
symbol = st.sidebar.text_input("代碼", "NVDA")
start_date = st.sidebar.date_input("開始", datetime.date(2023, 1, 1))
init_cash = 100000.0

if mode == "單次詳細分析":
    st.sidebar.subheader("策略參數")
    vix_b = st.sidebar.number_input("VIX 買入 >", 26.0)
    vix_s = st.sidebar.number_input("VIX 賣出 <", 14.0)
    
    config = {
        'use_vix': True, 'vix_buy_thres': vix_b, 'vix_buy_pct': 100, 
        'vix_sell_thres': vix_s, 'vix_sell_pct': 100,
        'use_ema': True, 'ema_len': 20, 'ema_buy_pct': 30, 'ema_sell_pct': 100
    }
else:
    st.sidebar.info("⚠️ 為防止當機，組合數請勿超過 100")
    b_start = st.sidebar.number_input("買入開始", 20, 40, 24)
    b_end = st.sidebar.number_input("買入結束", 20, 40, 28)
    s_start = st.sidebar.number_input("賣出開始", 10, 20, 12)
    s_end = st.sidebar.number_input("賣出結束", 10, 20, 16)
    step = st.sidebar.number_input("間隔", 1, 5, 2)

btn = st.sidebar.button("🚀 執行")

if btn:
    df = get_data(symbol, start_date)
    if df.empty: st.stop()

    # 1. 計算上帝視角 (數學極值)
    god_curve, god_log = calculate_god_mode(df, init_cash)
    god_final = god_curve.iloc[-1]
    
    # 2. 計算 Buy & Hold
    bh_curve = (df['Close'] / df['Close'].iloc[0]) * init_cash
    bh_final = bh_curve.iloc[-1]

    if mode == "單次詳細分析":
        # 執行 Backtrader
        cerebro = bt.Cerebro()
        cerebro.adddata(PandasDataPlus(dataname=df))
        cerebro.addstrategy(RobustStrategy, config=config)
        cerebro.broker.setcash(init_cash)
        
        res = cerebro.run()
        strat = res[0]
        final_val = cerebro.broker.getvalue()
        
        # 整理曲線
        t_ret = strat.analyzers.timereturn.get_analysis()
        equity_curve = pd.Series(t_ret).fillna(0)
        equity_curve = (1 + equity_curve).cumprod() * init_cash

        # --- UI 呈現 ---
        st.title(f"⚡ {symbol} 終極戰報")
        
        # 上帝 vs 凡人
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("😇 上帝視角 (God Mode)", f"${god_final:,.0f}", delta=f"{(god_final-init_cash)/init_cash*100:.0f}%", help="理論上的完美操作")
        c2.metric("😈 我的策略", f"${final_val:,.0f}", delta=f"{(final_val-init_cash)/init_cash*100:.2f}%")
        c3.metric("😴 Buy & Hold", f"${bh_final:,.0f}")
        c4.metric("策略效率", f"{(final_val/god_final)*100:.4f}%", help="你的策略是上帝的百分之幾？通常不到 10% 是正常的")

        # 資金曲線
        st.subheader("📈 凡人 vs 上帝")
        fig = go.Figure()
        # 上帝線 (金光閃閃)
        fig.add_trace(go.Scatter(x=god_curve.index, y=god_curve.values, mode='lines', name='上帝視角', line=dict(color='#FFD700', width=2)))
        # 策略線
        fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name='我的策略', line=dict(color='#00e676', width=2)))
        # B&H
        fig.add_trace(go.Scatter(x=bh_curve.index, y=bh_curve.values, mode='lines', name='B&H', line=dict(color='#555555', dash='dash')))
        
        # 使用 Log Scale (因為上帝賺太多了，用普通座標你的線會變成地板)
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500, yaxis_type="log", title="注意：已開啟對數座標 (Log Scale)")
        st.plotly_chart(fig, use_container_width=True)
        
        if strat.trade_list:
            st.subheader("📋 交易日記")
            st.dataframe(pd.DataFrame(strat.trade_list), use_container_width=True)

    else: # 窮舉模式
        buy_rng = range(int(b_start), int(b_end)+1, int(step))
        sell_rng = range(int(s_start), int(s_end)+1, int(step))
        combs = list(itertools.product(buy_rng, sell_rng))
        
        # 防爆檢查
        if len(combs) > 100:
            st.error(f"🛑 組合數過多 ({len(combs)} 組)，請縮小範圍或加大間隔！(建議 < 100)")
            st.stop()
            
        st.info(f"🧪 正在測試 {len(combs)} 種組合...")
        bar = st.progress(0)
        res_data = []
        
        for i, (b, s) in enumerate(combs):
            gc.collect() # 強制釋放記憶體
            
            c_tmp = {
                'use_vix': True, 'vix_buy_thres': b, 'vix_buy_pct': 100,
                'vix_sell_thres': s, 'vix_sell_pct': 100
            }
            cerebro = bt.Cerebro()
            cerebro.adddata(PandasDataPlus(dataname=df))
            cerebro.addstrategy(RobustStrategy, config=c_tmp)
            cerebro.broker.setcash(init_cash)
            r = cerebro.run()
            
            val = cerebro.broker.getvalue()
            res_data.append({"VIX買": b, "VIX賣": s, "權益": val, "ROI": (val-init_cash)/init_cash*100})
            bar.progress((i+1)/len(combs))
            
        res_df = pd.DataFrame(res_data).sort_values("權益", ascending=False)
        
        st.success("✅ 完成！")
        
        # 顯示比較
        best = res_df.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("上帝極限", f"${god_final:,.0f}")
        c2.metric("最佳參數結果", f"${best['權益']:,.0f}", f"買{best['VIX買']} / 賣{best['VIX賣']}")
        c3.metric("達成率", f"{(best['權益']/god_final)*100:.2f}%")
        
        st.dataframe(res_df.style.background_gradient(subset=['權益'], cmap='Greens'), use_container_width=True)
