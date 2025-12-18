import streamlit as st
import yfinance as yf
import pandas as pd
import backtrader as bt
import datetime
import numpy as np
from scipy.signal import argrelextrema
import collections.abc
import warnings

# 1. 基礎設定 (不使用任何 CSS 注入，防止黑屏)
warnings.filterwarnings("ignore")
collections.Iterable = collections.abc.Iterable

st.set_page_config(page_title="VIX 戰情室 (安全模式)", layout="wide")

st.title("🛡️ VIX 戰情室 - 安全模式 (Safe Mode)")
st.caption("如果您看到此畫面，代表系統運算核心正常，僅是先前的圖表負擔過重。")

# ==========================================
# 2. 數據下載 (含快取)
# ==========================================
@st.cache_data(ttl=3600)
def get_data_safe(symbol, start):
    end = datetime.date.today()
    try:
        # 下載數據
        df = yf.download(symbol, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # VIX
        vix_df = yf.download("^VIX", start=start, end=end, progress=False)
        if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.get_level_values(0)
        
        # 清洗與合併
        df.index = df.index.tz_localize(None)
        vix_df.index = vix_df.index.tz_localize(None)
        
        # 合併 VIX，並使用 ffill 填充空值
        df['vix'] = vix_df['Close'].reindex(df.index).ffill().fillna(0)
        
        return df
    except Exception as e:
        st.error(f"數據下載失敗: {e}")
        return pd.DataFrame()

# ==========================================
# 3. 上帝視角計算
# ==========================================
def calculate_god_safe(df, init_cash):
    if df.empty: return pd.Series()
    data = df['Close'].values
    min_idx = argrelextrema(data, np.less, order=5)[0]
    max_idx = argrelextrema(data, np.greater, order=5)[0]
    
    cash = init_cash
    shares = 0
    curve = []
    
    for i in range(len(df)):
        price = data[i]
        if i in min_idx and cash > 0:
            shares = cash / price
            cash = 0
        elif i in max_idx and shares > 0:
            cash = shares * price
            shares = 0
        
        val = (shares * price) if shares > 0 else cash
        curve.append(val)
        
    return pd.Series(curve, index=df.index)

# ==========================================
# 4. Backtrader 策略
# ==========================================
class SafeStrategy(bt.Strategy):
    params = (('config', {}),)
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.c = self.params.config
        self.vix = self.datas[0].vix
        self.trade_list = []
        self.inds = {}
        
        if self.c.get('use_ema'): self.inds['ema'] = bt.indicators.EMA(self.datas[0], period=int(self.c['ema_len']))
        if self.c.get('use_rsi'): self.inds['rsi'] = bt.indicators.RSI(self.datas[0], period=int(self.c['rsi_len']))

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
        target = cash * (pct/100.0) * 0.99
        size = int(target / self.dataclose[0])
        if size > 0: self.buy(size=size, info={'name': reason})

    def attempt_sell(self, pct, reason):
        size = self.position.size
        if size > 0:
            target = int(size * (pct/100.0))
            if target > 0: self.sell(size=target, info={'name': reason})

    def next(self):
        # VIX 邏輯
        if self.c['use_vix']:
            if self.vix[0] > self.c['vix_b_thres'] and self.vix[-1] <= self.c['vix_b_thres']:
                self.attempt_buy(self.c['vix_b_pct'], "VIX Buy")
            if self.vix[0] < self.c['vix_s_thres'] and self.vix[-1] >= self.c['vix_s_thres']:
                self.attempt_sell(self.c['vix_s_pct'], "VIX Sell")
        
        # EMA 邏輯
        if self.c['use_ema']:
            if self.dataclose[0] > self.inds['ema'][0] and self.dataclose[-1] <= self.inds['ema'][-1]:
                self.attempt_buy(self.c['ema_b_pct'], "EMA Buy")
            if self.dataclose[0] < self.inds['ema'][0] and self.dataclose[-1] >= self.inds['ema'][-1]:
                self.attempt_sell(self.c['ema_s_pct'], "EMA Sell")

class PandasDataPlus(bt.feeds.PandasData):
    lines = ('vix',)
    params = (('vix', -1),)

# ==========================================
# 5. 控制台
# ==========================================
with st.sidebar:
    st.header("🔧 設定參數")
    symbol = st.text_input("股票", "NVDA")
    start_date = st.date_input("開始", datetime.date(2023, 1, 1))
    init_cash = st.number_input("本金", 100000.0)
    
    st.divider()
    use_vix = st.checkbox("啟用 VIX", True)
    vix_b_thres = st.number_input("VIX > 買", 26.0)
    vix_b_pct = st.number_input("VIX 買 %", 100.0)
    vix_s_thres = st.number_input("VIX < 賣", 14.0)
    vix_s_pct = st.number_input("VIX 賣 %", 100.0)
    
    st.divider()
    use_ema = st.checkbox("啟用 EMA", True)
    ema_len = st.number_input("EMA 週期", 20)
    ema_b_pct = st.number_input("EMA 買 %", 30.0)
    ema_s_pct = st.number_input("EMA 賣 %", 50.0)
    
    st.divider()
    use_rsi = st.checkbox("啟用 RSI", False)
    rsi_len = 14

btn = st.sidebar.button("執行回測")

# ==========================================
# 6. 執行區
# ==========================================
if btn:
    with st.spinner("運算中..."):
        df = get_data_safe(symbol, start_date)
        
        if df.empty:
            st.error("無數據")
            st.stop()
            
        # 準備配置
        config = {
            'use_vix': use_vix, 'vix_b_thres': vix_b_thres, 'vix_b_pct': vix_b_pct,
            'vix_s_thres': vix_s_thres, 'vix_s_pct': vix_s_pct,
            'use_ema': use_ema, 'ema_len': ema_len, 'ema_b_pct': ema_b_pct, 'ema_s_pct': ema_s_pct,
            'use_rsi': use_rsi, 'rsi_len': rsi_len
        }
        
        # 跑上帝視角
        god_curve = calculate_god_safe(df, init_cash)
        
        # 跑 B&H
        bh_curve = (df['Close'] / df['Close'].iloc[0]) * init_cash
        
        # 跑策略
        cerebro = bt.Cerebro()
        cerebro.adddata(PandasDataPlus(dataname=df))
        cerebro.addstrategy(SafeStrategy, config=config)
        cerebro.broker.setcash(init_cash)
        cerebro.broker.setcommission(commission=0.001425)
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
        
        results = cerebro.run()
        strat = results[0]
        
        # 整理數據
        final_val = cerebro.broker.getvalue()
        t_ret = strat.analyzers.timereturn.get_analysis()
        equity_curve = pd.Series(t_ret).fillna(0)
        equity_curve = (1 + equity_curve).cumprod() * init_cash
        
        trade_log = pd.DataFrame(strat.trade_list)
        
        # === 顯示結果 (使用原生圖表，不黑屏) ===
        
        # 1. 績效數字
        c1, c2, c3 = st.columns(3)
        c1.metric("最終權益", f"${final_val:,.0f}")
        c2.metric("上帝視角", f"${god_curve.iloc[-1]:,.0f}")
        c3.metric("Buy & Hold", f"${bh_curve.iloc[-1]:,.0f}")
        
        # 2. 獲利曲線 (使用 st.line_chart，最輕量)
        st.subheader("📈 資金成長曲線")
        chart_data = pd.DataFrame({
            "我的策略": equity_curve,
            "Buy & Hold": bh_curve,
            "上帝視角": god_curve
        })
        st.line_chart(chart_data)
        
        # 3. K線與 VIX 預覽
        st.subheader("📊 價格與 VIX 走勢")
        price_vix_data = pd.DataFrame({
            "股價 (Close)": df['Close'],
            "VIX": df['vix']
        })
        st.line_chart(price_vix_data) # 簡單的雙線圖
        
        # 4. 交易明細
        if not trade_log.empty:
            st.subheader("📋 交易紀錄")
            st.dataframe(trade_log, use_container_width=True)
        else:
            st.warning("無交易紀錄，請檢查 VIX 條件是否太嚴格。")
