import streamlit as st
import yfinance as yf
import pandas as pd
import backtrader as bt
import datetime
from streamlit_lightweight_charts import renderLightweightCharts
import collections.abc
import numpy as np
import gc
import warnings
from scipy.signal import argrelextrema # 用於計算上帝視角波峰波谷

# --- 1. 系統設定 ---
warnings.filterwarnings("ignore")
collections.Iterable = collections.abc.Iterable
st.set_page_config(page_title="VIX 策略終極版 (上帝視角)", layout="wide")

st.markdown("""
<style>
    header {visibility: hidden;}
    .block-container {padding-top: 0rem !important; padding-bottom: 1rem !important;}
    .stApp {background-color: #0e1117;}
    input {font-weight: bold; color: #00e676 !important;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 數據下載
# ==========================================
@st.cache_data(ttl=3600)
def get_data(symbol, start):
    end = datetime.date.today()
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty: return pd.DataFrame()
        df.index = df.index.tz_localize(None)

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
# 3. 上帝視角算法 (God Mode)
# ==========================================
def calculate_god_mode(df, init_cash, fee_pct):
    """
    計算理論最大獲利：
    1. 找出局部高低點
    2. 低點全買，高點全賣
    3. 扣除手續費
    """
    data = df['Close'].values
    # order=5 代表前後5天內的極值，避免過度交易雜訊，抓取波段
    # 您可以把 order 改小 (例如 3) 來抓更細的波動，獲利會更誇張
    min_idx = argrelextrema(data, np.less, order=5)[0]
    max_idx = argrelextrema(data, np.greater, order=5)[0]
    
    cash = init_cash
    shares = 0
    god_curve = [] # 記錄每一天的資產
    
    # 模擬交易
    for i in range(len(df)):
        price = data[i]
        date_str = df.index[i].strftime('%Y-%m-%d')
        
        # 遇到低點 -> 全力買入 (如果手上有錢)
        if i in min_idx and cash > 0:
            # 扣手續費買入
            invest_amount = cash * (1 - fee_pct/100)
            shares = invest_amount / price
            cash = 0 # 現金歸零
            
        # 遇到高點 -> 全力賣出 (如果手上有票)
        elif i in max_idx and shares > 0:
            # 扣手續費賣出
            revenue = shares * price
            cash = revenue * (1 - fee_pct/100)
            shares = 0 # 股票歸零
            
        # 計算當日總資產
        current_val = cash + (shares * price)
        god_curve.append({"time": date_str, "value": current_val})
        
    return god_curve

# ==========================================
# 4. Backtrader 策略 (資金分割版)
# ==========================================
class PandasDataPlus(bt.feeds.PandasData):
    lines = ('vix',)
    params = (('vix', -1),)

class PositionSizingStrategy(bt.Strategy):
    params = (('config', {}),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.vix = self.datas[0].vix
        self.c = self.params.config
        
        self.trade_list = []
        self.cash_history = []
        self.value_history = []
        
        # 指標
        self.ma_short = bt.indicators.SMA(self.datas[0], period=int(self.c['ma_short_len']))
        self.ma_trend = bt.indicators.SMA(self.datas[0], period=int(self.c['ma_trend_len']))
        self.roc = bt.indicators.ROC(self.datas[0], period=int(self.c['roc_len']))
        self.roc_ma = bt.indicators.SMA(self.roc, period=int(self.c['roc_ma_len']))
        self.adx = bt.indicators.ADX(self.datas[0], period=int(self.c['adx_len']))
        self.di_plus = bt.indicators.PlusDI(self.datas[0], period=int(self.c['adx_len']))
        self.di_minus = bt.indicators.MinusDI(self.datas[0], period=int(self.c['adx_len']))

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.trade_list.append({
                'Date': bt.num2date(order.executed.dt),
                'Type': 'Buy' if order.isbuy() else 'Sell',
                'Price': order.executed.price,
                'Value': order.executed.value,
                'Size': order.executed.size,
                'Reason': getattr(order.info, 'name', 'Signal')
            })

    def next(self):
        self.cash_history.append(self.broker.getcash())
        self.value_history.append(self.broker.getvalue())
        if len(self) < 100: return
        
        portfolio_value = self.broker.getvalue()
        current_cash = self.broker.getcash()

        # 訊號判斷
        vix_buy = self.c['use_vix_force'] and (self.vix[0] >= self.c['vix_force_buy'])
        vix_sell = self.c['use_vix_force'] and (self.vix[0] <= self.c['vix_force_sell'])

        sig_ma_buy = (self.dataclose[0] > self.ma_trend[0]) if self.c['use_ma'] else False
        sig_ma_sell = (self.dataclose[0] < self.ma_short[0]) if self.c['use_ma'] else False
        sig_roc_buy = (self.roc[0] > self.roc_ma[0]) if self.c['use_roc'] else False
        sig_roc_sell = (self.roc[0] < self.roc_ma[0]) if self.c['use_roc'] else False
        sig_adx_buy = (self.adx[0] > self.c['adx_thres'] and self.di_plus[0] > self.di_minus[0]) if self.c['use_adx'] else False
        sig_adx_sell = (self.adx[-1] > self.c['adx_strong'] and self.adx[0] < self.adx[-1]) if self.c['use_adx'] else False

        ind_buy = False; ind_sell = False
        mode = self.c['logic_mode']

        if mode == "嚴格共識 (AND)":
            conds = []
            if self.c['use_ma']: conds.append(sig_ma_buy)
            if self.c['use_roc']: conds.append(sig_roc_buy)
            if self.c['use_adx']: conds.append(sig_adx_buy)
            if conds and all(conds): ind_buy = True
            if sig_ma_sell or sig_roc_sell or sig_adx_sell: ind_sell = True
        elif mode == "寬鬆投票 (OR)":
            if sig_ma_buy or sig_roc_buy or sig_adx_buy: ind_buy = True
            if sig_ma_sell or sig_roc_sell or sig_adx_sell: ind_sell = True

        # 交易執行 (買入)
        buy_amt = 0; buy_reason = ""
        if vix_buy:
            buy_amt = portfolio_value * (self.c['pct_vix_buy'] / 100.0)
            buy_reason = "VIX_Panic_Buy"
        elif ind_buy:
            buy_amt = portfolio_value * (self.c['pct_ind_buy'] / 100.0)
            buy_reason = "Ind_Buy"
        
        if buy_amt > 0 and current_cash >= buy_amt:
            size = int(buy_amt / self.dataclose[0])
            if size > 0: self.buy(size=size, info={'name': buy_reason})

        # 交易執行 (賣出)
        sell_pct = 0; sell_reason = ""
        if self.position.size > 0:
            if vix_sell:
                sell_pct = self.c['pct_vix_sell'] / 100.0
                sell_reason = "VIX_Greed_Sell"
            elif ind_sell:
                sell_pct = self.c['pct_ind_sell'] / 100.0
                sell_reason = "Ind_Sell"
            
            if sell_pct > 0:
                size_sell = int(self.position.size * sell_pct)
                if size_sell > 0: self.sell(size=size_sell, info={'name': sell_reason})

# ==========================================
# 5. UI 與 主程式
# ==========================================
st.sidebar.header("⚡ 策略實驗室 (God Mode)")

with st.sidebar.expander("1. 基礎設定", expanded=True):
    symbol = st.text_input("股票代碼", "NVDA")
    start_date = st.date_input("回測開始", datetime.date(2023, 1, 1))
    init_cash = st.number_input("初始本金", 10000.0)
    comm_pct = st.number_input("手續費 (%)", 0.1, step=0.01)

st.sidebar.subheader("2. 資金管控 (Position Sizing)")
c1, c2 = st.sidebar.columns(2)
pct_vix_buy = c1.number_input("VIX 買入 % (總資)", 30.0)
pct_ind_buy = c2.number_input("指標 買入 % (總資)", 20.0)
c3, c4 = st.sidebar.columns(2)
pct_vix_sell = c3.number_input("VIX 賣出 % (持倉)", 50.0)
pct_ind_sell = c4.number_input("指標 賣出 % (持倉)", 100.0)

st.sidebar.subheader("3. VIX 皇權")
use_vix_force = st.sidebar.checkbox("啟用 VIX 強制買賣", True)
c_v1, c_v2 = st.sidebar.columns(2)
vix_force_buy = c_v1.number_input("VIX > 強制買入", 30.0)
vix_force_sell = c_v2.number_input("VIX < 強制賣出", 13.0)

st.sidebar.subheader("4. 指標邏輯")
logic_mode = st.sidebar.selectbox("指標達成條件", ["嚴格共識 (AND)", "寬鬆投票 (OR)"])

with st.sidebar.expander("詳細指標參數", expanded=False):
    use_ma = st.checkbox("啟用 MA", True); ma_short_len=20; ma_trend_len=50
    use_roc = st.checkbox("啟用 ROC", True); roc_len=12; roc_ma_len=6
    use_adx = st.checkbox("啟用 ADX", True); adx_len=14; adx_thres=20; adx_strong=25
    # 為保持 UI 簡潔，這裡參數先寫死或您自行展開

config = {
    'pct_vix_buy': pct_vix_buy, 'pct_ind_buy': pct_ind_buy,
    'pct_vix_sell': pct_vix_sell, 'pct_ind_sell': pct_ind_sell,
    'use_vix_force': use_vix_force, 'vix_force_buy': vix_force_buy, 'vix_force_sell': vix_force_sell,
    'logic_mode': logic_mode,
    'use_ma': use_ma, 'ma_short_len': ma_short_len, 'ma_trend_len': ma_trend_len,
    'use_roc': use_roc, 'roc_len': roc_len, 'roc_ma_len': roc_ma_len,
    'use_adx': use_adx, 'adx_len': adx_len, 'adx_thres': adx_thres, 'adx_strong': adx_strong
}

btn = st.sidebar.button("🚀 執行神之回測", type="primary")

if btn:
    gc.collect()
    with st.spinner("召喚上帝中..."):
        df = get_data(symbol, start_date)
        if df.empty: st.error("無數據"); st.stop()

        # A. 上帝視角計算
        god_curve = calculate_god_mode(df, init_cash, comm_pct)
        god_final = god_curve[-1]['value']

        # B. 策略回測
        cerebro = bt.Cerebro()
        cerebro.adddata(PandasDataPlus(dataname=df))
        cerebro.addstrategy(PositionSizingStrategy, config=config)
        cerebro.broker.setcash(init_cash)
        cerebro.broker.setcommission(commission=comm_pct/100.0)
        
        results = cerebro.run()
        strat = results[0]
        final_val = cerebro.broker.getvalue()
        
        dates = df.index[-len(strat.value_history):]
        eq_data = [{"time": d.strftime('%Y-%m-%d'), "value": v} for d, v in zip(dates, strat.value_history)]
        
        # C. Buy & Hold
        initial_price = df['Close'].iloc[0]
        bh_series = (df['Close'] / initial_price) * init_cash
        bh_data = [{"time": t.strftime('%Y-%m-%d'), "value": v} for t, v in bh_series.items()]
        bh_final = bh_series.iloc[-1]
        
        trade_log = pd.DataFrame(strat.trade_list)

    # UI 
    st.title(f"⚡ {symbol} 終極戰報")
    
    # 績效大PK
    c1, c2, c3, c4 = st.columns(4)
    god_ret = ((god_final - init_cash) / init_cash) * 100
    strat_ret = ((final_val - init_cash) / init_cash) * 100
    bh_ret = ((bh_final - init_cash) / init_cash) * 100
    
    c1.metric("👼 上帝視角 (理論極限)", f"${god_final:,.0f}", delta=f"{god_ret:,.0f}%")
    c2.metric("😈 您的策略", f"${final_val:,.0f}", delta=f"{strat_ret:.1f}%")
    c3.metric("😴 Buy & Hold", f"${bh_final:,.0f}", delta=f"{bh_ret:.1f}%")
    
    # 計算分數：策略績效 / 上帝績效
    score = (strat_ret / god_ret) * 100 if god_ret > 0 else 0
    c4.metric("策略捕捉率", f"{score:.1f} %", help="您抓到了上帝績效的百分之幾？通常 20% 就已經是神人了")

    # 超級圖表：三線合一
    st.subheader("🏆 總資產競賽")
    st.caption("黃線：上帝視角 (最高最低全買賣) | 綠線：您的策略 | 灰線：傻傻抱著")
    
    chart_opts = {
        "chart": {"height": 400, "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"}},
        "series": [
            # 上帝線 (黃金)
            {"type": "Line", "data": god_curve, "options": {"color": "#FFD700", "lineWidth": 2, "lineStyle": 0, "title": "上帝視角"}},
            # 策略線 (亮綠 Area)
            {"type": "Area", "data": eq_data, "options": {"lineColor": "#00E676", "topColor": "rgba(0, 230, 118, 0.2)", "bottomColor": "rgba(0,0,0,0)", "title": "我的策略"}},
            # B&H (灰線虛線)
            {"type": "Line", "data": bh_data, "options": {"color": "#787B86", "lineWidth": 1, "lineStyle": 2, "title": "Buy & Hold"}}
        ]
    }
    renderLightweightCharts([chart_opts], key="god_chart")
    
    # 交易明細
    if not trade_log.empty:
        st.divider()
        st.subheader("📋 策略執行明細")
        trade_log['Amount'] = trade_log['Price'] * trade_log['Size']
        display_df = trade_log.copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        display_df['Price'] = display_df['Price'].map('{:,.2f}'.format)
        display_df['Amount'] = display_df['Amount'].map('{:,.0f}'.format)
        
        def highlight(row):
            c = '#00E676' if row['Type']=='Buy' else '#FF5252'
            bg = 'rgba(255, 215, 0, 0.15)' if 'VIX' in row['Reason'] else 'transparent'
            return [f'color: {c}; background-color: {bg}'] * len(row)
            
        st.dataframe(display_df.style.apply(highlight, axis=1), use_container_width=True)
