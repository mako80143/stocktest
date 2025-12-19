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
from scipy.signal import argrelextrema

# --- 1. 系統設定 ---
warnings.filterwarnings("ignore")
collections.Iterable = collections.abc.Iterable
st.set_page_config(page_title="VIX 階梯加碼修正版", layout="wide")

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

# 上帝視角
def calculate_god_mode(df, init_cash):
    data = df['Close'].values
    min_idx = argrelextrema(data, np.less, order=5)[0]
    max_idx = argrelextrema(data, np.greater, order=5)[0]
    cash = init_cash; shares = 0; god_curve = []
    for i in range(len(df)):
        price = data[i]
        if i in min_idx and cash > 0: shares = cash/price; cash = 0
        elif i in max_idx and shares > 0: cash = shares*price; shares = 0
        god_curve.append({"time": df.index[i].strftime('%Y-%m-%d'), "value": cash + (shares * price)})
    return god_curve

# ==========================================
# 3. Backtrader 策略 (多階梯獨立運作)
# ==========================================
class PandasDataPlus(bt.feeds.PandasData):
    lines = ('vix',)
    params = (('vix', -1),)

class LadderStrategy(bt.Strategy):
    params = (('config', {}),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.vix = self.datas[0].vix
        self.c = self.params.config
        
        self.trade_list = []
        self.cash_history = []
        self.value_history = []
        
        # 狀態追蹤字典：紀錄每一個「階梯」是否已經買入
        # 這樣 Level 1 買入後，不會影響 Level 2 的買入
        self.ladder_states = {
            'vix_1': False, 'vix_2': False, 'vix_3': False,
            'ma': False, 'roc': False, 'adx': False
        }

        # 指標
        self.ma = bt.indicators.SMA(self.datas[0], period=int(self.c['ma_len']))
        self.roc = bt.indicators.ROC(self.datas[0], period=int(self.c['roc_len']))
        self.adx = bt.indicators.ADX(self.datas[0], period=int(self.c['adx_len']))

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
        if len(self) < 50: return

        portfolio_val = self.broker.getvalue()
        cash = self.broker.getcash()

        # =============================================
        # 1. VIX 階梯加碼邏輯 (3 Levels)
        # =============================================
        # 我們遍歷 3 個設定好的階梯，每個階梯獨立判斷
        
        for i in range(1, 4):
            lvl_key = f'vix_{i}'
            # 取得該層級的參數
            buy_trig = self.c[f'vix_b_trig_{i}']
            sell_trig = self.c[f'vix_s_trig_{i}']
            buy_pct = self.c[f'vix_b_pct_{i}']
            sell_pct = self.c[f'vix_s_pct_{i}']
            enabled = self.c[f'vix_en_{i}']

            if not enabled: continue

            # --- 買入邏輯 ---
            # 條件：VIX > 觸發價 且 該層級尚未持有 (not self.ladder_states[lvl_key])
            if self.vix[0] > buy_trig and not self.ladder_states[lvl_key]:
                amt = portfolio_val * (buy_pct / 100.0)
                if cash >= amt and amt > 0:
                    self.buy(size=int(amt/self.dataclose[0]), info={'name': f'VIX_Lv{i}_Buy'})
                    self.ladder_states[lvl_key] = True # 標記 Lv X 已買入

            # --- 賣出邏輯 ---
            # 條件：VIX < 賣出價 且 該層級目前持有
            elif self.vix[0] < sell_trig and self.ladder_states[lvl_key]:
                if self.position.size > 0:
                    # 這裡的邏輯：賣出目前總持倉的 %
                    size_sell = int(self.position.size * (sell_pct / 100.0))
                    if size_sell > 0:
                        self.sell(size=size_sell, info={'name': f'VIX_Lv{i}_Sell'})
                    self.ladder_states[lvl_key] = False # 標記 Lv X 已賣出，歸零重置

        # =============================================
        # 2. 其他指標 (MA, ROC, ADX)
        # =============================================
        
        # --- MA ---
        if self.c['use_ma']:
            # 站上 MA 買入
            if self.dataclose[0] > self.ma[0] and not self.ladder_states['ma']:
                amt = portfolio_val * (self.c['ma_buy_pct'] / 100.0)
                if cash >= amt:
                    self.buy(size=int(amt/self.dataclose[0]), info={'name': 'MA_Buy'})
                    self.ladder_states['ma'] = True
            # 跌破 MA 賣出
            elif self.dataclose[0] < (self.ma[0] * 0.99) and self.ladder_states['ma']:
                if self.position.size > 0:
                    size_sell = int(self.position.size * (self.c['ma_sell_pct'] / 100.0))
                    if size_sell > 0:
                        self.sell(size=size_sell, info={'name': 'MA_Sell'})
                    self.ladder_states['ma'] = False

        # --- ROC ---
        if self.c['use_roc']:
            if self.roc[0] > self.c['roc_buy_trig'] and not self.ladder_states['roc']:
                amt = portfolio_val * (self.c['roc_buy_pct'] / 100.0)
                if cash >= amt:
                    self.buy(size=int(amt/self.dataclose[0]), info={'name': 'ROC_Buy'})
                    self.ladder_states['roc'] = True
            elif self.roc[0] < self.c['roc_sell_trig'] and self.ladder_states['roc']:
                 if self.position.size > 0:
                    size_sell = int(self.position.size * (self.c['roc_sell_pct'] / 100.0))
                    if size_sell > 0:
                        self.sell(size=size_sell, info={'name': 'ROC_Sell'})
                    self.ladder_states['roc'] = False

        # --- ADX ---
        if self.c['use_adx']:
            if self.adx[0] > self.c['adx_buy_trig'] and not self.ladder_states['adx']:
                amt = portfolio_val * (self.c['adx_buy_pct'] / 100.0)
                if cash >= amt:
                    self.buy(size=int(amt/self.dataclose[0]), info={'name': 'ADX_Buy'})
                    self.ladder_states['adx'] = True
            elif self.adx[0] < self.c['adx_sell_trig'] and self.ladder_states['adx']:
                 if self.position.size > 0:
                    size_sell = int(self.position.size * (self.c['adx_sell_pct'] / 100.0))
                    if size_sell > 0:
                        self.sell(size=size_sell, info={'name': 'ADX_Sell'})
                    self.ladder_states['adx'] = False

# ==========================================
# 4. UI 設定
# ==========================================
st.sidebar.header("🪜 VIX 階梯加碼回測")

with st.sidebar.expander("1. 基礎設定", expanded=True):
    symbol = st.text_input("股票代碼", "NVDA")
    start_date = st.date_input("開始日期", datetime.date(2023, 1, 1))
    init_cash = st.number_input("本金", value=10000.0)
    comm_pct = st.number_input("手續費 (%)", value=0.1)

# --- VIX 階梯設定 ---
st.sidebar.subheader("2. VIX 階梯 (獨立觸發)")
st.sidebar.caption("設定多個買入點，實現「越恐慌買越多」")

config = {}

# Level 1
c1, c2, c3, c4 = st.sidebar.columns([1,1,1,1])
vix_en_1 = c1.checkbox("Lv1", True)
vix_b_trig_1 = c2.number_input("Lv1 買入>", 20.0)
vix_b_pct_1 = c3.number_input("Lv1 買金%", 10.0)
vix_s_trig_1 = c4.number_input("Lv1 賣出<", 18.0)
vix_s_pct_1 = 50.0 # 簡化 UI，先隱藏賣出趴數，預設或可展開

# Level 2
c1, c2, c3, c4 = st.sidebar.columns([1,1,1,1])
vix_en_2 = c1.checkbox("Lv2", True)
vix_b_trig_2 = c2.number_input("Lv2 買入>", 25.0)
vix_b_pct_2 = c3.number_input("Lv2 買金%", 20.0)
vix_s_trig_2 = c4.number_input("Lv2 賣出<", 20.0)
vix_s_pct_2 = 50.0

# Level 3
c1, c2, c3, c4 = st.sidebar.columns([1,1,1,1])
vix_en_3 = c1.checkbox("Lv3", True)
vix_b_trig_3 = c2.number_input("Lv3 買入>", 30.0)
vix_b_pct_3 = c3.number_input("Lv3 買金%", 30.0)
vix_s_trig_3 = c4.number_input("Lv3 賣出<", 25.0)
vix_s_pct_3 = 100.0

config.update({
    'vix_en_1': vix_en_1, 'vix_b_trig_1': vix_b_trig_1, 'vix_b_pct_1': vix_b_pct_1, 'vix_s_trig_1': vix_s_trig_1, 'vix_s_pct_1': vix_s_pct_1,
    'vix_en_2': vix_en_2, 'vix_b_trig_2': vix_b_trig_2, 'vix_b_pct_2': vix_b_pct_2, 'vix_s_trig_2': vix_s_trig_2, 'vix_s_pct_2': vix_s_pct_2,
    'vix_en_3': vix_en_3, 'vix_b_trig_3': vix_b_trig_3, 'vix_b_pct_3': vix_b_pct_3, 'vix_s_trig_3': vix_s_trig_3, 'vix_s_pct_3': vix_s_pct_3,
})

# --- 其他指標 ---
st.sidebar.subheader("3. 其他指標 (獨立運作)")
with st.sidebar.expander("詳細參數", expanded=False):
    use_ma = st.checkbox("MA", True)
    ma_len = st.number_input("MA 週期", 20)
    ma_buy_pct = st.number_input("MA 買入 %", 10.0)
    ma_sell_pct = st.number_input("MA 賣出 %", 100.0)

    use_roc = st.checkbox("ROC", False)
    roc_len = st.number_input("ROC 週期", 12)
    roc_buy_trig = st.number_input("ROC 買入 >", 0.0)
    roc_buy_pct = st.number_input("ROC 買入 %", 10.0)
    roc_sell_trig = st.number_input("ROC 賣出 <", -2.0)
    roc_sell_pct = st.number_input("ROC 賣出 %", 100.0)

    use_adx = st.checkbox("ADX", False)
    adx_len = st.number_input("ADX 週期", 14)
    adx_buy_trig = st.number_input("ADX 買入 >", 25.0)
    adx_buy_pct = st.number_input("ADX 買入 %", 10.0)
    adx_sell_trig = st.number_input("ADX 賣出 <", 20.0)
    adx_sell_pct = st.number_input("ADX 賣出 %", 100.0)

config.update({
    'use_ma': use_ma, 'ma_len': ma_len, 'ma_buy_pct': ma_buy_pct, 'ma_sell_pct': ma_sell_pct,
    'use_roc': use_roc, 'roc_len': roc_len, 'roc_buy_trig': roc_buy_trig, 'roc_buy_pct': roc_buy_pct, 'roc_sell_trig': roc_sell_trig, 'roc_sell_pct': roc_sell_pct,
    'use_adx': use_adx, 'adx_len': adx_len, 'adx_buy_trig': adx_buy_trig, 'adx_buy_pct': adx_buy_pct, 'adx_sell_trig': adx_sell_trig, 'adx_sell_pct': adx_sell_pct,
})

btn = st.sidebar.button("🚀 執行修正回測", type="primary")

# ==========================================
# 5. 主程式
# ==========================================
if btn:
    gc.collect()
    with st.spinner("正在模擬階梯加碼..."):
        df = get_data(symbol, start_date)
        if df.empty: st.error("無數據"); st.stop()

        god_curve = calculate_god_mode(df, init_cash)
        
        cerebro = bt.Cerebro()
        cerebro.adddata(PandasDataPlus(dataname=df))
        cerebro.addstrategy(LadderStrategy, config=config)
        cerebro.broker.setcash(init_cash)
        cerebro.broker.setcommission(commission=comm_pct/100.0)
        
        results = cerebro.run()
        strat = results[0]
        final_val = cerebro.broker.getvalue()
        
        # 數據整理
        dates = df.index[-len(strat.value_history):]
        eq_data = [{"time": d.strftime('%Y-%m-%d'), "value": v} for d, v in zip(dates, strat.value_history)]
        cash_data = [{"time": d.strftime('%Y-%m-%d'), "value": v} for d, v in zip(dates, strat.cash_history)]
        trade_log = pd.DataFrame(strat.trade_list)
        bh_val = (df['Close'].iloc[-1] / df['Close'].iloc[0]) * init_cash

    # UI
    st.title(f"🪜 {symbol} 階梯加碼戰報")
    c1, c2, c3 = st.columns(3)
    c1.metric("最終權益", f"${final_val:,.0f}", delta=f"{((final_val-init_cash)/init_cash)*100:.1f}%")
    c2.metric("Buy & Hold", f"${bh_val:,.0f}", delta=f"{((bh_val-init_cash)/init_cash)*100:.1f}%")
    c3.metric("交易次數", f"{len(trade_log)}")

    # Chart
    st.subheader("📈 資金曲線")
    chart_opts = {
        "chart": {"height": 350, "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"}},
        "series": [
            {"type": "Line", "data": god_curve, "options": {"color": "#FFD700", "lineWidth": 1, "title": "God Mode"}},
            {"type": "Area", "data": eq_data, "options": {"lineColor": "#00E676", "topColor": "rgba(0, 230, 118, 0.2)", "bottomColor": "rgba(0,0,0,0)", "title": "策略權益"}},
            {"type": "Area", "data": cash_data, "options": {"lineColor": "#2962FF", "topColor": "rgba(41,98,255,0.4)", "bottomColor": "rgba(41,98,255,0.1)", "title": "現金水位"}}
        ]
    }
    renderLightweightCharts([chart_opts], key="main")

    # K線 + 買賣點可視化
    st.subheader("🕯️ 買賣點還原")
    kline_data = [{"time": i.strftime('%Y-%m-%d'), "open": r['Open'], "high": r['High'], "low": r['Low'], "close": r['Close']} for i, r in df.iterrows()]
    series = [{"type": 'Candlestick', "data": kline_data, "options": {"upColor": '#089981', "downColor": '#f23645'}}]
    
    if not trade_log.empty:
        markers = []
        for _, t in trade_log.iterrows():
            is_buy = t['Type'] == 'Buy'
            color = "#00E676" if is_buy else "#FF5252"
            text = t['Reason'].split('_')[1] if '_' in t['Reason'] else t['Reason'] # 簡化標籤顯示
            markers.append({
                "time": t['Date'].strftime('%Y-%m-%d'),
                "position": "belowBar" if is_buy else "aboveBar",
                "color": color,
                "shape": "arrowUp" if is_buy else "arrowDown",
                "text": text
            })
        series[0]["markers"] = markers
    
    renderLightweightCharts([{"chart": {"height": 500, "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"}}, "series": series}], key="candle")

    if not trade_log.empty:
        st.subheader("📋 交易明細")
        trade_log['Amount'] = trade_log['Price'] * trade_log['Size']
        display_df = trade_log.copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        display_df['Price'] = display_df['Price'].map('{:,.2f}'.format)
        display_df['Amount'] = display_df['Amount'].map('{:,.0f}'.format)
        
        def highlight(row):
            c = '#00E676' if row['Type']=='Buy' else '#FF5252'
            return [f'color: {c}'] * len(row)
        st.dataframe(display_df.style.apply(highlight, axis=1), use_container_width=True)
