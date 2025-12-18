import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import backtrader as bt
import datetime
import plotly.graph_objects as go
from streamlit_lightweight_charts import renderLightweightCharts
import collections.abc

# 兼容性修復
collections.Iterable = collections.abc.Iterable

st.set_page_config(page_title="全球宏觀量化戰情室 v7.2", layout="wide")
st.markdown("<style>.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True)

# ==========================================
# 1. 數據代碼字典
# ==========================================
MACRO_TICKERS = {
    "🇺🇸 10年美債殖利率": "^TNX",
    "💵 美元指數 (DXY)": "DX-Y.NYB",
    "🌊 恐慌指數 (VIX)": "^VIX",
    "🇯🇵 日經 225": "^N225",
    "🇺🇸 標普 500 (SPY)": "SPY",
    "🇺🇸 那斯達克 (QQQ)": "QQQ"
}

# ==========================================
# 2. Backtrader 策略核心 (真正的邏輯)
# ==========================================
class RealStrategy(bt.Strategy):
    params = (('config', {}),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.cfg = self.params.config
        self.order = None
        self.trade_list = [] # 紀錄交易明細
        self.inds = {}

        # --- A. 動態建立指標 ---
        # 1. SMA
        if 'SMA' in self.cfg['indicators']:
            self.inds['sma'] = bt.indicators.SMA(self.datas[0], period=self.cfg['sma_len'])
        
        # 2. RSI
        if 'RSI' in self.cfg['indicators']:
            self.inds['rsi'] = bt.indicators.RSI(self.datas[0], period=self.cfg['rsi_len'])

        # 3. MACD
        if 'MACD' in self.cfg['indicators']:
            self.inds['macd'] = bt.indicators.MACD(self.datas[0], 
                                                   period_me1=self.cfg['macd_fast'], 
                                                   period_me2=self.cfg['macd_slow'], 
                                                   period_signal=self.cfg['macd_signal'])
        
        # 4. BBands
        if 'BBands' in self.cfg['indicators']:
            self.inds['bbands'] = bt.indicators.BollingerBands(self.datas[0], period=self.cfg['bb_len'], devfactor=self.cfg['bb_dev'])

    def notify_order(self, order):
        if order.status in [order.Completed]:
            # 紀錄交易
            if order.isbuy():
                self.trade_list.append({'Type': 'Buy', 'Date': bt.num2date(order.executed.dt), 'Price': order.executed.price, 'Size': order.executed.size})
            elif order.issell():
                self.trade_list.append({'Type': 'Sell', 'Date': bt.num2date(order.executed.dt), 'Price': order.executed.price, 'Size': order.executed.size})
            self.order = None

    def next(self):
        if self.order: return

        # --- B. 停損停利 (優先) ---
        if self.position:
            pct_change = (self.dataclose[0] - self.position.price) / self.position.price
            if pct_change < -self.cfg['stop_loss']: # 停損
                self.close()
                return
            if pct_change > self.cfg['take_profit']: # 止盈
                self.close()
                return

        # --- C. 進場邏輯 (AND) ---
        buy_signal = False
        conditions = []
        inds_on = self.cfg['indicators']

        # 1. SMA: 收盤價 > SMA (趨勢向上)
        if 'SMA' in inds_on:
            conditions.append(self.dataclose[0] > self.inds['sma'][0])
        
        # 2. RSI: RSI < 買入閥值 (超賣)
        if 'RSI' in inds_on:
            conditions.append(self.inds['rsi'][0] < self.cfg['rsi_buy'])

        # 3. MACD: 柱狀圖 > 0 (動能翻紅)
        if 'MACD' in inds_on:
            conditions.append(self.inds['macd'].macd[0] > self.inds['macd'].signal[0])
            
        # 4. BBands: 收盤價觸碰下軌 (超跌)
        if 'BBands' in inds_on:
            conditions.append(self.dataclose[0] < self.inds['bbands'].bot[0])

        # 綜合判斷: 如果有選指標，且所有條件都成立
        if inds_on and all(conditions):
            buy_signal = True
        
        # --- D. 執行 ---
        if not self.position and buy_signal:
            # 資金管理：每次投入總資金的 N%
            cash = self.broker.getcash()
            size = int((cash * self.cfg['trade_pct']) / self.dataclose[0])
            if size > 0: self.buy(size=size)
            
        elif self.position:
            # 出場邏輯 (反向訊號)
            sell_conds = []
            if 'SMA' in inds_on: sell_conds.append(self.dataclose[0] < self.inds['sma'][0])
            if 'RSI' in inds_on: sell_conds.append(self.inds['rsi'][0] > self.cfg['rsi_sell'])
            if 'BBands' in inds_on: sell_conds.append(self.dataclose[0] > self.inds['bbands'].top[0])
            
            if any(sell_conds):
                self.close()

# ==========================================
# 3. 介面邏輯 (左側設定)
# ==========================================
st.sidebar.header("🌍 策略中控台")

with st.sidebar.expander("1. 標的與資金", expanded=True):
    symbol = st.text_input("主代號", "NVDA")
    start_date = st.date_input("開始", datetime.date(2022, 1, 1))
    init_cash = st.number_input("初始本金", 100000)
    trade_pct = st.slider("每次倉位 %", 10, 100, 50) / 100.0
    c1, c2 = st.columns(2)
    stop_loss = c1.number_input("停損 %", 10.0) / 100.0
    take_profit = c2.number_input("止盈 %", 50.0) / 100.0

with st.sidebar.expander("2. 買賣策略設定 (Conditions)", expanded=True):
    # 指標選擇
    tech_inds = []
    c1, c2 = st.columns(2)
    if c1.checkbox("SMA (趨勢)", True): tech_inds.append("SMA")
    if c2.checkbox("RSI (震盪)", True): tech_inds.append("RSI")
    c3, c4 = st.columns(2)
    if c3.checkbox("MACD (波段)"): tech_inds.append("MACD")
    if c4.checkbox("BBands (通道)"): tech_inds.append("BBands")

    config = {
        'indicators': tech_inds, 'cash': init_cash, 
        'trade_pct': trade_pct, 'stop_loss': stop_loss, 'take_profit': take_profit
    }
    
    st.divider()
    st.caption("⚙️ 指標參數微調")
    
    if "SMA" in tech_inds:
        config['sma_len'] = st.number_input("SMA 週期 (大於此線買進)", 20)
    
    if "RSI" in tech_inds:
        config['rsi_len'] = 14
        c_r1, c_r2 = st.columns(2)
        config['rsi_buy'] = c_r1.number_input("RSI 買入 < (超賣)", 10, 50, 30)
        config['rsi_sell'] = c_r2.number_input("RSI 賣出 > (超買)", 50, 90, 70)
        
    if "MACD" in tech_inds:
        config['macd_fast'] = 12
        config['macd_slow'] = 26
        config['macd_signal'] = 9
        st.caption("MACD 邏輯：快線向上穿越慢線買入")
        
    if "BBands" in tech_inds:
        config['bb_len'] = 20
        config['bb_dev'] = 2.0
        st.caption("布林邏輯：觸碰下軌買入，觸碰上軌賣出")

with st.sidebar.expander("3. 宏觀疊加", expanded=False):
    selected_macros = st.multiselect("副圖宏觀指標", list(MACRO_TICKERS.keys()), default=["🌊 恐慌指數 (VIX)"])

btn_run = st.sidebar.button("🚀 執行完整回測", type="primary")

# ==========================================
# 4. 主程式邏輯
# ==========================================
if btn_run:
    # --- A. 數據下載 ---
    with st.spinner("運算中..."):
        df = yf.download(symbol, start=start_date, end=datetime.date.today(), progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # 宏觀數據
        macro_data = {}
        for m_name in selected_macros:
            try:
                m_df = yf.download(MACRO_TICKERS[m_name], start=start_date, end=datetime.date.today(), progress=False)
                if not m_df.empty:
                    if isinstance(m_df.columns, pd.MultiIndex): m_df.columns = m_df.columns.get_level_values(0)
                    macro_data[m_name] = m_df['Close']
            except: pass

    # --- B. 執行 Backtrader 回測 ---
    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.addstrategy(RealStrategy, config=config)
    cerebro.broker.setcash(init_cash)
    
    # 加入分析器 (計算報酬與回撤)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    
    results = cerebro.run()
    strat = results[0]
    
    # --- C. 計算「策略」與「Buy & Hold」的曲線 (重點) ---
    
    # 1. 策略淨值 (Strategy Equity)
    t_ret = strat.analyzers.timereturn.get_analysis()
    # 將回報率轉為每日淨值: 初始資金 * (1 + 累積回報率)
    equity_series = pd.Series(t_ret).fillna(0)
    equity_curve = (1 + equity_series).cumprod() * init_cash
    
    # 2. Buy & Hold 基準 (Benchmark Equity)
    # 模擬如果你第一天就把錢全部買這檔股票
    bh_ret = df['Close'].pct_change().fillna(0)
    # 確保日期索引對齊
    bh_ret = bh_ret.reindex(equity_curve.index).fillna(0) 
    bh_curve = (1 + bh_ret).cumprod() * init_cash
    
    # 績效計算
    strat_final = equity_curve.iloc[-1]
    bh_final = bh_curve.iloc[-1]
    strat_pct = (strat_final - init_cash) / init_cash * 100
    bh_pct = (bh_final - init_cash) / init_cash * 100
    
    mdd = strat.analyzers.drawdown.get_analysis()['max']['drawdown']

    # --- D. 介面呈現 ---
    
    # 1. 績效總結區
    st.subheader("💰 績效大對決")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("策略最終淨值", f"${strat_final:,.0f}", f"{strat_pct:.2f}%")
    col2.metric("Buy & Hold (傻瓜持有)", f"${bh_final:,.0f}", f"{bh_pct:.2f}%")
    col3.metric("超額報酬 (Alpha)", f"{strat_pct - bh_pct:.2f}%", "你的策略 vs 大盤")
    col4.metric("最大回撤 (風險)", f"{mdd:.2f}%", help="資產從最高點下跌的最大幅度")

    # 2. 雙曲線圖表 (Plotly)
    st.subheader("📈 獲利曲線比較")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name='我的策略', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=bh_curve.index, y=bh_curve.values, mode='lines', name='Buy & Hold', line=dict(color='gray', dash='dash')))
    fig.add_trace(go.Scatter(x=equity_curve.index[equity_curve.index.isin([x['Date'] for x in strat.trade_list if x['Type']=='Buy'])], 
                             y=equity_curve[equity_curve.index.isin([x['Date'] for x in strat.trade_list if x['Type']=='Buy'])],
                             mode='markers', name='買點', marker=dict(color='green', size=8, symbol='triangle-up')))
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # 3. Canvas K 線圖 (含指標與宏觀)
    st.divider()
    
    # (準備數據 - 略為精簡，保留你的 Canvas 需求)
    kline_data = []
    for idx, row in df.iterrows():
        kline_data.append({"time": idx.strftime('%Y-%m-%d'), "open": float(row['Open']), "high": float(row['High']), "low": float(row['Low']), "close": float(row['Close'])})
    
    charts_to_render = [
        {"chart": {"height": 400}, "series": [{"type": 'Candlestick', "data": kline_data, "options": {"upColor": '#26a69a', "downColor": '#ef5350'}}]}
    ]
    
    # 如果有 RSI，畫出來
    if "RSI" in tech_inds:
        rsi_vals = ta.rsi(df['Close'], length=config['rsi_len'])
        rsi_data = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in rsi_vals.items() if pd.notnull(v)]
        charts_to_render.append({"chart": {"height": 150}, "series": [{"type": "Line", "data": rsi_data, "options": {"color": "purple", "title": "RSI"}}]})
    
    # 宏觀指標
    if macro_data:
         m_list = []
         colors = ['#FF9800', '#2962FF']
         for i, (m_name, m_series) in enumerate(macro_data.items()):
             m_data = [{"time": idx.strftime('%Y-%m-%d'), "value": float(val)} for idx, val in m_series.items() if idx in df.index]
             m_list.append({"type": "Line", "data": m_data, "options": {"color": colors[i%2], "title": m_name}})
         charts_to_render.append({"chart": {"height": 200, "layout": {"background": {"color": "#f0f2f6"}}, "title": "宏觀數據"}, "series": m_list})

    renderLightweightCharts(charts_to_render, key="v7_2_chart")

    # 4. 交易明細
    st.subheader("📋 交易日記")
    if strat.trade_list:
        df_trade = pd.DataFrame(strat.trade_list)
        df_trade['Date'] = df_trade['Date'].dt.strftime('%Y-%m-%d')
        df_trade['Price'] = df_trade['Price'].round(2)
        df_trade['Value'] = (df_trade['Price'] * df_trade['Size']).round(0)
        st.dataframe(df_trade, use_container_width=True)
    else:
        st.warning("在此回測期間內，您的策略條件太嚴格，未觸發任何交易。")

else:
    st.info("👈 請在左側設定「買賣條件」與「指標參數」，開始回測。")
