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

st.set_page_config(page_title="全指標回測實驗室 v12", layout="wide")
st.markdown("""
<style>
    .stMetric {background-color: #f8f9fa; padding: 10px; border-radius: 5px; border: 1px solid #dee2e6;}
    .block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 全球宏觀資產清單 (新增黃金/白銀)
# ==========================================
MACRO_ASSETS = {
    "VIX (恐慌指數)": "^VIX",
    "Gold (黃金)": "GC=F",
    "Silver (白銀)": "SI=F",
    "Oil (原油)": "CL=F",
    "10Y Bond (美債)": "^TNX",
    "DXY (美元指數)": "DX-Y.NYB"
}

# 指標說明庫 (教育用途)
IND_INFO = {
    "SMA": "價格高於均線買入 (趨勢多頭)",
    "EMA": "價格高於指數均線買入 (反應較快)",
    "RSI": "數值低於設定值買入 (超賣反彈)",
    "MACD": "快線(DIF)向上穿越慢線(DEM)買入 (黃金交叉)",
    "BBands": "價格跌破布林下軌買入 (超跌)",
    "KD": "K值由下往上穿越D值買入 (低檔金叉)",
    "ADX": "ADX數值大於25 (趨勢確立) 且 +DI > -DI"
}

# ==========================================
# 2. Backtrader 策略核心 (通用邏輯引擎)
# ==========================================
class UniversalStrategy(bt.Strategy):
    params = (('config', {}),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.c = self.params.config
        
        self.order = None
        self.trade_list = [] # 紀錄明細
        self.inds = {}

        # --- A. 動態初始化指標 ---
        # 趨勢
        if 'SMA' in self.c['indicators']:
            self.inds['sma'] = bt.indicators.SMA(self.datas[0], period=self.c['sma_len'])
        if 'EMA' in self.c['indicators']:
            self.inds['ema'] = bt.indicators.EMA(self.datas[0], period=self.c['ema_len'])
        if 'ADX' in self.c['indicators']:
            self.inds['adx'] = bt.indicators.ADX(self.datas[0], period=14)
            
        # 震盪
        if 'RSI' in self.c['indicators']:
            self.inds['rsi'] = bt.indicators.RSI(self.datas[0], period=14)
        if 'MACD' in self.c['indicators']:
            self.inds['macd'] = bt.indicators.MACD(self.datas[0])
        if 'KD' in self.c['indicators']:
            self.inds['kd'] = bt.indicators.Stochastic(self.datas[0])
            
        # 通道
        if 'BBands' in self.c['indicators']:
            self.inds['bbands'] = bt.indicators.BollingerBands(self.datas[0], period=20, devfactor=2.0)

    def notify_order(self, order):
        if order.status in [order.Completed]:
            # 紀錄交易明細 (含手續費)
            if order.isbuy():
                self.trade_list.append({
                    'Type': 'Buy', 'Date': bt.num2date(order.executed.dt),
                    'Price': order.executed.price, 'Size': order.executed.size,
                    'Comm': order.executed.comm, 'Value': order.executed.value
                })
            elif order.issell():
                self.trade_list.append({
                    'Type': 'Sell', 'Date': bt.num2date(order.executed.dt),
                    'Price': order.executed.price, 'Size': order.executed.size,
                    'Comm': order.executed.comm, 'Value': order.executed.value
                })
            self.order = None

    def next(self):
        if self.order: return

        # --- B. 條件檢查 (Signal Logic) ---
        buy_conditions = []
        sell_conditions = []
        inds_on = self.c['indicators']

        # 1. SMA (價格 > SMA 視為多頭)
        if 'SMA' in inds_on:
            buy_conditions.append(self.dataclose[0] > self.inds['sma'][0])
            sell_conditions.append(self.dataclose[0] < self.inds['sma'][0])

        # 2. EMA
        if 'EMA' in inds_on:
            buy_conditions.append(self.dataclose[0] > self.inds['ema'][0])
            sell_conditions.append(self.dataclose[0] < self.inds['ema'][0])

        # 3. RSI (低於買入閥值 / 高於賣出閥值)
        if 'RSI' in inds_on:
            buy_conditions.append(self.inds['rsi'][0] < self.c['rsi_buy'])
            sell_conditions.append(self.inds['rsi'][0] > self.c['rsi_sell'])

        # 4. MACD (DIF > DEM 金叉 / 死叉)
        if 'MACD' in inds_on:
            buy_conditions.append(self.inds['macd'].macd[0] > self.inds['macd'].signal[0])
            sell_conditions.append(self.inds['macd'].macd[0] < self.inds['macd'].signal[0])

        # 5. BBands (跌破下軌買 / 突破上軌賣)
        if 'BBands' in inds_on:
            buy_conditions.append(self.dataclose[0] < self.inds['bbands'].bot[0])
            sell_conditions.append(self.dataclose[0] > self.inds['bbands'].top[0])

        # 6. KD (K > D 金叉 / 死叉)
        if 'KD' in inds_on:
            buy_conditions.append(self.inds['kd'].percK[0] > self.inds['kd'].percD[0])
            sell_conditions.append(self.inds['kd'].percK[0] < self.inds['kd'].percD[0])
            
        # 7. ADX (趨勢強度濾網)
        if 'ADX' in inds_on:
            # ADX 只當作買入濾網：ADX > 25 且 +DI > -DI 代表多頭趨勢強
            buy_conditions.append(self.inds['adx'].adx[0] > 25 and self.inds['adx'].DIplus[0] > self.inds['adx'].DIminus[0])
            # ADX 通常不單獨做為賣出訊號，這裡簡化處理

        # --- C. 執行交易 ---
        
        # 進場：目前沒持倉 + 所有勾選指標都符合 (AND 邏輯)
        if not self.position:
            if inds_on and all(buy_conditions):
                # 資金管理：每次投入本金的 N%
                cash = self.broker.getcash()
                target_amt = self.broker.getvalue() * self.c['invest_pct']
                size = int(target_amt / self.dataclose[0])
                if size > 0: self.buy(size=size)

        # 出場：持有中 + 任意指標發出賣訊 (OR 邏輯 - 比較安全) 
        # *註：也可以改為 AND，看你想做長線還是短線
        elif self.position:
            # 停損優先
            pct_chg = (self.dataclose[0] - self.position.price) / self.position.price
            if pct_chg < -self.c['stop_loss']:
                self.close()
                return

            if inds_on and any(sell_conditions): # 只要有一個指標說賣就賣 (保守)
                self.close()

# ==========================================
# 3. 介面設定 (左側參數)
# ==========================================
st.sidebar.header("🔬 策略參數實驗室")

with st.sidebar.expander("1. 數據設定", expanded=True):
    symbol = st.text_input("股票代碼", "NVDA")
    interval = st.selectbox("K線週期", ["1d (日線)", "1h (60分K)"])
    interval_code = "1h" if "1h" in interval else "1d"
    
    # 時間限制提醒
    if interval_code == "1h":
        st.caption("⚠️ 小時線限制最近 730 天")
        start_date = st.date_input("開始", datetime.date.today() - datetime.timedelta(days=365))
    else:
        start_date = st.date_input("開始", datetime.date(2022, 1, 1))
    
    end_date = st.date_input("結束", datetime.date.today())

with st.sidebar.expander("2. 宏觀避險對照", expanded=True):
    macro_selected = st.multiselect("顯示於副圖", list(MACRO_ASSETS.keys()), default=["Gold (黃金)", "VIX (恐慌指數)"])

with st.sidebar.expander("3. 買賣條件設定 (AND)", expanded=True):
    st.info("勾選越多，條件越嚴格 (同時成立才買)")
    
    # 指標選擇器
    tech_inds = []
    
    st.markdown("**趨勢類**")
    c1, c2, c3 = st.columns(3)
    if c1.checkbox("SMA", True): tech_inds.append("SMA")
    if c2.checkbox("EMA"): tech_inds.append("EMA")
    if c3.checkbox("ADX"): tech_inds.append("ADX")
    
    st.markdown("**震盪/反轉類**")
    c4, c5, c6 = st.columns(3)
    if c4.checkbox("RSI", True): tech_inds.append("RSI")
    if c5.checkbox("MACD"): tech_inds.append("MACD")
    if c6.checkbox("KD"): tech_inds.append("KD")
    
    st.markdown("**通道類**")
    if st.checkbox("BBands (布林)"): tech_inds.append("BBands")

    # 參數微調區
    st.divider()
    st.caption("⚙️ 參數微調")
    config = {'indicators': tech_inds}
    
    if "SMA" in tech_inds: config['sma_len'] = st.number_input("SMA 週期", 20)
    if "EMA" in tech_inds: config['ema_len'] = st.number_input("EMA 週期", 20)
    if "RSI" in tech_inds: 
        c_r1, c_r2 = st.columns(2)
        config['rsi_buy'] = c_r1.number_input("RSI 買點 <", 30)
        config['rsi_sell'] = c_r2.number_input("RSI 賣點 >", 70)

with st.sidebar.expander("4. 資金與手續費", expanded=True):
    init_cash = st.number_input("初始本金", 100000)
    invest_pct = st.slider("每次投入資金 %", 10, 100, 50) / 100.0
    comm_rate = st.number_input("手續費率 (%)", 0.0, 1.0, 0.1425, format="%.4f") / 100.0
    stop_loss = st.number_input("強制停損 %", 10.0) / 100.0
    
    config.update({'invest_pct': invest_pct, 'stop_loss': stop_loss})

btn_run = st.sidebar.button("🚀 執行模擬回測", type="primary")

# ==========================================
# 4. 主程式邏輯
# ==========================================
if btn_run:
    with st.spinner("正在下載數據、計算指標、模擬交易..."):
        # 1. 下載主數據
        df = yf.download(symbol, start=start_date, end=end_date, interval=interval_code, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # 2. 下載宏觀數據
        macro_data = {}
        for m_name in macro_selected:
            try:
                m_ticker = MACRO_ASSETS[m_name]
                m_df = yf.download(m_ticker, start=start_date, end=end_date, interval="1d", progress=False) # 宏觀通常看日線
                if not m_df.empty:
                    if isinstance(m_df.columns, pd.MultiIndex): m_df.columns = m_df.columns.get_level_values(0)
                    # 如果主圖是小時線，宏觀數據需要 ffill 對齊
                    m_series = m_df['Close'].reindex(df.index, method='ffill')
                    macro_data[m_name] = m_series
            except: pass

        if df.empty:
            st.error("❌ 查無數據，請檢查代碼或日期。")
            st.stop()

        # 3. 執行 Backtrader
        cerebro = bt.Cerebro()
        cerebro.adddata(bt.feeds.PandasData(dataname=df))
        cerebro.addstrategy(UniversalStrategy, config=config)
        cerebro.broker.setcash(init_cash)
        cerebro.broker.setcommission(commission=comm_rate)
        
        # 分析器
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        
        results = cerebro.run()
        strat = results[0]

        # 4. 數據整理
        # 資金曲線
        t_ret = strat.analyzers.timereturn.get_analysis()
        equity_curve = pd.Series(t_ret).fillna(0)
        equity_curve = (1 + equity_curve).cumprod() * init_cash
        
        # Buy & Hold 曲線
        bh_ret = df['Close'].pct_change().fillna(0)
        bh_curve = (1 + bh_ret).cumprod() * init_cash
        
        # 績效指標
        final_val = equity_curve.iloc[-1]
        roi = (final_val - init_cash) / init_cash * 100
        bh_roi = (bh_curve.iloc[-1] - init_cash) / init_cash * 100
        mdd = strat.analyzers.drawdown.get_analysis()['max']['drawdown']
        
        # 交易明細
        trade_log = pd.DataFrame(strat.trade_list)
        if not trade_log.empty:
            trade_log['Date'] = trade_log['Date'].dt.strftime('%Y-%m-%d %H:%M')
            trade_log['Value'] = trade_log['Value'].abs().round(0)
            trade_log['Comm'] = trade_log['Comm'].round(2)

    # ==========================================
    # 5. 結果呈現 (Dashboard)
    # ==========================================
    st.title(f"📊 {symbol} 策略回測報告 ({interval_code})")
    
    # A. 績效看板
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最終權益", f"${final_val:,.0f}", f"{roi:.2f}%")
    c2.metric("Buy & Hold", f"${bh_curve.iloc[-1]:,.0f}", f"{bh_roi:.2f}%")
    c3.metric("最大回撤 (MDD)", f"{mdd:.2f}%", help="越低越好")
    c4.metric("交易次數", len(trade_log) if not trade_log.empty else 0)

    # B. 資金曲線對比 (Plotly)
    st.subheader("📈 獲利能力曲線 (Equity Curve)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name='我的策略', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=bh_curve.index, y=bh_curve.values, mode='lines', name='Buy & Hold', line=dict(color='gray', dash='dash')))
    
    # 標記買賣點
    if not trade_log.empty:
        buys = trade_log[trade_log['Type'] == 'Buy']
        sells = trade_log[trade_log['Type'] == 'Sell']
        # 注意：這裡的時間要跟 equity_curve index 對齊可能會有微小誤差，Plotly 能自動處理大部分
        # 為了準確顯示在 K 線上，我們主要在下方的 LWC 顯示，這裡僅畫曲線
        
    st.plotly_chart(fig, use_container_width=True)

    # C. 專業 K 線圖 (LWC)
    st.subheader("🕯️ 價格與指標詳情")
    
    # 準備數據
    kline_data = [{"time": i.strftime('%Y-%m-%d %H:%M') if interval_code=='1h' else i.strftime('%Y-%m-%d'), 
                   "open": r['Open'], "high": r['High'], "low": r['Low'], "close": r['Close']} for i, r in df.iterrows()]
    
    series_main = [{
        "type": 'Candlestick',
        "data": kline_data,
        "options": {"upColor": '#26a69a', "downColor": '#ef5350', "borderVisible": False}
    }]
    
    # 疊加指標 (預先計算供繪圖)
    if "SMA" in tech_inds:
        sma = ta.sma(df['Close'], length=config.get('sma_len', 20))
        d = [{"time": i.strftime('%Y-%m-%d %H:%M') if interval_code=='1h' else i.strftime('%Y-%m-%d'), "value": v} for i, v in sma.items() if not pd.isna(v)]
        series_main.append({"type": "Line", "data": d, "options": {"color": "yellow", "lineWidth": 2, "title": "SMA"}})
        
    if "BBands" in tech_inds:
        bb = ta.bbands(df['Close'])
        if bb is not None:
            # 模糊抓取
            bbu = bb[bb.columns[0]]; bbl = bb[bb.columns[2]] 
            d_u = [{"time": i.strftime('%Y-%m-%d %H:%M') if interval_code=='1h' else i.strftime('%Y-%m-%d'), "value": v} for i, v in bbu.items() if not pd.isna(v)]
            d_l = [{"time": i.strftime('%Y-%m-%d %H:%M') if interval_code=='1h' else i.strftime('%Y-%m-%d'), "value": v} for i, v in bbl.items() if not pd.isna(v)]
            series_main.append({"type": "Line", "data": d_u, "options": {"color": "rgba(0,100,255,0.3)"}})
            series_main.append({"type": "Line", "data": d_l, "options": {"color": "rgba(0,100,255,0.3)"}})

    # 買賣標記
    markers = []
    if not trade_log.empty:
        for _, t in trade_log.iterrows():
            markers.append({
                "time": pd.to_datetime(t['Date']).strftime('%Y-%m-%d %H:%M') if interval_code=='1h' else pd.to_datetime(t['Date']).strftime('%Y-%m-%d'),
                "position": "belowBar" if t['Type']=='Buy' else "aboveBar",
                "color": "green" if t['Type']=='Buy' else "red",
                "shape": "arrowUp" if t['Type']=='Buy' else "arrowDown",
                "text": t['Type']
            })
    series_main[0]["markers"] = markers
    
    charts = [{"chart": {"height": 400}, "series": series_main}]
    
    # 宏觀副圖
    if macro_data:
        m_list = []
        colors = ['#FF9800', '#E91E63', '#9C27B0', '#2962FF']
        for i, (name, series) in enumerate(macro_data.items()):
             d = [{"time": idx.strftime('%Y-%m-%d %H:%M') if interval_code=='1h' else idx.strftime('%Y-%m-%d'), "value": float(val)} for idx, val in series.items()]
             m_list.append({"type": "Line", "data": d, "options": {"color": colors[i%4], "title": name}})
        charts.append({"chart": {"height": 200, "title": "宏觀避險指標"}, "series": m_list})

    renderLightweightCharts(charts, key="final_result")

    # D. 交易明細表
    st.subheader("📋 交易明細 (Trade Log)")
    if not trade_log.empty:
        st.dataframe(trade_log, use_container_width=True)
        # 簡單統計
        total_comm = trade_log['Comm'].sum()
        st.info(f"🧾 本次策略共支付手續費: ${total_comm:.2f}")
    else:
        st.warning("⚠️ 策略條件太嚴格，期間內無交易。請放寬條件 (例如取消 ADX 或放寬 RSI)。")

else:
    st.info("👈 請在左側設定策略條件，並開始模擬。")
