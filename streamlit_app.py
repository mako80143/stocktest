import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import backtrader as bt
import datetime
import plotly.express as px
from streamlit_lightweight_charts import renderLightweightCharts
import collections.abc

# 兼容性修復
collections.Iterable = collections.abc.Iterable

st.set_page_config(page_title="全球宏觀量化戰情室 v7.0", layout="wide")
st.markdown("<style>.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True)

# ==========================================
# 1. 數據代碼字典 (全球宏觀)
# ==========================================
# 這裡定義了你要的所有宏觀指標
MACRO_TICKERS = {
    "🇺🇸 10年美債殖利率": "^TNX",
    "🇺🇸 2年美債殖利率 (期貨)": "ZT=F",  # Yahoo 2年債數據較難抓，用期貨或 SHY 代替
    "💵 美元指數 (DXY)": "DX-Y.NYB",
    "🇯🇵 日經 225": "^N225",
    "🇯🇵 日本東證 (Topix)": "^TOPX", # 假如抓不到可改 EWJ (ETF)
    "💴 美元/日圓 (USD/JPY)": "JPY=X",
    "🇺🇸 標普 500 (SPY)": "SPY",
    "🇺🇸 那斯達克 (QQQ)": "QQQ",
    "🇺🇸 道瓊工業 (DIA)": "DIA",
    "🌊 恐慌指數 (VIX)": "^VIX"
}

# 指標全家桶介紹
INDICATOR_LIB = {
    "Trend (趨勢)": {
        "SMA": "簡單移動平均。最基礎的趨勢線。",
        "EMA": "指數移動平均。對近期價格反應更快。",
        "ADX": "平均趨向指標。數值 > 25 代表有強趨勢 (不分多空)，< 20 代表盤整。",
        "Ichimoku": "一目均衡表。日本高級指標，包含雲帶、基準線、轉換線。",
        "Parabolic SAR": "拋物線轉向指標。點在K線下方做多，上方做空。"
    },
    "Oscillator (震盪)": {
        "RSI": "相對強弱。30超賣，70超買。",
        "MACD": "指數平滑異同。柱狀體與快慢線。",
        "KD": "隨機指標。尋找短線轉折。",
        "CCI": "順勢指標。>100 強勢，<-100 弱勢，適合抓突破。",
        "Williams %R": "威廉指標。反應極其靈敏的超買超賣指標。"
    },
    "Volatility/Volume (波動/量能)": {
        "BBands": "布林通道。壓縮後通常會有大行情。",
        "ATR": "平均真實波幅。用來設停損非常好用 (例如 2倍 ATR)。",
        "OBV": "能量潮。股價沒漲但 OBV 先漲，代表主力在吸籌。",
        "VWAP": "成交量加權平均價。當日交易者的平均成本線 (僅限分時圖)。"
    }
}

# ==========================================
# 2. Backtrader 策略 (支援擴充指標)
# ==========================================
class MacroStrategy(bt.Strategy):
    params = (('config', {}),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.cfg = self.params.config
        self.inds = {}
        
        # 為了簡化範例，這裡只示範動態加載基礎指標
        # 實際全指標回測需要寫很長的 if-else 對應
        if 'SMA' in self.cfg['indicators']:
             self.inds['sma'] = bt.indicators.SMA(self.datas[0], period=20)
        if 'RSI' in self.cfg['indicators']:
             self.inds['rsi'] = bt.indicators.RSI(self.datas[0], period=14)

    def next(self):
        if not self.position:
            # 簡易買入邏輯範例
            if 'SMA' in self.inds and self.dataclose[0] > self.inds['sma'][0]:
                self.buy(size=int(self.cfg['cash']*0.5 / self.dataclose[0]))
        else:
            if 'SMA' in self.inds and self.dataclose[0] < self.inds['sma'][0]:
                self.close()

# ==========================================
# 3. 介面邏輯 (左側設定)
# ==========================================
st.sidebar.header("🌍 全球戰情控制台")

with st.sidebar.expander("1. 標的與時間", expanded=True):
    symbol = st.text_input("主代號", "NVDA")
    start_date = st.date_input("開始", datetime.date(2022, 1, 1))
    end_date = st.date_input("結束", datetime.date.today())

with st.sidebar.expander("2. 宏觀疊加 (Macro Overlay)", expanded=True):
    st.caption("選擇要與 K 線同步對照的宏觀數據")
    selected_macros = st.multiselect("選擇宏觀指標", list(MACRO_TICKERS.keys()), default=["💵 美元指數 (DXY)", "🌊 恐慌指數 (VIX)"])

with st.sidebar.expander("3. 指標全家桶 (Canvas)", expanded=True):
    # 分類顯示
    tech_inds = []
    st.write("📈 **趨勢型**")
    cols = st.columns(3)
    if cols[0].checkbox("SMA", True): tech_inds.append("SMA")
    if cols[1].checkbox("EMA"): tech_inds.append("EMA")
    if cols[2].checkbox("Ichimoku"): tech_inds.append("Ichimoku")
    if st.checkbox("ADX (趨勢強度)"): tech_inds.append("ADX")
    
    st.write("🌊 **震盪型**")
    cols2 = st.columns(3)
    if cols2[0].checkbox("RSI", True): tech_inds.append("RSI")
    if cols2[1].checkbox("MACD", True): tech_inds.append("MACD")
    if cols2[2].checkbox("KD"): tech_inds.append("KD")
    if st.checkbox("CCI"): tech_inds.append("CCI")
    
    st.write("📊 **波動與量能**")
    cols3 = st.columns(2)
    if cols3[0].checkbox("BBands (布林)"): tech_inds.append("BBands")
    if cols3[1].checkbox("OBV (能量)"): tech_inds.append("OBV")
    if st.checkbox("ATR (波動率)"): tech_inds.append("ATR")

btn_run = st.sidebar.button("🚀 啟動戰情室", type="primary")

# ==========================================
# 4. 主程式邏輯
# ==========================================
if btn_run:
    # --- A. 數據下載區 ---
    with st.spinner("📡 正在連線全球交易所下載數據..."):
        # 1. 主數據
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        if df.empty:
            st.error("找不到主代號數據")
            st.stop()
            
        # 2. 宏觀數據 (Macro Fetching)
        macro_data = {}
        for m_name in selected_macros:
            ticker = MACRO_TICKERS[m_name]
            try:
                m_df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not m_df.empty:
                    if isinstance(m_df.columns, pd.MultiIndex): m_df.columns = m_df.columns.get_level_values(0)
                    macro_data[m_name] = m_df['Close']
            except Exception as e:
                st.warning(f"無法下載 {m_name}: {e}")

    # --- B. 介面分頁 (Tab) ---
    tab_chart, tab_corr, tab_backtest = st.tabs(["📊 宏觀 K 線戰情", "🔥 相關性熱力圖", "💰 回測數據"])

    # === Tab 1: 宏觀 K 線 (Canvas 重頭戲) ===
    with tab_chart:
        st.subheader(f"{symbol} 綜合技術分析")
        
        # 1. 計算技術指標 (pandas_ta)
        # 為了不讓畫面太亂，我們將數據計算好，並放入不同的 Pane (窗格)
        
        # 主圖指標
        sma_data = []
        bb_upper, bb_lower = [], []
        if "SMA" in tech_inds:
            df['SMA'] = ta.sma(df['Close'], length=20)
        if "BBands" in tech_inds:
            bb = ta.bbands(df['Close'], length=20, std=2.0)
            if bb is not None:
                df['BBU'] = bb[f'BBU_20_2.0']
                df['BBL'] = bb[f'BBL_20_2.0']

        # 副圖指標 (Sub-charts)
        rsi_vals = ta.rsi(df['Close'], length=14) if "RSI" in tech_inds else None
        macd = ta.macd(df['Close']) if "MACD" in tech_inds else None
        adx = ta.adx(df['High'], df['Low'], df['Close']) if "ADX" in tech_inds else None
        obv = ta.obv(df['Close'], df['Volume']) if "OBV" in tech_inds else None
        cci = ta.cci(df['High'], df['Low'], df['Close']) if "CCI" in tech_inds else None

        # 2. 轉換為 Lightweight Charts 格式
        kline_data = []
        vol_data = []
        
        # 宏觀數據對齊
        macro_series_data = {name: [] for name in macro_data}

        for idx, row in df.iterrows():
            t_str = idx.strftime('%Y-%m-%d')
            
            # K線
            kline_data.append({"time": t_str, "open": float(row['Open']), "high": float(row['High']), "low": float(row['Low']), "close": float(row['Close'])})
            # 量
            color = 'rgba(0, 150, 136, 0.5)' if row['Close'] > row['Open'] else 'rgba(255, 82, 82, 0.5)'
            vol_data.append({"time": t_str, "value": float(row['Volume']), "color": color})
            
            # 宏觀
            for m_name, m_series in macro_data.items():
                if idx in m_series.index and not pd.isna(m_series.loc[idx]):
                    macro_series_data[m_name].append({"time": t_str, "value": float(m_series.loc[idx])})

        # 3. 組合圖表 (Layering)
        charts_to_render = []
        
        # [Pane 0] 主圖: K線 + SMA + BBands
        main_series = [{
            "type": 'Candlestick',
            "data": kline_data,
            "options": {"upColor": '#26a69a', "downColor": '#ef5350', "borderVisible": False, "wickUpColor": '#26a69a', "wickDownColor": '#ef5350'}
        }]
        
        if "SMA" in tech_inds:
            sma_line = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in df['SMA'].items() if pd.notnull(v)]
            main_series.append({"type": "Line", "data": sma_line, "options": {"color": "yellow", "lineWidth": 2, "title": "SMA 20"}})
            
        if "BBands" in tech_inds and 'BBU' in df.columns:
            bbu_line = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in df['BBU'].items() if pd.notnull(v)]
            bbl_line = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in df['BBL'].items() if pd.notnull(v)]
            main_series.append({"type": "Line", "data": bbu_line, "options": {"color": "rgba(0, 150, 255, 0.5)", "lineWidth": 1}})
            main_series.append({"type": "Line", "data": bbl_line, "options": {"color": "rgba(0, 150, 255, 0.5)", "lineWidth": 1}})

        charts_to_render.append({"chart": {"height": 400, "layout": {"background": {"color": "white"}}, "crosshair": {"mode": 0}}, "series": main_series})

        # [Pane 1] 成交量 + OBV
        vol_series = [{"type": 'Histogram', "data": vol_data, "options": {"priceFormat": {"type": 'volume'}, "title": "Volume"}}]
        if "OBV" in tech_inds and obv is not None:
             obv_line = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in obv.items() if pd.notnull(v)]
             # OBV 數值很大，建議單獨放，這裡為示範疊加
             # vol_series.append({"type": "Line", "data": obv_line, "options": {"color": "blue", "priceScaleId": "right"}})
        
        charts_to_render.append({"chart": {"height": 100, "layout": {"background": {"color": "white"}}}, "series": vol_series})

        # [Pane 2] 宏觀疊加區 (Macro Pane)
        if macro_series_data:
            macro_series_list = []
            colors = ['#2962FF', '#E91E63', '#FF9800', '#9C27B0'] # 不同顏色
            for i, (m_name, m_data) in enumerate(macro_series_data.items()):
                color = colors[i % len(colors)]
                macro_series_list.append({
                    "type": "Line",
                    "data": m_data,
                    "options": {"color": color, "lineWidth": 2, "title": m_name}
                })
            
            charts_to_render.append({"chart": {"height": 200, "layout": {"background": {"color": "#f0f2f6"}}, "title": "全球宏觀趨勢對照"}, "series": macro_series_list})

        # [Pane 3] 技術指標副圖 (RSI/MACD/CCI/ADX)
        # 這裡為了展示 Canvas 的強大，我們動態生成多個 Pane
        
        if "RSI" in tech_inds and rsi_vals is not None:
            rsi_data = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in rsi_vals.items() if pd.notnull(v)]
            charts_to_render.append({
                "chart": {"height": 150},
                "series": [{"type": "Line", "data": rsi_data, "options": {"color": "purple", "title": "RSI (14)"}}]
            })
            
        if "MACD" in tech_inds and macd is not None:
            # 這裡簡化只畫 MACD Line
            macd_line = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in macd['MACD_12_26_9'].items() if pd.notnull(v)]
            hist_data = [{"time": i.strftime('%Y-%m-%d'), "value": float(v), "color": "green" if v>0 else "red"} for i, v in macd['MACDH_12_26_9'].items() if pd.notnull(v)]
            charts_to_render.append({
                "chart": {"height": 150},
                "series": [
                    {"type": "Line", "data": macd_line, "options": {"color": "blue", "title": "MACD"}},
                    {"type": "Histogram", "data": hist_data, "options": {"title": "Hist"}}
                ]
            })

        if "ADX" in tech_inds and adx is not None:
             adx_line = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in adx['ADX_14'].items() if pd.notnull(v)]
             charts_to_render.append({
                "chart": {"height": 150},
                "series": [{"type": "Line", "data": adx_line, "options": {"color": "red", "title": "ADX Trend Strength"}}]
            })

        # 最終渲染
        st.info("💡 提示：按住 Shift + 滑鼠滾輪可以快速瀏覽不同窗格。")
        renderLightweightCharts(charts_to_render, key="macro_canvas")

    # === Tab 2: 相關性熱力圖 (Pro Feature) ===
    with tab_corr:
        st.subheader("🔥 資產相關性矩陣 (Correlation Heatmap)")
        st.markdown("這張圖告訴你：**你的股票跟宏觀指標的連動性如何？**")
        st.markdown("- **紅色 (接近 1)**: 正相關 (例如：美股漲，你的股票就漲)")
        st.markdown("- **藍色 (接近 -1)**: 負相關 (例如：美元漲，你的股票就跌)")
        
        # 準備數據
        corr_df = pd.DataFrame()
        corr_df[symbol] = df['Close']
        for m_name, m_series in macro_data.items():
            corr_df[m_name] = m_series
        
        # 計算相關係數
        corr_matrix = corr_df.pct_change().corr()
        
        # 畫熱力圖
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig_corr, use_container_width=True)

    # === Tab 3: 回測數據 (Backtest) ===
    with tab_backtest:
        st.subheader("策略模擬結果")
        cerebro = bt.Cerebro()
        cerebro.adddata(bt.feeds.PandasData(dataname=df))
        cerebro.addstrategy(MacroStrategy, config={'cash': 100000, 'indicators': tech_inds})
        cerebro.broker.setcash(100000)
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
        res = cerebro.run()
        
        st.success("✅ 回測完成！這裡展示簡單的策略運算結果。若要深度回測，請結合 v6.0 的資金管理模組。")
        val = cerebro.broker.getvalue()
        st.metric("最終權益", f"${val:,.0f}")

else:
    st.info("👈 請在左側勾選你想看的指標與宏觀數據，然後點擊啟動。")
