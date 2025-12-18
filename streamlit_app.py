import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import datetime
from streamlit_lightweight_charts import renderLightweightCharts

# --- 1. 全局設定 ---
st.set_page_config(page_title="盤感訓練核心 v11", layout="wide")
st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 5px; font-weight: bold;}
    .metric-box {background-color: #f0f2f6; padding: 10px; border-radius: 8px; text-align: center;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心邏輯：數據預處理引擎
# ==========================================
def prepare_game_data(symbol, start, end, init_cash):
    """預先計算所有數據，包含指標、B&H曲線、上帝視角"""
    
    # A. 下載主數據
    df = yf.download(symbol, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    
    # B. 下載宏觀數據 (VIX, 10年美債)
    try:
        vix = yf.download("^VIX", start=start, end=end, progress=False)['Close']
        tnx = yf.download("^TNX", start=start, end=end, progress=False)['Close']
        if isinstance(vix, pd.DataFrame): vix = vix.iloc[:, 0]
        if isinstance(tnx, pd.DataFrame): tnx = tnx.iloc[:, 0]
        # 對齊索引
        df['VIX'] = vix.reindex(df.index).ffill()
        df['TNX'] = tnx.reindex(df.index).ffill()
    except:
        df['VIX'] = 0
        df['TNX'] = 0

    # C. 計算技術指標 (全家桶)
    # 趨勢
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_60'] = ta.sma(df['Close'], length=60)
    # 震盪
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    df = pd.concat([df, macd], axis=1) # MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    # 通道
    bb = ta.bbands(df['Close'], length=20, std=2)
    df = pd.concat([df, bb], axis=1) # BBU_20_2.0, BBL_20_2.0
    # 能量
    df['OBV'] = ta.obv(df['Close'], df['Volume'])

    # D. 計算 Buy & Hold 曲線 (基準)
    # 假設第一天用所有現金買入
    first_price = df['Close'].iloc[0]
    bh_shares = init_cash / first_price
    df['BH_Equity'] = bh_shares * df['Close']

    # E. 計算 God Mode (上帝視角) - 簡化版極值策略
    # 邏輯：只要是 5 日低點就全買，5 日高點就全賣
    df['God_Equity'] = init_cash # 初始化
    cash = init_cash
    pos = 0
    # 使用滾動窗口找極值
    df['Min_5'] = df['Close'].rolling(window=5, center=True).min()
    df['Max_5'] = df['Close'].rolling(window=5, center=True).max()
    
    god_curve = []
    for idx, row in df.iterrows():
        price = row['Close']
        # 簡單上帝模擬
        if row['Close'] == row['Min_5'] and cash > 0: # 低點買
            pos = cash / price
            cash = 0
        elif row['Close'] == row['Max_5'] and pos > 0: # 高點賣
            cash = pos * price
            pos = 0
        
        # 每日結算
        curr_val = cash + (pos * price)
        god_curve.append(curr_val)
    
    df['God_Equity'] = god_curve

    return df

# ==========================================
# 3. Session State 初始化 (遊戲存檔)
# ==========================================
def init_session():
    if 'game_active' not in st.session_state:
        st.session_state.game_active = False
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = 50 # 從第50天開始
    if 'cash' not in st.session_state:
        st.session_state.cash = 100000.0
    if 'holdings' not in st.session_state:
        st.session_state.holdings = 0
    if 'avg_cost' not in st.session_state:
        st.session_state.avg_cost = 0.0
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    if 'user_equity_curve' not in st.session_state:
        # 紀錄每一天的資產淨值 (時間, 金額)
        st.session_state.user_equity_curve = []

init_session()

# ==========================================
# 4. 側邊欄：遊戲設定
# ==========================================
with st.sidebar:
    st.title("🎮 盤感訓練設定")
    symbol = st.text_input("股票代碼", "NVDA")
    col1, col2 = st.columns(2)
    start_d = col1.date_input("開始", datetime.date(2023, 1, 1))
    end_d = col2.date_input("結束", datetime.date(2024, 1, 1))
    init_cash = st.number_input("初始資金", 100000.0)
    
    st.divider()
    st.subheader("📊 指標顯示設定")
    show_sma = st.checkbox("SMA (均線)", True)
    show_bb = st.checkbox("BBands (布林)", True)
    show_rsi = st.checkbox("RSI (副圖)", True)
    show_macd = st.checkbox("MACD (副圖)", False)
    show_macro = st.checkbox("宏觀 (VIX/美債)", False)

    st.divider()
    if st.button("🚀 開始新遊戲", type="primary"):
        with st.spinner("數據下載與策略計算中..."):
            # 重置變數
            st.session_state.current_idx = 50
            st.session_state.cash = init_cash
            st.session_state.holdings = 0
            st.session_state.avg_cost = 0.0
            st.session_state.trade_log = []
            st.session_state.user_equity_curve = []
            
            # 獲取數據
            df = prepare_game_data(symbol, start_d, end_d, init_cash)
            st.session_state.game_data = df
            st.session_state.game_active = True
            
            # 初始化前面的淨值曲線 (前50天假裝都是現金)
            for i in range(50):
                date_str = df.index[i].strftime('%Y-%m-%d')
                st.session_state.user_equity_curve.append({"time": date_str, "value": init_cash})
            
            st.rerun()

# ==========================================
# 5. 遊戲主畫面
# ==========================================
if st.session_state.game_active:
    df = st.session_state.game_data
    
    # 防呆：避免 index 超出範圍
    if st.session_state.current_idx >= len(df):
        st.session_state.current_idx = len(df) - 1
        game_over = True
    else:
        game_over = False

    # 切片數據 (Slice)：只拿到「今天之前」的數據
    curr_idx = st.session_state.current_idx
    slice_df = df.iloc[:curr_idx+1]
    today_row = slice_df.iloc[-1]
    today_date = slice_df.index[-1].strftime('%Y-%m-%d')
    current_price = float(today_row['Close'])

    # --- A. 帳戶狀態更新 (HUD) ---
    market_val = st.session_state.holdings * current_price
    total_assets = st.session_state.cash + market_val
    
    # 紀錄今天的淨值
    # 如果今天還沒紀錄過 (避免 refresh 重複寫入)
    if len(st.session_state.user_equity_curve) <= curr_idx:
        st.session_state.user_equity_curve.append({"time": today_date, "value": total_assets})
    
    # 績效計算
    roi = (total_assets - init_cash) / init_cash * 100
    bh_val = today_row['BH_Equity']
    bh_roi = (bh_val - init_cash) / init_cash * 100
    god_val = today_row['God_Equity']
    god_roi = (god_val - init_cash) / init_cash * 100

    st.markdown(f"### 📅 {today_date} | 股價: ${current_price:.2f}")
    
    # HUD 儀表板
    col_hud1, col_hud2, col_hud3, col_hud4 = st.columns(4)
    col_hud1.metric("💰 我的資產", f"${total_assets:,.0f}", f"{roi:.2f}%")
    col_hud2.metric("😴 Buy & Hold", f"${bh_val:,.0f}", f"{bh_roi:.2f}%")
    col_hud3.metric("😇 上帝視角", f"${god_val:,.0f}", f"{god_roi:.2f}%")
    
    unrealized = (current_price - st.session_state.avg_cost) * st.session_state.holdings if st.session_state.holdings > 0 else 0
    col_hud4.metric("📈 持倉損益", f"${unrealized:,.0f}", f"持股: {st.session_state.holdings}")

    # --- B. 繪圖引擎 (Canvas) ---
    # 1. 準備主圖數據
    kline_data = [{"time": i.strftime('%Y-%m-%d'), "open": r['Open'], "high": r['High'], "low": r['Low'], "close": r['Close']} for i, r in slice_df.iterrows()]
    
    series_main = [{
        "type": 'Candlestick',
        "data": kline_data,
        "options": {"upColor": '#26a69a', "downColor": '#ef5350', "borderVisible": False}
    }]

    # 疊加指標
    if show_sma:
        sma20 = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in slice_df['SMA_20'].items() if not pd.isna(v)]
        series_main.append({"type": "Line", "data": sma20, "options": {"color": "yellow", "lineWidth": 2, "title": "SMA20"}})
    
    if show_bb:
        # 模糊匹配 BB 欄位
        bbu_col = [c for c in df.columns if 'BBU' in c][0]
        bbl_col = [c for c in df.columns if 'BBL' in c][0]
        bbu_data = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in slice_df[bbu_col].items() if not pd.isna(v)]
        bbl_data = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in slice_df[bbl_col].items() if not pd.isna(v)]
        series_main.append({"type": "Line", "data": bbu_data, "options": {"color": "rgba(0,100,255,0.3)"}})
        series_main.append({"type": "Line", "data": bbl_data, "options": {"color": "rgba(0,100,255,0.3)"}})

    # 交易標記 (Markers)
    markers = []
    for t in st.session_state.trade_log:
        # 只顯示目前時間點之前的交易
        if pd.to_datetime(t['Date']) <= slice_df.index[-1]:
            markers.append({
                "time": t['Date'],
                "position": "belowBar" if t['Type'] == 'Buy' else "aboveBar",
                "color": "green" if t['Type'] == 'Buy' else "red",
                "shape": "arrowUp" if t['Type'] == 'Buy' else "arrowDown",
                "text": f"{t['Type']} {t['Pct']}%"
            })
    series_main[0]["markers"] = markers

    # 副圖
    series_sub = []
    
    # RSI
    if show_rsi:
        rsi_data = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in slice_df['RSI'].items() if not pd.isna(v)]
        series_sub.append({"chart": {"height": 150}, "series": [{"type": "Line", "data": rsi_data, "options": {"color": "purple", "title": "RSI"}}]})

    # MACD
    if show_macd:
        # 模糊匹配 MACD
        hist_col = [c for c in df.columns if 'MACDh' in c][0]
        macd_col = [c for c in df.columns if 'MACD_' in c and 'h' not in c and 's' not in c][0]
        
        hist_data = [{"time": i.strftime('%Y-%m-%d'), "value": float(v), "color": "green" if v>0 else "red"} for i, v in slice_df[hist_col].items() if not pd.isna(v)]
        macd_data = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in slice_df[macd_col].items() if not pd.isna(v)]
        
        series_sub.append({"chart": {"height": 150}, "series": [
            {"type": "Histogram", "data": hist_data},
            {"type": "Line", "data": macd_data, "options": {"color": "blue"}}
        ]})

    # 宏觀 (VIX/TNX)
    if show_macro:
        vix_data = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in slice_df['VIX'].items()]
        tnx_data = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in slice_df['TNX'].items()]
        series_sub.append({"chart": {"height": 150}, "series": [
            {"type": "Line", "data": vix_data, "options": {"color": "red", "title": "VIX"}},
            {"type": "Line", "data": tnx_data, "options": {"color": "orange", "title": "10y Yield"}}
        ]})

    # 資產曲線比較圖 (Canvas 繪製)
    # 將 Session 中的 user_equity_curve 轉換格式
    # 為了能即時看到勝負，我們把三條線畫在一起
    
    # 取出 B&H 和 God 的曲線數據
    bh_curve_data = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in slice_df['BH_Equity'].items()]
    god_curve_data = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in slice_df['God_Equity'].items()]
    user_curve_data = st.session_state.user_equity_curve # 已經是正確格式
    
    equity_chart_config = {
        "chart": {"height": 200, "title": "資金曲線對決"},
        "series": [
            {"type": "Line", "data": user_curve_data, "options": {"color": "blue", "lineWidth": 3, "title": "我 (User)"}},
            {"type": "Line", "data": bh_curve_data, "options": {"color": "gray", "lineStyle": 2, "title": "Buy&Hold"}},
            {"type": "Line", "data": god_curve_data, "options": {"color": "gold", "lineStyle": 1, "title": "God Mode"}}
        ]
    }

    # 組合所有圖表
    all_charts = [{"chart": {"height": 400}, "series": series_main}] + series_sub + [equity_chart_config]
    
    # 渲染
    renderLightweightCharts(all_charts, key=f"game_chart_{curr_idx}")

    # --- C. 操作控制區 ---
    st.divider()
    
    if not game_over:
        c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([1.5, 1.5, 2])
        
        with c_ctrl1:
            st.write("🟢 **買入**")
            buy_pct = st.slider("買入資金 %", 10, 100, 50, key="buy_pct")
            amt = st.session_state.cash * (buy_pct/100)
            can_buy_shares = int(amt / current_price)
            if st.button(f"買進 {can_buy_shares} 股", use_container_width=True):
                if can_buy_shares > 0:
                    cost = can_buy_shares * current_price
                    st.session_state.cash -= cost
                    st.session_state.holdings += can_buy_shares
                    # 更新均價
                    total_cost = st.session_state.avg_cost * (st.session_state.holdings - can_buy_shares) + cost
                    st.session_state.avg_cost = total_cost / st.session_state.holdings
                    # 紀錄
                    st.session_state.trade_log.append({"Date": today_date, "Type": "Buy", "Pct": buy_pct, "Price": current_price})
                    st.rerun()

        with c_ctrl2:
            st.write("🔴 **賣出**")
            sell_pct = st.slider("賣出持倉 %", 10, 100, 50, key="sell_pct")
            shares_to_sell = int(st.session_state.holdings * (sell_pct/100))
            if st.button(f"賣出 {shares_to_sell} 股", use_container_width=True):
                if shares_to_sell > 0:
                    rev = shares_to_sell * current_price
                    st.session_state.cash += rev
                    st.session_state.holdings -= shares_to_sell
                    if st.session_state.holdings == 0: st.session_state.avg_cost = 0
                    st.session_state.trade_log.append({"Date": today_date, "Type": "Sell", "Pct": sell_pct, "Price": current_price})
                    st.rerun()

        with c_ctrl3:
            st.write("⏩ **推進**")
            c_n1, c_n2 = st.columns(2)
            if c_n1.button("下一天", type="primary", use_container_width=True):
                st.session_state.current_idx += 1
                st.rerun()
            if c_n2.button("快轉 10天", use_container_width=True):
                st.session_state.current_idx += 10
                st.rerun()
    else:
        st.balloons()
        st.error("🏁 遊戲結束！請看上方的最終績效對比。")
        st.dataframe(pd.DataFrame(st.session_state.trade_log))

else:
    st.info("👈 請在左側設定後，點擊「開始新遊戲」")
