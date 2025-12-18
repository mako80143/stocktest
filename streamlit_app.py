import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import datetime
from streamlit_lightweight_charts import renderLightweightCharts

# 嘗試匯入 scipy，若失敗則給出友善提示 (防止崩潰)
try:
    from scipy.signal import argrelextrema
except ImportError:
    st.error("❌ 缺少關鍵套件 'scipy'。請在 requirements.txt 中加入 'scipy' 並重新安裝。")
    st.stop()

# --- 1. 頁面設定 ---
st.set_page_config(page_title="操盤手訓練營 v10.0", layout="wide")
st.markdown("""
<style>
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
    div.stButton > button:first-child {font-weight: bold; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心邏輯：上帝視角演算法
# ==========================================
def calculate_god_mode(df, initial_cash, buy_pct, sell_pct):
    """
    計算上帝視角的理論最大獲利
    buy_pct: 低點買入資金百分比 (0.1 ~ 1.0)
    sell_pct: 高點賣出持倉百分比 (0.1 ~ 1.0)
    """
    # 尋找局部極值 (前後 5 天比較)
    n = 5
    df['is_min'] = df.iloc[argrelextrema(df.Close.values, np.less_equal, order=n)[0]]['Close']
    df['is_max'] = df.iloc[argrelextrema(df.Close.values, np.greater_equal, order=n)[0]]['Close']
    
    cash = initial_cash
    position = 0
    
    # 模擬回測
    for idx, row in df.iterrows():
        price = row['Close']
        
        # 遇到波段低點 -> 買進
        if not np.isnan(row['is_min']) and cash > 1000: # 餘額太少就不買
            spend = cash * buy_pct
            size = int(spend / price)
            if size > 0:
                cash -= size * price
                position += size
                
        # 遇到波段高點 -> 賣出
        elif not np.isnan(row['is_max']) and position > 0:
            sell_size = int(position * sell_pct)
            if sell_size > 0:
                cash += sell_size * price
                position -= sell_size
                
    final_equity = cash + (position * df.iloc[-1]['Close'])
    return final_equity

# ==========================================
# 3. 指標設定檔
# ==========================================
INDICATORS_CONFIG = {
    "SMA (簡單均線)": {"func": "sma", "args": [20], "color": "yellow", "overlay": True, "desc": "基礎趨勢"},
    "EMA (指數均線)": {"func": "ema", "args": [20], "color": "orange", "overlay": True, "desc": "加權趨勢"},
    "BBands (布林通道)": {"func": "bbands", "args": [20, 2], "overlay": True, "desc": "波動範圍"},
    "RSI (相對強弱)": {"func": "rsi", "args": [14], "color": "purple", "overlay": False, "desc": "超買超賣"},
    "MACD (動能)": {"func": "macd", "args": [], "overlay": False, "desc": "波段操作"},
    "KD (隨機指標)": {"func": "stoch", "args": [], "overlay": False, "desc": "短線轉折"},
    "ADX (趨勢強度)": {"func": "adx", "args": [14], "color": "red", "overlay": False, "desc": "趨勢力度"},
    "ATR (真實波幅)": {"func": "atr", "args": [14], "color": "brown", "overlay": False, "desc": "波動率"},
    "OBV (能量潮)": {"func": "obv", "args": [], "overlay": False, "desc": "籌碼分析"}
}

# ==========================================
# 4. Session State 初始化
# ==========================================
if 'idx' not in st.session_state: st.session_state.idx = 50 
if 'cash' not in st.session_state: st.session_state.cash = 100000.0
if 'position' not in st.session_state: st.session_state.position = 0
if 'avg_cost' not in st.session_state: st.session_state.avg_cost = 0.0
if 'realized_pnl' not in st.session_state: st.session_state.realized_pnl = 0.0
if 'trade_log' not in st.session_state: st.session_state.trade_log = []
if 'data' not in st.session_state: st.session_state.data = None
# 紀錄上帝與基準分數
if 'bh_score' not in st.session_state: st.session_state.bh_score = 0.0
if 'god_score' not in st.session_state: st.session_state.god_score = 0.0

# ==========================================
# 5. 側邊欄控制台
# ==========================================
with st.sidebar:
    st.header("⚙️ 遊戲參數設定")
    symbol = st.text_input("股票代碼", "NVDA")
    
    col1, col2 = st.columns(2)
    start_date = col1.date_input("開始", datetime.date(2023, 1, 1))
    end_date = col2.date_input("結束", datetime.date(2024, 1, 1))
    
    init_cash = st.number_input("初始資金 (固定)", value=100000.0, disabled=True)
    
    st.divider()
    st.subheader("😇 上帝視角參數")
    god_buy_pct = st.slider("上帝低點買入資金 %", 10, 100, 50, key="god_buy") / 100.0
    god_sell_pct = st.slider("上帝高點賣出持倉 %", 10, 100, 30, key="god_sell") / 100.0
    
    st.divider()
    st.subheader("🛠️ 技術指標")
    selected_ind_names = st.multiselect(
        "選擇指標", list(INDICATORS_CONFIG.keys()), default=["SMA (簡單均線)", "RSI (相對強弱)"]
    )
    
    st.divider()
    if st.button("🔄 重置並開始新局", type="primary"):
        st.session_state.idx = 50
        st.session_state.cash = 100000.0
        st.session_state.position = 0
        st.session_state.avg_cost = 0.0
        st.session_state.realized_pnl = 0.0
        st.session_state.trade_log = []
        st.session_state.data = None # 清空數據強制重抓
        st.rerun()

# ==========================================
# 6. 數據處理 (Data Processing)
# ==========================================
if st.session_state.data is None:
    with st.spinner("正在下載數據、計算指標與上帝劇本..."):
        # 下載數據
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # 修正 yfinance 可能導致索引未對齊問題
        df.index = pd.to_datetime(df.index)

        # 計算指標
        for name in INDICATORS_CONFIG:
            cfg = INDICATORS_CONFIG[name]
            try:
                # 呼叫 pandas_ta
                method = getattr(ta, cfg['func'])
                args = cfg['args']
                if cfg['func'] == 'stoch': # KD 特殊處理
                    res = method(df['High'], df['Low'], df['Close'])
                elif cfg['func'] == 'adx':
                    res = method(df['High'], df['Low'], df['Close'], length=args[0])
                else:
                    # 一般指標
                    res = method(df['Close'], length=args[0] if args else None)
                
                # 合併
                if isinstance(res, pd.DataFrame):
                    df = pd.concat([df, res], axis=1)
                else:
                    df[name] = res
            except Exception as e:
                pass # 忽略計算錯誤

        st.session_state.data = df
        
        # 預先計算 Benchmark (比較基準)
        # 1. Buy & Hold
        first_p = df['Close'].iloc[0]
        last_p = df['Close'].iloc[-1]
        bh_shares = int(100000.0 / first_p)
        st.session_state.bh_score = (100000.0 - bh_shares*first_p) + (bh_shares * last_p)
        
        # 2. God Mode
        st.session_state.god_score = calculate_god_mode(df.copy(), 100000.0, god_buy_pct, god_sell_pct)

df = st.session_state.data
if df is None or df.empty:
    st.error("無數據，請檢查代碼或日期。")
    st.stop()

# 完賽畫面
if st.session_state.idx >= len(df):
    st.session_state.idx = len(df) - 1
    st.balloons()
    
    final_asset = st.session_state.cash + (st.session_state.position * df.iloc[-1]['Close'])
    user_roi = (final_asset - 100000)/1000
    god_roi = (st.session_state.god_score - 100000)/1000
    bh_roi = (st.session_state.bh_score - 100000)/1000
    
    st.success("🏆 回測結束！最終戰績表")
    col1, col2, col3 = st.columns(3)
    col1.metric("😈 你的績效", f"${final_asset:,.0f}", f"{user_roi:.2f}%")
    col2.metric("😇 上帝視角", f"${st.session_state.god_score:,.0f}", f"{god_roi:.2f}%")
    col3.metric("😴 傻瓜持有", f"${st.session_state.bh_score:,.0f}", f"{bh_roi:.2f}%")
    
# ==========================================
# 7. 儀表板與切片數據
# ==========================================
current_slice = df.iloc[:st.session_state.idx+1]
curr_row = current_slice.iloc[-1]
curr_price = float(curr_row['Close'])
curr_date = current_slice.index[-1].strftime('%Y-%m-%d')

# 資產計算
mkt_val = st.session_state.position * curr_price
total_asset = st.session_state.cash + mkt_val
unrealized_pnl = (curr_price - st.session_state.avg_cost) * st.session_state.position if st.session_state.position > 0 else 0

st.title(f"🕹️ {symbol} 自由操盤室 ({curr_date})")

# HUD
c1, c2, c3, c4 = st.columns(4)
c1.metric("💰 總資產", f"${total_asset:,.0f}", delta=f"{(total_asset-100000)/1000:.1f}%")
c2.metric("💵 現金餘額", f"${st.session_state.cash:,.0f}")
c3.metric("✅ 已實現損益", f"${st.session_state.realized_pnl:,.0f}", help="已落袋為安的錢")
c4.metric("📈 未實現損益", f"${unrealized_pnl:,.0f}", help="帳面浮動盈虧")

# ==========================================
# 8. 繪圖 (LWC)
# ==========================================
kline_data = []
for idx, row in current_slice.iterrows():
    kline_data.append({"time": idx.strftime('%Y-%m-%d'), "open": row['Open'], "high": row['High'], "low": row['Low'], "close": row['Close']})

series_main = [{
    "type": 'Candlestick',
    "data": kline_data,
    "options": {"upColor": '#26a69a', "downColor": '#ef5350', "borderVisible": False}
}]

# 處理指標
series_sub = []
for name in selected_ind_names:
    cfg = INDICATORS_CONFIG[name]
    # 模糊搜尋欄位
    col = None
    if name in df.columns: col = name
    else:
        candidates = [c for c in df.columns if c.startswith(cfg['func'].upper()) or c.startswith(cfg['func'].lower())]
        if candidates: col = candidates[0]
        if 'BBands' in name: # BBands 特殊處理
             bbu = [c for c in df.columns if 'BBU' in c]; bbl = [c for c in df.columns if 'BBL' in c]
             if bbu and bbl:
                 bbu_d = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in current_slice[bbu[0]].items() if pd.notnull(v)]
                 bbl_d = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in current_slice[bbl[0]].items() if pd.notnull(v)]
                 series_main.append({"type": "Line", "data": bbu_d, "options": {"color": "rgba(0,100,255,0.3)"}})
                 series_main.append({"type": "Line", "data": bbl_d, "options": {"color": "rgba(0,100,255,0.3)"}})
                 col = None # 已處理

    if col:
        data = [{"time": i.strftime('%Y-%m-%d'), "value": float(v)} for i, v in current_slice[col].items() if pd.notnull(v)]
        if cfg['overlay']:
            series_main.append({"type": "Line", "data": data, "options": {"color": cfg['color'], "lineWidth": 2}})
        else:
            series_sub.append({"chart": {"height": 150}, "series": [{"type": "Line", "data": data, "options": {"color": cfg.get('color', 'blue')}}]})

# 買賣標記
markers = []
for t in st.session_state.trade_log:
    markers.append({
        "time": t['Date'], "position": "belowBar" if t['Type']=='Buy' else "aboveBar",
        "color": "green" if t['Type']=='Buy' else "red", "shape": "arrowUp" if t['Type']=='Buy' else "arrowDown",
        "text": f"{t['Type']} ({t['Pct']}%)"
    })
series_main[0]["markers"] = markers

charts = [{"chart": {"height": 450}, "series": series_main}] + series_sub
renderLightweightCharts(charts, key=f"v10_chart_{st.session_state.idx}")

# ==========================================
# 9. 操盤控制區 (靈活資金版)
# ==========================================
st.divider()

# 使用 columns 佈局控制區
c_buy, c_sell, c_nav = st.columns([1.5, 1.5, 2])

with c_buy:
    st.markdown("#### 🟢 買入操作")
    # 滑桿決定買入資金比例
    buy_pct_manual = st.slider("投入現金比例 (%)", 10, 100, 50, key="buy_slider", help="選擇要用目前現金的多少比例來買入")
    buy_amt = st.session_state.cash * (buy_pct_manual / 100)
    buy_shares_est = int(buy_amt / curr_price) if curr_price > 0 else 0
    
    if st.button(f"買進 {buy_shares_est} 股 (約 ${buy_amt:.0f})", use_container_width=True):
        if buy_shares_est > 0:
            cost = buy_shares_est * curr_price
            st.session_state.cash -= cost
            # 更新均價
            prev_val = st.session_state.avg_cost * st.session_state.position
            st.session_state.position += buy_shares_est
            st.session_state.avg_cost = (prev_val + cost) / st.session_state.position
            # 紀錄
            st.session_state.trade_log.append({
                "Date": curr_date, "Type": "Buy", "Price": curr_price, "Size": buy_shares_est, "Pct": buy_pct_manual
            })
            st.success("買入成功！")
            st.rerun()
        else:
            st.error("現金不足或金額過小")

with c_sell:
    st.markdown("#### 🔴 賣出操作")
    # 滑桿決定賣出持倉比例
    sell_pct_manual
