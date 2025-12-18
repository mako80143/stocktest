import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy.signal import argrelextrema
import datetime
from streamlit_lightweight_charts import renderLightweightCharts

# --- 1. 頁面設定 ---
st.set_page_config(page_title="上帝視角訓練營 v9.0", layout="wide")
st.markdown("""
<style>
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
    .block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心邏輯函式庫
# ==========================================

def calculate_god_mode(df, initial_cash, buy_pct=0.5, sell_pct=0.3):
    """
    上帝視角演算法：
    1. 找出所有波段高低點
    2. 低點買入 50% 現金
    3. 高點賣出 30% 持倉
    4. 回傳最終淨值與交易次數
    """
    # 使用 Scipy 尋找局部極值 (n=5 代表前後5天都要比它高/低)
    n = 5 
    df['min'] = df.iloc[argrelextrema(df.Close.values, np.less_equal, order=n)[0]]['Close']
    df['max'] = df.iloc[argrelextrema(df.Close.values, np.greater_equal, order=n)[0]]['Close']
    
    cash = initial_cash
    position = 0
    trades = 0
    
    # 模擬交易
    for idx, row in df.iterrows():
        price = row['Close']
        
        # 遇到波段低點 -> 買入
        if not np.isnan(row['min']) and cash > 0:
            spend = cash * buy_pct
            size = int(spend / price)
            if size > 0:
                cash -= size * price
                position += size
                trades += 1
                
        # 遇到波段高點 -> 賣出
        elif not np.isnan(row['max']) and position > 0:
            sell_size = int(position * sell_pct)
            if sell_size > 0:
                cash += sell_size * price
                position -= sell_size
                trades += 1
                
    final_equity = cash + (position * df.iloc[-1]['Close'])
    return final_equity, trades

# 指標全家桶設定
INDICATORS_CONFIG = {
    "SMA (簡單均線)": {"func": "sma", "args": [20], "color": "yellow", "overlay": True, "desc": "趨勢判斷"},
    "EMA (指數均線)": {"func": "ema", "args": [20], "color": "orange", "overlay": True, "desc": "靈敏趨勢"},
    "BBands (布林通道)": {"func": "bbands", "args": [20, 2], "overlay": True, "desc": "波動範圍"},
    "RSI (相對強弱)": {"func": "rsi", "args": [14], "color": "purple", "overlay": False, "desc": "超買超賣"},
    "MACD (動能指標)": {"func": "macd", "args": [], "overlay": False, "desc": "波段神器"},
    "KD (隨機指標)": {"func": "stoch", "args": [], "overlay": False, "desc": "短線轉折"},
    "ADX (趨勢強度)": {"func": "adx", "args": [14], "color": "red", "overlay": False, "desc": "趨勢力度"},
    "ATR (真實波幅)": {"func": "atr", "args": [14], "color": "brown", "overlay": False, "desc": "波動率"},
    "CCI (順勢指標)": {"func": "cci", "args": [14], "color": "blue", "overlay": False, "desc": "抓突破"},
    "OBV (能量潮)": {"func": "obv", "args": [], "overlay": False, "desc": "成交量籌碼"}
}

# ==========================================
# 3. Session State 初始化
# ==========================================
if 'idx' not in st.session_state: st.session_state.idx = 50 
if 'cash' not in st.session_state: st.session_state.cash = 100000.0
if 'position' not in st.session_state: st.session_state.position = 0
if 'avg_cost' not in st.session_state: st.session_state.avg_cost = 0.0
if 'realized_pnl' not in st.session_state: st.session_state.realized_pnl = 0.0
if 'trade_log' not in st.session_state: st.session_state.trade_log = []
if 'god_score' not in st.session_state: st.session_state.god_score = 0.0
if 'bh_score' not in st.session_state: st.session_state.bh_score = 0.0
if 'data' not in st.session_state: st.session_state.data = None

# ==========================================
# 4. 側邊欄：設定與指標
# ==========================================
with st.sidebar:
    st.header("⚙️ 遊戲設定")
    symbol = st.text_input("股票代碼", "NVDA")
    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.date_input("開始", datetime.date(2023, 1, 1))
    end_date = col_d2.date_input("結束", datetime.date(2024, 1, 1))
    init_cash = st.number_input("初始資金", 100000.0)
    
    st.divider()
    st.subheader("🛠️ 指標自助餐")
    st.info("請從下方選單加入指標：")
    
    selected_ind_names = st.multiselect(
        "選擇要顯示的指標", 
        list(INDICATORS_CONFIG.keys()),
        default=["SMA (簡單均線)", "RSI (相對強弱)"]
    )
    
    # 顯示選中指標的說明
    for name in selected_ind_names:
        st.caption(f"**{name.split()[0]}**: {INDICATORS_CONFIG[name]['desc']}")

    st.divider()
    if st.button("🔄 重置/開始回測", type="primary"):
        st.session_state.idx = 50
        st.session_state.cash = init_cash
        st.session_state.position = 0
        st.session_state.avg_cost = 0.0
        st.session_state.realized_pnl = 0.0
        st.session_state.trade_log = []
        st.session_state.data = None # 強制重抓
        st.rerun()

# ==========================================
# 5. 數據處理與上帝視角計算
# ==========================================
if st.session_state.data is None:
    with st.spinner("正在下載數據並計算上帝劇本..."):
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # 計算指標
        for name in INDICATORS_CONFIG:
            cfg = INDICATORS_CONFIG[name]
            try:
                # 動態呼叫 pandas_ta 函式
                method = getattr(ta, cfg['func'])
                res = method(df['Close'] if 'High' not in df else df['High'], 
                             df['Low'] if 'High' in df else None, 
                             df['Close'] if 'High' in df else None, 
                             length=cfg['args'][0] if cfg['args'] else None)
                
                # 處理 ta 回傳可能是 DataFrame 或 Series 的情況
                if isinstance(res, pd.DataFrame):
                    df = pd.concat([df, res], axis=1)
                else:
                    df[name] = res
            except:
                # 簡單指標如 SMA/RSI 參數處理
                try:
                    res = getattr(ta, cfg['func'])(df['Close'], length=cfg['args'][0] if cfg['args'] else None)
                    df[name] = res
                except: pass

        st.session_state.data = df
        
        # --- 計算 Benchmark ---
        # 1. Buy & Hold (一開始全買)
        first_price = df['Close'].iloc[0]
        bh_shares = int(init_cash / first_price)
        bh_final = bh_shares * df['Close'].iloc[-1] + (init_cash - bh_shares * first_price)
        st.session_state.bh_score = bh_final

        # 2. God Mode (波段全吃)
        god_equity, _ = calculate_god_mode(df.copy(), init_cash, buy_pct=0.5, sell_pct=0.3)
        st.session_state.god_score = god_equity

df = st.session_state.data

# 完賽判斷
if st.session_state.idx >= len(df):
    st.session_state.idx = len(df) - 1
    st.balloons()
    
    # 結算畫面
    st.success("🏆 回測結束！最終成績單")
    
    # 計算目前總淨值
    final_equity = st.session_state.cash + (st.session_state.position * df.iloc[-1]['Close'])
    user_ret = (final_equity - init_cash) / init_cash * 100
    bh_ret = (st.session_state.bh_score - init_cash) / init_cash * 100
    god_ret = (st.session_state.god_score - init_cash) / init_cash * 100
    
    c1, c2, c3 = st.columns(3)
    c1.metric("😈 你的績效", f"${final_equity:,.0f}", f"{user_ret:.2f}%")
    c2.metric("😴 傻瓜持有 (Buy&Hold)", f"${st.session_state.bh_score:,.0f}", f"{bh_ret:.2f}%")
    c3.metric("😇 上帝視角 (God Mode)", f"${st.session_state.god_score:,.0f}", f"{god_ret:.2f}%")
    
    if user_ret > bh_ret:
        st.info("🔥 太強了！你戰勝了買入持有策略！")
    else:
        st.warning("📉 可惜，忙進忙出還不如第一天買了就去睡覺。")

# ==========================================
# 6. 戰情儀表板 (HUD)
# ==========================================
current_slice = df.iloc[:st.session_state.idx+1]
curr_row = current_slice.iloc[-1]
curr_price = float(curr_row['Close'])
curr_date = current_slice.index[-1].strftime('%Y-%m-%d')

# 計算帳務
mkt_value = st.session_state.position * curr_price
total_asset = st.session_state.cash + mkt_value
unrealized = (curr_price - st.session_state.avg_cost) * st.session_state.position if st.session_state.position > 0 else 0

st.title(f"🕹️ {symbol} 操盤模擬器 ({curr_date})")

# HUD
c1, c2, c3, c4 = st.columns(4)
c1.metric("💰 總資產", f"${total_asset:,.0f}", delta=f"{(total_asset-init_cash)/init_cash*100:.1f}%")
c2.metric("💵 現金水位", f"${st.session_state.cash:,.0f}")
c3.metric("✅ 已實現損益", f"${st.session_state.realized_pnl:,.0f}", help="已經賣出入袋的錢")
c4.metric("📈 未實現損益", f"${unrealized:,.0f}", help="帳面上浮動的盈虧")

# ==========================================
# 7. 繪圖引擎 (Lightweight Charts)
# ==========================================
# 主圖 K 線
kline_data = []
for idx, row in current_slice.iterrows():
    t_str = idx.strftime('%Y-%m-%d')
    kline_data.append({"time": t_str, "open": row['Open'], "high": row['High'], "low": row['Low'], "close": row['Close']})

series_main = [{
    "type": 'Candlestick',
    "data": kline_data,
    "options": {"upColor": '#26a69a', "downColor": '#ef5350', "borderVisible": False, "wickUpColor": '#26a69a', "wickDownColor": '#ef5350'}
}]

# 處理指標 (主圖疊加 vs 副圖)
series_sub = []
color_cycle = ['#2962FF', '#E91E63', '#FF9800', '#9C27B0']

for i, name in enumerate(selected_ind_names):
    cfg = INDICATORS_CONFIG[name]
    
    # 嘗試找對應的欄位名稱 (pandas_ta 產生的名稱可能不固定)
    # 這裡做簡單匹配
    found_col = None
    if name in df.columns: found_col = name
    else:
        # 模糊搜尋
        candidates = [c for c in df.columns if c.startswith(cfg['func'].upper()) or c.startswith(cfg['func'].lower())]
        if candidates: found_col = candidates[0]
        # BBands 特殊處理
        if 'BBands' in name:
            bbu = [c for c in df.columns if 'BBU' in c]
            bbl = [c for c in df.columns if 'BBL' in c]
            if bbu and bbl:
                bbu_data = [{"time": idx.strftime('%Y-%m-%d'), "value": float(row[bbu[0]])} for idx, row in current_slice.iterrows() if pd.notnull(row[bbu[0]])]
                bbl_data = [{"time": idx.strftime('%Y-%m-%d'), "value": float(row[bbl[0]])} for idx, row in current_slice.iterrows() if pd.notnull(row[bbl[0]])]
                series_main.append({"type": "Line", "data": bbu_data, "options": {"color": "rgba(0,100,255,0.3)", "lineWidth": 1}})
                series_main.append({"type": "Line", "data": bbl_data, "options": {"color": "rgba(0,100,255,0.3)", "lineWidth": 1}})
                continue # BBands 處理完跳過

    if found_col:
        line_data = [{"time": idx.strftime('%Y-%m-%d'), "value": float(row[found_col])} for idx, row in current_slice.iterrows() if pd.notnull(row[found_col])]
        
        if cfg['overlay']:
            # 疊加在主圖
            series_main.append({"type": "Line", "data": line_data, "options": {"color": cfg.get('color', 'blue'), "lineWidth": 2, "title": name}})
        else:
            # 獨立副圖
            series_sub.append({
                "chart": {"height": 150},
                "series": [{"type": "Line", "data": line_data, "options": {"color": cfg.get('color', color_cycle[i%4]), "title": name}}]
            })

# 買賣標記
markers = []
for t in st.session_state.trade_log:
    markers.append({
        "time": t['Date'],
        "position": "belowBar" if t['Type']=='Buy' else "aboveBar",
        "color": "green" if t['Type']=='Buy' else "red",
        "shape": "arrowUp" if t['Type']=='Buy' else "arrowDown",
        "text": f"{t['Type']} @ {t['Price']:.1f}"
    })
series_main[0]["markers"] = markers

# 組合圖表
charts_to_render = [{"chart": {"height": 450, "crosshair": {"mode": 0}}, "series": series_main}]
charts_to_render.extend(series_sub)

renderLightweightCharts(charts_to_render, key=f"god_replay_{st.session_state.idx}")

# ==========================================
# 8. 操作區 (固定資金比例)
# ==========================================
st.divider()
c_btn1, c_btn2, c_btn3, c_btn4 = st.columns([1, 1, 1, 2])

# 買入邏輯：投入當前現金的 50%
buy_amount = st.session_state.cash * 0.5
buy_shares = int(buy_amount / curr_price)

# 賣出邏輯：賣出當前持倉的 30%
sell_shares = int(st.session_state.position * 0.3)

if c_btn1.button(f"🟢 買進 (資金50%: {buy_shares}股)", use_container_width=True):
    if buy_shares > 0:
        cost = buy_shares * curr_price
        st.session_state.cash -= cost
        # 更新均價
        prev_val = st.session_state.avg_cost * st.session_state.position
        st.session_state.position += buy_shares
        st.session_state.avg_cost = (prev_val + cost) / st.session_state.position
        # 紀錄
        st.session_state.trade_log.append({"Date": curr_date, "Type": "Buy", "Price": curr_price, "Size": buy_shares, "Val": cost})
        st.rerun()
    else:
        st.error("現金不足以購買！")

if c_btn2.button(f"🔴 賣出 (持倉30%: {sell_shares}股)", use_container_width=True):
    if sell_shares > 0:
        revenue = sell_shares * curr_price
        # 計算這一筆的已實現損益
        cost_of_sold = sell_shares * st.session_state.avg_cost
        pnl = revenue - cost_of_sold
        st.session_state.realized_pnl += pnl
        
        st.session_state.cash += revenue
        st.session_state.position -= sell_shares
        if st.session_state.position == 0: st.session_state.avg_cost = 0
        # 紀錄
        st.session_state.trade_log.append({"Date": curr_date, "Type": "Sell", "Price": curr_price, "Size": sell_shares, "Val": revenue, "PnL": pnl})
        st.rerun()
    else:
        st.error("持倉不足！")

if c_btn3.button("⏭️ 快轉 10 天"):
    st.session_state.idx += 10
    st.rerun()

if c_btn4.button("⏩ 下一天", type="primary", use_container_width=True):
    st.session_state.idx += 1
    st.rerun()

# 交易紀錄表
if st.session_state.trade_log:
    with st.expander("📋 交易日記 (Trade Log)", expanded=True):
        st.dataframe(pd.DataFrame(st.session_state.trade_log))
