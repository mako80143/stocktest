import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai

# --- 1. 頁面設定 ---
st.set_page_config(page_title="AI 智能股票分析儀表板", layout="wide")

st.title("📈 AI 智能股票分析 & 資產管理")
st.markdown("結合 **Google Gemini** 與 **即時數據** 的全方位投資助手")

# --- 2. 側邊欄：輸入參數 ---
st.sidebar.header("⚙️ 設定參數")

# 安全性設計：API Key 輸入框 (避免 Key 寫死在程式碼中被盜用)
api_key = st.sidebar.text_input("請輸入 Google Gemini API Key", type="password")
st.sidebar.caption("還沒有 Key? [點此免費申請](https://aistudio.google.com/app/apikey)")

ticker = st.sidebar.text_input("輸入股票代碼 (台股請加 .TW)", value="2330.TW").upper()
my_avg_price = st.sidebar.number_input("你的買入均價 (若無持股可填 0)", value=0.0)
share_qty = st.sidebar.number_input("持股股數", value=1000, step=100)

run_btn = st.sidebar.button("🚀 開始分析")

# --- 3. 核心功能函數 ---

def get_stock_data(ticker):
    """抓取股價與計算技術指標"""
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo") # 抓半年數據
    
    if df.empty:
        return None, None
    
    # 計算簡單指標 (給 AI 參考用)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    
    # 計算 RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    info = stock.info
    return df, info

def ai_analysis(api_key, ticker, info, current_price, rsi_val, sma_status):
    """呼叫 Gemini 進行分析"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    你是一位專業的華爾街頂級分析師。請針對股票代碼：{ticker} ({info.get('longName', 'Unknown')}) 進行分析。
    
    【市場數據】
    - 目前股價：{current_price}
    - RSI (14)：{rsi_val:.2f}
    - 均線狀態：{sma_status}
    - 產業領域：{info.get('industry', 'Unknown')}
    
    請以繁體中文，用專業且易讀的 Markdown 格式回答以下三點：
    1. **公司簡介與性質挖掘**：這家公司是做什麼的？屬於成長股、價值股還是循環股？有什麼護城河？
    2. **長短期操作判斷**：適合長期存股還是短期波段？為什麼？
    3. **買賣時機建議**：綜合目前 RSI 與均線技術指標，現在是買點還是賣點？請給出具體的策略建議。
    
    (請保持語氣客觀冷靜，並強調風險)
    """
    
    with st.spinner('🤖 Gemini 正在閱讀財報與分析線圖中...'):
        response = model.generate_content(prompt)
        return response.text

# --- 4. 主程式邏輯 ---

if run_btn:
    if not api_key:
        st.error("⚠️ 請先在左側輸入 Gemini API Key 才能啟動 AI 大腦！")
    else:
        try:
            df, info = get_stock_data(ticker)
            
            if df is None:
                st.error("❌ 找不到股票代碼，請確認是否輸入正確 (台股需加 .TW)")
            else:
                # 取得最新數據
                current_price = df['Close'].iloc[-1]
                latest_rsi = df['RSI'].iloc[-1]
                sma_20 = df['SMA_20'].iloc[-1]
                
                sma_status = "股價在月線(20MA)之上 (強勢)" if current_price > sma_20 else "股價在月線(20MA)之下 (弱勢)"

                # --- 區塊 A: 資產配置儀表板 ---
                st.subheader(f"📊 {info.get('shortName', ticker)} 資產概況")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("目前股價", f"{current_price:.2f}", f"{df['Close'].diff().iloc[-1]:.2f}")
                
                if my_avg_price > 0:
                    market_value = current_price * share_qty
                    cost = my_avg_price * share_qty
                    profit = market_value - cost
                    profit_pct = (profit / cost) * 100
                    
                    col2.metric("持倉市值", f"${market_value:,.0f}")
                    col3.metric("未實現損益", f"${profit:,.0f}", f"{profit_pct:.2f}%")
                else:
                    col2.info("尚未輸入買入均價")
                    col3.info("無法計算損益")

                # --- 區塊 B: 互動式 K 線圖 ---
                st.subheader("📈 技術線圖")
                fig = go.Figure(data=[go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='K線')])
                fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)

                # --- 區塊 C: Gemini AI 深度分析 ---
                st.divider()
                st.subheader("🤖 Gemini 投資顧問分析報告")
                
                analysis_text = ai_analysis(api_key, ticker, info, current_price, latest_rsi, sma_status)
                st.markdown(analysis_text)
                
        except Exception as e:
            st.error(f"發生錯誤：{str(e)}")

else:
    st.info("👈 請在左側輸入股票代碼並按下「開始分析」")