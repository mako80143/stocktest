import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
from streamlit_gsheets import GSheetsConnection

# --- 1. 頁面與緩存設定 ---
st.set_page_config(page_title="AI 智能股票分析儀表板", layout="wide")
st.title("📈 AI 智能股票分析 & 資產配置")
st.markdown("結合 **Google Gemini** 與 **個人化資產數據** 的全方位投資助手")

# 使用緩存機制，避免重複呼叫 API (節省額度並加速)
@st.cache_data(ttl=24*3600) # 緩存 24 小時
def get_stock_data(ticker):
    """抓取股價與計算技術指標"""
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo")
    if df.empty:
        return None, None
    
    # 計算 RSI 和 SMA
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean() # 新增 60 MA
    
    return df, stock.info

@st.cache_data(ttl=300) # 匯率 5 分鐘緩存
def get_fx_rate():
    """獲取台幣兌美金即時匯率"""
    # USDTWD=X 是 Yahoo Finance 的美金兌台幣代碼
    try:
        usd_twd = yf.Ticker("USDTWD=X").info['regularMarketPrice']
        return usd_twd
    except:
        return 30.0 # 獲取失敗時使用預設值

@st.cache_data(ttl=24*3600)
def ai_analysis(api_key, ticker, info, current_price, rsi_val, sma_status):
    """呼叫 Gemini 進行分析"""
    genai.configure(api_key=api_key)
    # 使用 gemini-2.5-flash 確保速度和成本效益
    model = genai.GenerativeModel('gemini-2.5-flash') 
    
    # 這裡的 Prompt 請使用您在對話中希望的內容
    prompt = f"""
    你是一位專業的華爾街頂級分析師。請針對股票代碼：{ticker} ({info.get('longName', 'Unknown')}) 進行分析。
    
    【市場數據參考】
    - 目前股價：{current_price:.2f}
    - RSI (14)：{rsi_val:.2f}
    - 均線狀態：{sma_status}
    - 產業領域：{info.get('industry', 'Unknown')}
    
    請以繁體中文，用專業且易讀的 Markdown 格式回答以下三點：
    1. **公司簡介與性質挖掘**：這家公司屬於成長股、價值股還是循環股？請簡述其護城河。
    2. **長短期操作判斷**：適合長期存股還是短期波段？請說明判斷理由。
    3. **買賣時機建議**：綜合目前 RSI 與均線趨勢，請給出具體的策略建議。
    """
    
    try:
        with st.spinner('🤖 Gemini 正在分析中，請稍候...'):
            response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        return f"❌ Gemini API 錯誤：{str(e)}。請確認您的 API Key 是否正確。"


# --- 2. 側邊欄設定與 AI Key 輸入 ---
st.sidebar.header("⚙️ 應用程式參數")

st.sidebar.subheader("🔑 Gemini API 設定")
api_key = st.sidebar.text_input("請輸入您的 Gemini API Key", type="password")
st.sidebar.caption("還沒有 Key? [點此免費申請](https://aistudio.google.com/app/apikey)")
st.sidebar.divider()

# --- 3. Google Sheets 資料庫連接 ---
st.sidebar.subheader("💾 個人資產資料庫")
try:
    # 嘗試連接 Google Sheets (需要 Streamlit Secrets 配置)
    conn = st.connection("gsheets", type=GSheetsConnection)
    portfolio_df = conn.read(worksheet="Portfolio", usecols=list(range(5)))
    portfolio_df = portfolio_df.dropna(subset=['Ticker']) # 移除空行
    
    tickers_list = portfolio_df['Ticker'].tolist()
    
    # 讓用戶從自己的持股清單中選擇股票進行分析
    selected_ticker = st.sidebar.selectbox(
        "選擇持股清單中的股票進行分析",
        options=[''] + tickers_list
    )
    
except Exception as e:
    st.sidebar.error("❌ Google Sheets 連線失敗，請檢查 Streamlit Secrets 設定。")
    portfolio_df = pd.DataFrame()
    selected_ticker = st.sidebar.text_input("或手動輸入股票代碼 (台股請加 .TW)", value="TSLA").upper()


# 手動輸入股票代碼 (如果 Sheets 連線失敗或不想用清單)
manual_ticker = st.sidebar.text_input("或手動輸入股票代碼 (台股請加 .TW)", value="TSLA").upper()
ticker_to_run = selected_ticker if selected_ticker else manual_ticker

run_btn = st.sidebar.button("🚀 開始分析")


# --- 4. 主程式邏輯 ---

if run_btn and ticker_to_run:
    if not api_key:
        st.error("⚠️ 請先在左側輸入 Gemini API Key！")
    else:
        try:
            df, info = get_stock_data(ticker_to_run)
            fx_rate = get_fx_rate()
            
            if df is None:
                st.error(f"❌ 找不到股票代碼 {ticker_to_run} 的數據。")
                st.stop()
            
            # 獲取最新數據
            current_price = df['Close'].iloc[-1]
            latest_rsi = df['RSI'].iloc[-1]
            sma_20 = df['SMA_20'].iloc[-1]
            sma_60 = df['SMA_60'].iloc[-1]
            
            sma_status = f"20MA: {current_price > sma_20} | 60MA: {current_price > sma_60}"

            # --- 區塊 A: 資產配置儀表板 (從 Sheets 讀取數據) ---
            st.header(f"💼 {info.get('shortName', ticker_to_run)} ({ticker_to_run}) 概況")
            st.caption(f"即時匯率 (USD/TWD): {fx_rate:.2f}")
            
            # 篩選個人持股資料
            my_holding = portfolio_df[portfolio_df['Ticker'] == ticker_to_run]
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("即時股價", f"${current_price:.2f}", f"{df['Close'].diff().iloc[-1]:.2f}")
            col2.metric("RSI (14日)", f"{latest_rsi:.2f}%")
            
            if not my_holding.empty:
                # 假設只取第一筆資料 (如果有多筆需加總)
                holding = my_holding.iloc[0]
                share_qty = holding['Quantity']
                avg_price = holding['AvgPrice']
                currency = holding['Currency']
                
                # 計算市值與損益
                current_market_value_usd = current_price * share_qty
                cost_usd = avg_price * share_qty
                profit_usd = current_market_value_usd - cost_usd
                profit_pct = (profit_usd / cost_usd) * 100 if cost_usd != 0 else 0
                
                # 匯率換算
                if currency == 'USD':
                    total_twd_profit = profit_usd * fx_rate
                    total_twd_market_value = current_market_value_usd * fx_rate
                    col3.metric("持股均價", f"USD ${avg_price:.2f}")
                    col4.metric("未實現損益 (TWD)", f"NT${total_twd_profit:,.0f}", f"{profit_pct:.2f}%")
                else: # 假設為 TWD
                    total_twd_profit = profit_usd # 假設 TWD 資產不需要匯率換算
                    col3.metric("持股均價", f"NT${avg_price:,.0f}")
                    col4.metric("未實現損益 (TWD)", f"NT${profit_usd:,.0f}", f"{profit_pct:.2f}%")

            else:
                col3.info("無持股數據")
                col4.info("無法計算損益")

            # --- 區塊 B: 互動式 K 線圖 (優化互動性) ---
            st.subheader("📈 技術線圖")
            fig = go.Figure(data=[
                go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'),
                go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='blue', width=1), name='20 MA'),
                go.Scatter(x=df.index, y=df['SMA_60'], line=dict(color='red', width=1), name='60 MA')
            ])
            
            fig.update_layout(
                height=500,
                xaxis_rangeslider_visible=True, # **啟用時間軸滑塊**
                xaxis=dict(type="category"), 
                hovermode="x unified",
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True}) # **啟用右上角工具欄**

            # --- 區塊 C: Gemini AI 深度分析 ---
            st.divider()
            st.subheader("🤖 Gemini 投資顧問分析報告")
            
            analysis_text = ai_analysis(api_key, ticker_to_run, info, current_price, latest_rsi, sma_status)
            st.markdown(analysis_text)
            
        except Exception as e:
            st.error(f"應用程式運行發生錯誤：{str(e)}")

else:
    st.info("👈 請在左側輸入 Gemini API Key 或設定資產資料庫，並點擊「開始分析」")
