import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
from streamlit_gsheets import GSheetsConnection

# --- 1. 頁面與緩存設定 ---
st.set_page_config(page_title="AI 智能股票分析儀表板", layout="wide")
st.title("📈 AI 智能股票分析 & 個人化資產儀表板")
st.markdown("結合 **Google Gemini** 與 **個人化資產數據** 的全方位投資助手")

# 使用緩存機制，避免重複呼叫 API (節省額度並加速)
@st.cache_data(ttl=24*3600) 
def get_stock_data(ticker):
    """抓取股價、計算技術指標與獲取大盤指數"""
    
    # 決定大盤指數代碼
    if ticker.endswith('.TW'):
        benchmark_ticker = '^TWII' # 台灣加權指數
    elif ticker.endswith('.HK'):
        benchmark_ticker = '^HSI' # 香港恆生指數
    else:
        benchmark_ticker = '^GSPC' # S&P 500 (美股預設)
        
    # 批量下載數據
    tickers_to_fetch = [ticker, benchmark_ticker]
    data = yf.download(tickers_to_fetch, period="6mo")
    
    if (ticker, 'Close') not in data.columns:
        return None, None, None, benchmark_ticker 
        
    df = data.loc[:, (slice(None), ticker)].droplevel(1, axis=1)
    
    if df.empty:
        return None, None, None, benchmark_ticker
    
    # 計算 RSI 和 SMA
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    
    benchmark_df = data.loc[:, (slice(None), benchmark_ticker)].droplevel(1, axis=1)
    
    stock_info = yf.Ticker(ticker).info
    
    return df, stock_info, benchmark_df, benchmark_ticker

@st.cache_data(ttl=300) # 匯率 5 分鐘緩存
def get_fx_rate():
    """獲取台幣兌美金即時匯率"""
    try:
        usd_twd = yf.Ticker("USDTWD=X").info['regularMarketPrice']
        return usd_twd
    except:
        return 32.0 

@st.cache_data(ttl=24*3600)
def ai_analysis(api_key, ticker, info, current_price, rsi_val, sma_status):
    """呼叫 Gemini 進行分析"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash') 
    
    prompt = f"""
    你是一位專業的華爾街頂級分析師。請針對股票代碼：{ticker} ({info.get('longName', '未知公司')}) 進行分析。
    
    【市場數據參考】
    - 最新股價：{current_price:.2f}
    - RSI (14日強度)：{rsi_val:.2f}
    - 均線狀態：{sma_status}
    - 產業領域：{info.get('industry', '未知產業')}
    
    請以專業、嚴謹且易讀的 **繁體中文 Markdown 格式** 回答以下三點：
    1. **公司本質判斷**：這家公司屬於成長股、價值股還是循環股？請簡述其商業護城河。
    2. **長短期操作策略**：根據公司性質和技術指標，此股票適合長期存股還是短期波段操作？請說明判斷依據。
    3. **買賣時機建議**：綜合目前的 RSI 與雙均線（20MA/60MA）趨勢，請給出具體的策略建議 (例如: 繼續持有/觀察壓力位/尋找低點介入)。
    """
    
    try:
        with st.spinner('🤖 Gemini 正在分析中，請稍候...'):
            response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        return f"❌ Gemini API 錯誤：{str(e)}。請確認您的 API Key 是否正確或檢查網路連線。"


# --- 2. 側邊欄設定與 AI Key 輸入 ---
st.sidebar.header("⚙️ 應用程式參數設定")

st.sidebar.subheader("🔑 Google Gemini API Key")
api_key = st.sidebar.text_input("請輸入您的 Gemini API Key", type="password")
st.sidebar.caption("還沒有 Key? [點此免費申請](https://aistudio.google.com/app/apikey)")
st.sidebar.divider()

# --- 3. Google Sheets 資料庫連接 ---
st.sidebar.subheader("💾 個人資產資料庫 (Google Sheets)")
st.sidebar.markdown("**Sheets 欄位名稱必須為英文：** `Ticker`, `Quantity`, `AvgPrice`, `Currency`。")

try:
    conn = st.connection("gsheets", type=GSheetsConnection)
    portfolio_df = conn.read(worksheet="Portfolio", usecols=list(range(4))) 
    portfolio_df = portfolio_df.dropna(subset=['Ticker']) 
    
    tickers_list = portfolio_df['Ticker'].tolist()
    
    selected_ticker = st.sidebar.selectbox(
        "選擇您的持股 (來自 Google Sheets)",
        options=[''] + tickers_list
    )
    
except Exception as e:
    st.sidebar.error("❌ Google Sheets 連線失敗。")
    portfolio_df = pd.DataFrame()
    selected_ticker = ''
    
    # === 設定輔助區塊 (連線失敗時顯示) ===
    st.sidebar.divider()
    with st.sidebar.expander("❓ Sheets 連線失敗？點此查看設定步驟"):
        st.markdown("---")
        st.markdown("#### **連線步驟 (請確認您已完成)**")
        st.markdown("""
        1. **建立 Sheets 檔案**：檔案名稱不拘，工作表命名為 `Portfolio`。
        2. **獲取金鑰**：需在 Google Cloud Console 建立服務帳號並下載 **JSON 金鑰**。
        3. **給予權限**：將服務帳號 Email 加入 Sheets 檔案的**編輯者**權限。
        """)

        st.markdown("#### **Streamlit Secrets 貼上格式 (一鍵複製)**")
        st.code("""
[connections.gsheets]
# 您的 Sheets 網址中 /d/ 後面的長代碼
spreadsheet = "請貼上您的 Google Sheet ID" 

# 這是您下載的服務帳號 JSON 金鑰內容
service_account_info = {
    "type": "service_account",
    "project_id": "YOUR_PROJECT_ID",
    "private_key_id": "...",
    "private_key": "-----BEGIN PRIVATE KEY-----\\n...貼上您的私鑰內容...\\n-----END PRIVATE KEY-----\\n",
    "client_email": "YOUR_SERVICE_ACCOUNT_EMAIL",
    "client_id": "...",
    "auth_uri": "...",
    "token_uri": "...",
    "auth_provider_x509_cert_url": "...",
    "client_x509_cert_url": "..."
}
        """, language="toml")
    # ==================================


# 手動輸入股票代碼 (作為備用或補充)
manual_ticker = st.sidebar.text_input("或手動輸入股票代碼 (例: 2330.TW)", value="TSLA").upper()
ticker_to_run = selected_ticker if selected_ticker else manual_ticker

run_btn = st.sidebar.button("🚀 開始分析")


# --- 4. 主程式邏輯 ---

if run_btn and ticker_to_run:
    if not api_key:
        st.error("⚠️ 請先在左側輸入 Gemini API Key！")
        st.stop()
    
    try:
        df, info, benchmark_df, benchmark_ticker = get_stock_data(ticker_to_run)
        fx_rate = get_fx_rate()
        
        if df is None:
            st.error(f"❌ 找不到股票代碼 **{ticker_to_run}** 的歷史數據。請確認代碼是否正確。")
            st.stop()
        
        # 獲取最新數據
        current_price = df['Close'].iloc[-1]
        latest_rsi = df['RSI'].iloc[-1]
        sma_20 = df['SMA_20'].iloc[-1]
        sma_60 = df['SMA_60'].iloc[-1]
        
        # 均線狀態說明
        sma_20_status = "股價 > 20MA (短期強勢)" if current_price > sma_20 else "股價 < 20MA (短期弱勢)"
        sma_60_status = "股價 > 60MA (中期強勢)" if current_price > sma_60 else "股價 < 60MA (中期弱勢)"
        sma_status = f"20MA趨勢: {sma_20_status} | 60MA趨勢: {sma_60_status}"

        # --- 區塊 A: 資產配置儀表板 ---
        st.header(f"💼 **{info.get('longName', ticker_to_run)} ({ticker_to_run}) 概況**")
        st.caption(f"即時匯率 (USD/TWD): **{fx_rate:.2f}**")
        
        my_holding = portfolio_df[portfolio_df['Ticker'] == ticker_to_run]
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("即時市場價格", f"${current_price:.2f}", f"{df['Close'].diff().iloc[-1]:.2f}")
        col2.metric("RSI (14日強度)", f"{latest_rsi:.2f}%")
        
        if not my_holding.empty:
            holding = my_holding.iloc[0]
            share_qty = holding['Quantity']
            avg_price = holding['AvgPrice']
            currency = holding['Currency']
            
            # 計算市值與損益
            current_market_value = current_price * share_qty
            cost = avg_price * share_qty
            profit = current_market_value - cost
            profit_pct = (profit / cost) * 100 if cost != 0 and not pd.isna(cost) and cost != 0 else 0
            
            # 總台幣價值換算
            exchange_rate = fx_rate if currency == 'USD' else 1
            total_twd_profit = profit * exchange_rate
            
            col3.metric("您的買入均價", f"{currency} {avg_price:,.2f}")
            col4.metric("總未實現損益 (TWD)", f"NT${total_twd_profit:,.0f}", f"{profit_pct:.2f}%")

        else:
            col3.info("無持股數據")
            col4.info("無法計算個人損益")

        # --- 區塊 B: 互動式 K 線圖 ---
        st.subheader("📈 股價趨勢與雙均線 (K線圖)")
        
        fig = go.Figure(data=[
            go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'),
            go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='blue', width=1), name='20 日均線'),
            go.Scatter(x=df.index, y=df['SMA_60'], line=dict(color='red', width=1), name='60 日均線')
        ])
        
        fig.update_layout(
            height=500,
            xaxis_rangeslider_visible=True, # 啟用時間軸滑塊
            xaxis=dict(type="category"), 
            hovermode="x unified",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True}) # 啟用右上角工具欄

        # --- 區塊 D: 個人獲利曲線 vs. 市場大盤 ---
        if not my_holding.empty and share_qty > 0:
            st.subheader("📊 **個人持股累積報酬 vs. 市場大盤比較**")
            
            portfolio_value = df['Close'] * share_qty
            
            comparison_df = pd.DataFrame({
                'Portfolio_Value': portfolio_value,
                'Benchmark_Close': benchmark_df['Close']
            }).dropna()
            
            if not comparison_df.empty:
                comparison_df['Portfolio_Norm'] = (comparison_df['Portfolio_Value'] / comparison_df['Portfolio_Value'].iloc[0]) * 100
                comparison_df['Benchmark_Norm'] = (comparison_df['Benchmark_Close'] / comparison_df['Benchmark_Close'].iloc[0]) * 100

                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['Portfolio_Norm'], 
                                            mode='lines', name='您的持股曲線', line=dict(color='green', width=3)))
                fig_comp.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['Benchmark_Norm'], 
                                            mode='lines', name=f'大盤指數 ({benchmark_ticker})', line=dict(color='orange', width=2, dash='dash')))
                
                fig_comp.update_layout(
                    title="近六個月累積報酬比較 (起始點=100)",
                    xaxis_title="日期",
                    yaxis_title="相對報酬指數 (%)",
                    hovermode="x unified",
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20),
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.warning("數據不足，無法繪製比較曲線。")
        else:
            st.info("請在 Sheets 中設定 **Quantity (持股數量)** 欄位，才能計算與大盤比較的獲利曲線。")

        # --- 區塊 C: Gemini AI 深度分析 ---
        st.divider()
        st.subheader("🤖 Gemini 投資顧問分析報告")
        
        analysis_text = ai_analysis(api_key, ticker_to_run, info, current_price, latest_rsi, sma_status)
        st.markdown(analysis_text)
        
    except Exception as e:
        st.error(f"應用程式運行發生錯誤：{str(e)}")

else:
    st.info("👈 請在左側輸入 Gemini API Key 或設定資產資料庫，並點擊「開始分析」")
