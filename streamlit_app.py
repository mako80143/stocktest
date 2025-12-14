import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import gspread
from gspread_dataframe import get_as_dataframe

# --- 1. 頁面與緩存設定 ---
st.set_page_config(page_title="AI 智能股票儀表板", layout="wide")
st.title("🤖 AI 智能股票分析儀表板")
st.markdown("---")

# 使用緩存機制，避免重複呼叫 API (節省額度並加速)
@st.cache_data(ttl=24*3600) 
def get_stock_data(ticker):
    """抓取股價、計算技術指標與獲取大盤指數"""
    if ticker.endswith('.TW'):
        benchmark_ticker = '^TWII' 
    else:
        benchmark_ticker = '^GSPC' 
        
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

@st.cache_data(ttl=300) 
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
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini API 錯誤：{str(e)}。請確認您的 API Key 是否正確或檢查網路連線。"

# --- 2. 環境檢查函數 ---
def check_environment(api_key):
    """檢查 API Key 和 Sheets Secrets 是否配置，並返回狀態字典"""
    status = {}
    
    # 檢查 Gemini API Key
    status['gemini_ok'] = bool(api_key)

    # 檢查 Google Sheets Secrets
    if "gcp_service_account" in st.secrets and "spreadsheet" in st.secrets:
        status['sheets_ok'] = True
    else:
        status['sheets_ok'] = False
        
    return status

# --- 3. 側邊欄與輸入整合 (UI/UX 升級) ---

st.sidebar.header("⚙️ 應用程式參數設定")

# 獲取 API Key
st.sidebar.subheader("🔑 Gemini API 設定")
api_key = st.sidebar.text_input("請輸入您的 Gemini API Key", type="password")
st.sidebar.caption("還沒有 Key? [點此免費申請](https://aistudio.google.com/app/apikey)")

# 環境檢查
env_status = check_environment(api_key)
if env_status['gemini_ok']:
    st.sidebar.success("✅ Gemini Key 已配置")
else:
    st.sidebar.warning("⚠️ 請輸入 Gemini Key")

st.sidebar.divider()

# --- 4. Google Sheets 資料庫連接與輸入整合 ---

st.sidebar.subheader("🎯 股票代碼選擇")

portfolio_df = pd.DataFrame()
tickers_list = []
selected_ticker = ''

# 嘗試連接 Google Sheets
if env_status['sheets_ok']:
    try:
        # 設置 gspread 連接
        creds = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(creds)
        
        # 開啟 Sheets 檔案
        spreadsheet_id = st.secrets["spreadsheet"]["id"]
        sh = gc.open_by_key(spreadsheet_id)
        
        # 讀取 'Portfolio' 工作表
        worksheet = sh.worksheet("Portfolio")
        portfolio_df = get_as_dataframe(worksheet, header=0, usecols=['Ticker', 'Quantity', 'AvgPrice', 'Currency']).dropna(subset=['Ticker'])
        
        tickers_list = portfolio_df['Ticker'].tolist()
        st.sidebar.success("✅ Sheets 資料庫連線成功")
        
    except Exception as e:
        st.sidebar.error("❌ Sheets 連線失敗，請檢查權限或金鑰。")
        env_status['sheets_ok'] = False # 連線失敗就視為未配置

if env_status['sheets_ok'] and tickers_list:
    options = [''] + tickers_list
    placeholder = "請從持股清單中選擇或手動輸入..."
else:
    options = [''] 
    placeholder = "請手動輸入股票代碼 (例: 2330.TW)"

# 整合輸入欄位
ticker_input = st.sidebar.text_input(
    placeholder,
    value=options[0] if options else "TSLA",
    key="ticker_input"
).upper()

if ticker_input in tickers_list:
    ticker_to_run = ticker_input
else:
    ticker_to_run = ticker_input

run_btn = st.sidebar.button("🚀 開始分析")

# Sheets 錯誤輔助區塊
if not env_status['sheets_ok']:
    with st.sidebar.expander("❓ Google Sheets 連線輔助"):
        st.markdown("#### **Sheets 數據庫配置**")
        st.markdown("**請注意：** 欄位名稱必須為 `Ticker`, `Quantity`, `AvgPrice`, `Currency`。")
        st.markdown("#### **Streamlit Secrets 貼上格式**")
        st.code("""
[gcp_service_account]
# 貼上您下載的 JSON 金鑰檔案的全部內容
type = "service_account"
# ... (其他 JSON 欄位)

[spreadsheet]
id = "請貼上您的 Google Sheet ID"
        """, language="toml")


# --- 5. 主程式邏輯 ---

if run_btn and ticker_to_run and env_status['gemini_ok']:
    
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
        sma_20_status = "股價 > 20MA (短期強勢)" if current_price > sma_20 else "股價 < 20MA (短期弱勢)"
        sma_60_status = "股價 > 60MA (中期強勢)" if current_price > sma_60 else "股價 < 60MA (中期弱勢)"
        sma_status = f"20MA趨勢: {sma_20_status} | 60MA趨勢: {sma_60_status}"

        # --- 區塊 A: 概況儀表板 ---
        st.header(f"💼 **{info.get('longName', ticker_to_run)} ({ticker_to_run}) 概況**")
        st.caption(f"即時匯率 (USD/TWD): **{fx_rate:.2f}**")
        
        my_holding = portfolio_df[portfolio_df['Ticker'] == ticker_to_run]
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("即時市場價格", f"${current_price:.2f}", f"{df['Close'].diff().iloc[-1]:.2f}")
        col2.metric("RSI (14日強度)", f"{latest_rsi:.2f}%")
        
        # 個人持股計算
        share_qty = 0
        if not my_holding.empty:
            holding = my_holding.iloc[0]
            share_qty = holding['Quantity']
            avg_price = holding['AvgPrice']
            currency = holding['Currency']
            
            current_market_value = current_price * share_qty
            cost = avg_price * share_qty
            profit = current_market_value - cost
            profit_pct = (profit / cost) * 100 if cost != 0 and not pd.isna(cost) and cost != 0 else 0
            
            exchange_rate = fx_rate if currency == 'USD' else 1
            total_twd_profit = profit * exchange_rate
            
            col3.metric("您的買入均價", f"{currency} {avg_price:,.2f}")
            col4.metric("總未實現損益 (TWD)", f"NT${total_twd_profit:,.0f}", f"{profit_pct:.2f}%")

        else:
            col3.info("無持股數據")
            col4.info("無法計算個人損益")
            
        st.divider()

        # --- 區塊 B: 使用 st.tabs 進行 UI 分隔 ---
        tab1, tab2, tab3 = st.tabs(["📊 技術線圖", "🤖 AI 深度分析", "📈 報酬比較"])

        with tab1:
            st.subheader("股價趨勢與雙均線 (K線圖)")
            fig = go.Figure(data=[
                go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'),
                go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='blue', width=1), name='20 日均線'),
                go.Scatter(x=df.index, y=df['SMA_60'], line=dict(color='red', width=1), name='60 日均線')
            ])
            
            fig.update_layout(
                height=500, xaxis_rangeslider_visible=True, xaxis=dict(type="category"), 
                hovermode="x unified", margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})

        with tab2:
            st.subheader("Gemini 投資顧問分析報告")
            with st.spinner('🧠 AI 正在進行深度分析...'):
                analysis_text = ai_analysis(api_key, ticker_to_run, info, current_price, latest_rsi, sma_status)
                st.markdown(analysis_text)

        with tab3:
            st.subheader("個人持股累積報酬 vs. 市場大盤比較")
            if share_qty > 0:
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
                        title="近六個月累積報酬比較 (起始點=100)", xaxis_title="日期", yaxis_title="相對報酬指數 (%)",
                        hovermode="x unified", height=400, margin=dict(l=20, r=20, t=50, b=20)
                    )
                    
                    st.plotly_chart(fig_comp, use_container_width=True)
                else:
                    st.warning("數據不足，無法繪製比較曲線。")
            else:
                st.info("請在 Sheets 中設定 **Quantity (持股數量)** 欄位，才能計算與大盤比較的獲利曲線。")

        
    except Exception as e:
        st.error(f"應用程式運行發生錯誤：{str(e)}")

elif run_btn and not ticker_to_run:
    st.error("⚠️ 請輸入或選擇一支股票代碼！")
    
elif not env_status['gemini_ok']:
    st.info("👈 請先在側邊欄輸入您的 Gemini API Key 來啟用 AI 分析功能。")
