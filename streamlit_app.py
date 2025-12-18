import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import backtrader as bt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# --- 頁面設定 ---
st.set_page_config(page_title="Professional Quant Backtest v5", layout="wide")

# ==========================================
# 1. Backtrader 專業策略引擎
# ==========================================
class ProStrategy(bt.Strategy):
    params = (
        ('config', {}),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.vix = self.getdatabyname('VIX').close if 'VIX' in self.dnames else None
        self.order = None
        self.trade_list = [] # 用於記錄每筆交易明細
        
        # 動態指標初始化
        c = self.params.config
        self.ma_fast = bt.indicators.SMA(self.datas[0], period=c['ma_fast'])
        self.ma_slow = bt.indicators.SMA(self.datas[0], period=c['ma_slow'])
        self.rsi = bt.indicators.RSI(self.datas[0], period=c['rsi_period'])
        self.macd = bt.indicators.MACD(self.datas[0])
        self.crossover = bt.indicators.CrossOver(self.ma_fast, self.ma_slow)

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_list.append({
                'Date': self.data.datetime.date(0),
                'Type': 'CLOSE',
                'Price': trade.price,
                'Profit': trade.pnl,
                'Profit_Pct': (trade.pnl / (trade.price * trade.size)) * 100 if trade.size else 0
            })

    def notify_order(self, order):
        if order.status in [order.Completed]:
            type_str = 'BUY' if order.isbuy() else 'SELL'
            self.trade_list.append({
                'Date': self.data.datetime.date(0),
                'Type': type_str,
                'Price': order.executed.price,
                'Size': order.executed.size,
                'Value': order.executed.value
            })

    def next(self):
        if self.order: return
        
        c = self.params.config
        
        # --- 1. 進場邏輯 (Sequential Priority: VIX -> Tech) ---
        if not self.position:
            buy_signal = False
            
            # A. 宏觀濾網 (優先順位)
            vix_ok = True
            if c['use_vix'] and self.vix:
                if self.vix[0] < c['vix_threshold']: # 假設設定 26，低於此值不買
                    vix_ok = False
            
            if vix_ok:
                # B. 技術指標 (同時滿足 AND 邏輯)
                ma_cond = self.crossover > 0 if 'MA' in c['active_ind'] else True
                rsi_cond = self.rsi[0] < c['rsi_buy'] if 'RSI' in c['active_ind'] else True
                
                if ma_cond and rsi_cond:
                    # 分批買入設定 (例如買入可用資金的 X%)
                    target_pct = c['buy_pct'] / 100
                    self.order_target_percent(target=target_pct)
        
        # --- 2. 出場邏輯 ---
        else:
            # 強制停損
            cost = self.position.price
            if (self.dataclose[0] - cost) / cost < -c['stop_loss']:
                self.close()
                return

            # 策略出場
            sell_signal = False
            if 'MA' in c['active_ind'] and self.crossover < 0: sell_signal = True
            if 'RSI' in c['active_ind'] and self.rsi[0] > c['rsi_sell']: sell_signal = True
            
            if sell_signal:
                self.close()

# ==========================================
# 2. 專業繪圖引擎 (仿富途牛牛三分屏)
# ==========================================
def plot_v5(df, df_bench, trades, equity):
    # 建立子圖
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("主圖：K線與交易點位", "副圖一：RSI 動能", "副圖二：策略淨值 vs 大盤基準"),
        row_heights=[0.5, 0.2, 0.3]
    )

    # 1. K線圖
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="TSLA"), row=1, col=1)
    
    # 2. 均線
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_F'], name="Fast MA", line=dict(width=1, color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_S'], name="Slow MA", line=dict(width=1, color='blue')), row=1, col=1)

    # 3. 交易點位標記
    if not trades.empty:
        buys = trades[trades['Type'] == 'BUY']
        sells = trades[trades['Type'] == 'SELL']
        fig.add_trace(go.Scatter(x=buys['Date'], y=buys['Price'], mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime'), name='買入點'), row=1, col=1)
        fig.add_trace(go.Scatter(x=sells['Date'], y=sells['Price'], mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name='賣出點'), row=1, col=1)

    # 4. 副圖一：RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # 5. 副圖二：淨值曲線對比
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="我的策略", fill='tozeroy', line=dict(color='red')), row=3, col=1)
    
    # 計算大盤 (QQQ) 基準
    bench_norm = (df_bench['Close'] / df_bench['Close'].iloc[0]) * equity.iloc[0]
    fig.add_trace(go.Scatter(x=bench_norm.index, y=bench_norm.values, name="大盤基準(持股不動)", line=dict(color='gray', dash='dot')), row=3, col=1)

    fig.update_layout(height=900, hovermode='x unified', template='plotly_dark', xaxis_rangeslider_visible=False)
    return fig

# ==========================================
# 3. Streamlit UI 介面
# ==========================================
st.title("🏹 V5 Pro Quant System")

with st.sidebar:
    st.header("⚙️ 全域參數設定")
    symbol = st.text_input("股票代號", "TSLA")
    vix_symbol = "^VIX"
    bench_symbol = "QQQ"
    
    dates = st.date_input("回測區間", [datetime.date(2025, 1, 1), datetime.date(2025, 12, 15)])
    cash = st.number_input("起始資金", 100000)
    
    st.divider()
    st.header("📈 指標庫與權重")
    active_ind = st.multiselect("啟用指標 (AND 邏輯)", ["MA", "RSI", "MACD"], default=["MA", "RSI"])
    
    buy_pct = st.slider("每次進場買入比例 (%)", 10, 100, 50)
    stop_loss = st.slider("強制停損 (%)", 1, 20, 10) / 100

    if "MA" in active_ind:
        st.subheader("MA 設定")
        ma_f = st.number_input("短均線", 5, 50, 10)
        ma_s = st.number_input("長均線", 10, 200, 50)
    
    if "RSI" in active_ind:
        st.subheader("RSI 設定")
        rsi_p = st.number_input("RSI 週期", 5, 30, 14)
        rsi_b = st.slider("買入閾值", 10, 40, 30)
        rsi_s = st.slider("賣出閾值", 60, 90, 70)

    st.divider()
    st.header("🌪️ 宏觀濾網")
    use_vix = st.checkbox("啟用 VIX 恐慌濾網", True)
    vix_t = st.number_input("VIX 買入警戒值 (高於此值買入)", 15, 50, 26)

# --- 執行回測 ---
if st.button("🚀 開始專業回測", type="primary"):
    with st.spinner("正在獲取數據與計算..."):
        # 1. 抓取數據
        df = yf.download(symbol, start=dates[0], end=dates[1])
        df_vix = yf.download(vix_symbol, start=dates[0], end=dates[1])
        df_bench = yf.download(bench_symbol, start=dates[0], end=dates[1])

        # 處理資料清洗
        for d in [df, df_vix, df_bench]:
            if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)

        # 2. 初始化 Backtrader
        cerebro = bt.Cerebro()
        data_feed = bt.feeds.PandasData(dataname=df)
        vix_feed = bt.feeds.PandasData(dataname=df_vix)
        cerebro.adddata(data_feed, name=symbol)
        cerebro.adddata(vix_feed, name='VIX')
        
        config = {
            'active_ind': active_ind, 'ma_fast': ma_f, 'ma_slow': ma_s,
            'rsi_period': rsi_p, 'rsi_buy': rsi_b, 'rsi_sell': rsi_s,
            'use_vix': use_vix, 'vix_threshold': vix_t, 'buy_pct': buy_pct, 'stop_loss': stop_loss
        }
        
        cerebro.addstrategy(ProStrategy, config=config)
        cerebro.broker.setcash(cash)
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
        
        results = cerebro.run()
        strat = results[0]
        
        # 3. 處理結果
        equity_dict = strat.analyzers.timereturn.get_analysis()
        equity_curve = pd.Series(equity_dict).fillna(0).cumsum().apply(lambda x: cash * (1+x))
        trades_df = pd.DataFrame(strat.trade_list)
        
        # 準備繪圖數據
        df['MA_F'] = df['Close'].rolling(ma_f).mean()
        df['MA_S'] = df['Close'].rolling(ma_s).mean()
        df['RSI'] = ta.rsi(df['Close'], length=rsi_p)

        # --- 4. 顯示結果 ---
        st.subheader("📊 績效走勢看板")
        st.plotly_chart(plot_v5(df, df_bench, trades_df, equity_curve), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📜 交易詳細日誌")
            if not trades_df.empty:
                st.dataframe(trades_df.style.highlight_max(axis=0, color='#2e7d32'), use_container_width=True)
            else:
                st.write("此期間無交易發生。")
        
        with col2:
            st.subheader("💡 策略教育對比")
            final_strat = (equity_curve.iloc[-1] - cash) / cash * 100
            final_bench = (df_bench['Close'].iloc[-1] - df_bench['Close'].iloc[0]) / df_bench['Close'].iloc[0] * 100
            
            st.metric("策略最終報酬率", f"{final_strat:.2f}%", delta=f"{final_strat - final_bench:.2f}% vs 大盤")
            st.info(f"如果你一開始就買入 {symbol} 並持有到結束，獲利率會是 {((df['Close'].iloc[-1]-df['Close'].iloc[0])/df['Close'].iloc[0]*100):.2f}%。這能幫助你判斷頻繁交易是否有意義。")
