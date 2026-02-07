import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="MarketMind AI Hedge Fund Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .signal-buy {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .signal-sell {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .signal-neutral {
        background: linear-gradient(135deg, #8e9eab 0%, #eef2f3 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: #333;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header"> MarketMind AI - Hedge Fund Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/stock-share.png", width=80)
    st.markdown("###  Dashboard Controls")
    
    # Asset Selection
    asset_classes = {
        "Forex": ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD"],
        "Cryptocurrency": ["BTC/USD", "ETH/USD", "BNB/USD", "XRP/USD", "SOL/USD"],
        "Indices": ["S&P 500", "NASDAQ", "Dow Jones", "FTSE 100", "DAX"],
        "Commodities": ["Gold", "Silver", "Crude Oil", "Natural Gas", "Copper"],
        "Stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    }
    
    selected_class = st.selectbox("Asset Class", list(asset_classes.keys()))
    asset = st.selectbox("Asset", asset_classes[selected_class])
    
    symbols = {
        "Gold": "GC=F", "Silver": "SI=F", "Crude Oil": "CL=F", "Natural Gas": "NG=F", "Copper": "HG=F",
        "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X", "USD/CHF": "CHF=X", "AUD/USD": "AUDUSD=X",
        "BTC/USD": "BTC-USD", "ETH/USD": "ETH-USD", "BNB/USD": "BNB-USD", "XRP/USD": "XRP-USD", "SOL/USD": "SOL-USD",
        "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow Jones": "^DJI", "FTSE 100": "^FTSE", "DAX": "^GDAXI",
        "AAPL": "AAPL", "MSFT": "MSFT", "GOOGL": "GOOGL", "AMZN": "AMZN", "TSLA": "TSLA"
    }
    
    symbol = symbols[asset]
    
    # Timeframe selection
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "1d"])
    period_map = {
        "1m": "1d", "5m": "5d", "15m": "15d", 
        "30m": "30d", "1h": "60d", "1d": "1y"
    }
    period = period_map[timeframe]
    
    # AI Settings
    st.markdown("---")
    st.markdown("###  AI Settings")
    use_ml = st.checkbox("Enable Machine Learning Predictions", value=True)
    risk_level = st.select_slider("Risk Level", options=["Low", "Medium", "High"], value="Medium")
    
    # Auto-refresh
    refresh_rate = st.selectbox("Refresh Rate", ["Manual", "10s", "30s", "1m", "5m"])
    if refresh_rate != "Manual":
        seconds = {"10s": 10, "30s": 30, "1m": 60, "5m": 300}[refresh_rate]
        st_autorefresh(interval=seconds*1000, key="refresh")

# Download data
@st.cache_data(ttl=30)
def load_data(symbol, period, interval):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.dropna()
        return df
    except:
        return pd.DataFrame()

df = load_data(symbol, period, timeframe)

if df.empty:
    st.error(" No data available for selected asset/timeframe")
    st.stop()

# Calculate advanced indicators
close = df["Close"]
high = df["High"]
low = df["Low"]
volume = df["Volume"]

# Trend indicators
df["SMA_20"] = SMAIndicator(close, window=20).sma_indicator()
df["SMA_50"] = SMAIndicator(close, window=50).sma_indicator()
df["EMA_12"] = EMAIndicator(close, window=12).ema_indicator()
df["EMA_26"] = EMAIndicator(close, window=26).ema_indicator()

macd = MACD(close)
df["MACD"] = macd.macd()
df["MACD_Signal"] = macd.macd_signal()
df["MACD_Hist"] = macd.macd_diff()

# Momentum indicators
df["RSI"] = RSIIndicator(close, window=14).rsi()
stoch = StochasticOscillator(high=high, low=low, close=close)
df["Stoch_K"] = stoch.stoch()
df["Stoch_D"] = stoch.stoch_signal()

# Volatility indicators
bb = BollingerBands(close, window=20, window_dev=2)
df["BB_Upper"] = bb.bollinger_hband()
df["BB_Lower"] = bb.bollinger_lband()
df["BB_Middle"] = bb.bollinger_mavg()
df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]

atr = AverageTrueRange(high=high, low=low, close=close, window=14)
df["ATR"] = atr.average_true_range()

# Volume indicators
df["OBV"] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
df["VWAP"] = VolumeWeightedAveragePrice(
    high=high, low=low, close=close, volume=volume, window=20
).volume_weighted_average_price()

# Advanced trend detection
def advanced_trend_analysis(df):
    last = df.iloc[-1]
    
    # Multiple timeframe analysis
    trend_score = 0
    
    # EMA alignment
    if last["EMA_12"] > last["EMA_26"] > last["SMA_50"]:
        trend_score += 2
    elif last["EMA_12"] < last["EMA_26"] < last["SMA_50"]:
        trend_score -= 2
    
    # MACD trend
    if last["MACD"] > last["MACD_Signal"] and df["MACD"].iloc[-5:].mean() > 0:
        trend_score += 1
    elif last["MACD"] < last["MACD_Signal"] and df["MACD"].iloc[-5:].mean() < 0:
        trend_score -= 1
    
    # RSI trend
    if last["RSI"] > 60:
        trend_score += 0.5
    elif last["RSI"] < 40:
        trend_score -= 0.5
    
    # Determine trend
    if trend_score >= 2:
        return "STRONG UPTREND 📈", trend_score, "success"
    elif trend_score >= 1:
        return "UPTREND ", trend_score, "info"
    elif trend_score <= -2:
        return "STRONG DOWNTREND ", trend_score, "error"
    elif trend_score <= -1:
        return "DOWNTREND ", trend_score, "warning"
    else:
        return "CONSOLIDATION ", trend_score, "secondary"

# AI Signal Generation
def generate_ai_signal(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    signal_score = 0
    reasons = []
    
    # RSI conditions
    if last["RSI"] < 30:
        signal_score += 2
        reasons.append("RSI oversold")
    elif last["RSI"] > 70:
        signal_score -= 2
        reasons.append("RSI overbought")
    elif 40 < last["RSI"] < 60:
        signal_score += 0.5
    
    # MACD crossover
    if last["MACD"] > last["MACD_Signal"] and prev["MACD"] <= prev["MACD_Signal"]:
        signal_score += 2
        reasons.append("MACD bullish crossover")
    elif last["MACD"] < last["MACD_Signal"] and prev["MACD"] >= prev["MACD_Signal"]:
        signal_score -= 2
        reasons.append("MACD bearish crossover")
    
    # Bollinger Band position
    if last["Close"] < last["BB_Lower"]:
        signal_score += 1.5
        reasons.append("Price at lower BB")
    elif last["Close"] > last["BB_Upper"]:
        signal_score -= 1.5
        reasons.append("Price at upper BB")
    
    # Volume confirmation
    if last["Volume"] > df["Volume"].rolling(20).mean().iloc[-1] * 1.5:
        if last["Close"] > last["Open"]:
            signal_score += 1
            reasons.append("High volume on up move")
        else:
            signal_score -= 1
            reasons.append("High volume on down move")
    
    # Stochastic
    if last["Stoch_K"] < 20 and last["Stoch_D"] < 20:
        signal_score += 1
        reasons.append("Stochastic oversold")
    elif last["Stoch_K"] > 80 and last["Stoch_D"] > 80:
        signal_score -= 1
        reasons.append("Stochastic overbought")
    
    # Generate signal
    if signal_score >= 4:
        return "STRONG BUY ", signal_score, reasons, "buy"
    elif signal_score >= 2:
        return "BUY ", signal_score, reasons, "buy"
    elif signal_score <= -4:
        return "STRONG SELL ", signal_score, reasons, "sell"
    elif signal_score <= -2:
        return "SELL ", signal_score, reasons, "sell"
    else:
        return "NEUTRAL ", signal_score, reasons, "neutral"

# Calculate support and resistance
def calculate_support_resistance(df, window=20):
    highs = df["High"].rolling(window=window).max()
    lows = df["Low"].rolling(window=window).min()
    
    support_levels = lows.iloc[-5:].unique()
    resistance_levels = highs.iloc[-5:].unique()
    
    current_price = df["Close"].iloc[-1]
    
    nearest_support = max([s for s in support_levels if s < current_price], default=min(support_levels))
    nearest_resistance = min([r for r in resistance_levels if r > current_price], default=max(resistance_levels))
    
    return nearest_support, nearest_resistance, support_levels, resistance_levels

# Get analysis results
trend, trend_score, trend_color = advanced_trend_analysis(df)
signal, signal_score, reasons, signal_type = generate_ai_signal(df)
support, resistance, sup_levels, res_levels = calculate_support_resistance(df)

# Main Dashboard Layout
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}", 
              f"{((df['Close'].iloc[-1] - df['Close'].iloc[-2])/df['Close'].iloc[-2]*100):.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("24h Volume", f"{df['Volume'].iloc[-1]:,.0f}", 
              f"{((df['Volume'].iloc[-1] - df['Volume'].iloc[-24:].mean())/df['Volume'].iloc[-24:].mean()*100):.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("ATR (Volatility)", f"{df['ATR'].iloc[-1]:.2f}", 
              f"{((df['ATR'].iloc[-1] - df['ATR'].mean())/df['ATR'].mean()*100):.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}", 
              "Oversold" if df['RSI'].iloc[-1] < 30 else "Overbought" if df['RSI'].iloc[-1] > 70 else "Neutral")
    st.markdown('</div>', unsafe_allow_html=True)

# Signal and Trend Row
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader(" AI Trading Signal")
    if signal_type == "buy":
        st.markdown(f'<div class="signal-buy">{signal}</div>', unsafe_allow_html=True)
    elif signal_type == "sell":
        st.markdown(f'<div class="signal-sell">{signal}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="signal-neutral">{signal}</div>', unsafe_allow_html=True)
    
    st.caption(f"Signal Score: {signal_score:.1f}")
    if reasons:
        with st.expander("Signal Reasons"):
            for reason in reasons:
                st.write(f"• {reason}")

with col2:
    st.subheader(" Market Trend")
    if trend_color == "success":
        st.success(trend)
    elif trend_color == "error":
        st.error(trend)
    elif trend_color == "warning":
        st.warning(trend)
    else:
        st.info(trend)
    st.caption(f"Trend Score: {trend_score:.1f}")

with col3:
    st.subheader(" Key Levels")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Support", f"${support:.2f}")
    with col_b:
        st.metric("Resistance", f"${resistance:.2f}")
    st.progress((df['Close'].iloc[-1] - support) / (resistance - support) if resistance > support else 0.5)

# Advanced Charts
st.markdown("---")
st.subheader(" Advanced Chart Analysis")

# Create Plotly chart
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.6, 0.2, 0.2],
    subplot_titles=(f"{asset} Price with Indicators", "MACD", "RSI")
)

# Price with Bollinger Bands
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Price"
    ),
    row=1, col=1
)

# Bollinger Bands
fig.add_trace(
    go.Scatter(x=df.index, y=df['BB_Upper'], 
               line=dict(color='rgba(255,0,0,0.3)', width=1),
               name='BB Upper'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df.index, y=df['BB_Lower'], 
               line=dict(color='rgba(0,255,0,0.3)', width=1),
               fill='tonexty',
               fillcolor='rgba(0,255,0,0.1)',
               name='BB Lower'),
    row=1, col=1
)

# Moving averages
fig.add_trace(
    go.Scatter(x=df.index, y=df['SMA_50'], 
               line=dict(color='orange', width=2),
               name='SMA 50'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df.index, y=df['EMA_12'], 
               line=dict(color='blue', width=2),
               name='EMA 12'),
    row=1, col=1
)

# MACD
fig.add_trace(
    go.Scatter(x=df.index, y=df['MACD'], 
               line=dict(color='blue', width=1),
               name='MACD'),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=df.index, y=df['MACD_Signal'], 
               line=dict(color='red', width=1),
               name='Signal'),
    row=2, col=1
)

# MACD Histogram
colors = ['green' if val >= 0 else 'red' for val in df['MACD_Hist']]
fig.add_trace(
    go.Bar(x=df.index, y=df['MACD_Hist'], 
           marker_color=colors,
           name='MACD Hist'),
    row=2, col=1
)

# RSI
fig.add_trace(
    go.Scatter(x=df.index, y=df['RSI'], 
               line=dict(color='purple', width=2),
               name='RSI'),
    row=3, col=1
)

# RSI levels
fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)

# Update layout
fig.update_layout(
    height=800,
    showlegend=True,
    xaxis_rangeslider_visible=False,
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# Technical Indicators Dashboard
st.markdown("---")
st.subheader("Technical Indicators Overview")

ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)

with ind_col1:
    st.metric("MACD", f"{df['MACD'].iloc[-1]:.4f}", 
              f"{(df['MACD'].iloc[-1] - df['MACD_Signal'].iloc[-1]):.4f}")
    st.caption("Above Signal: Bullish" if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else "Below Signal: Bearish")

with ind_col2:
    st.metric("Stochastic %K", f"{df['Stoch_K'].iloc[-1]:.1f}")
    st.caption("Oversold" if df['Stoch_K'].iloc[-1] < 20 else "Overbought" if df['Stoch_K'].iloc[-1] > 80 else "Neutral")

with ind_col3:
    st.metric("BB Width %", f"{(df['BB_Width'].iloc[-1]*100):.2f}%")
    st.caption("High Vol" if df['BB_Width'].iloc[-1] > df['BB_Width'].mean() else "Low Vol")

with ind_col4:
    obv_trend = "Bullish" if df['OBV'].iloc[-1] > df['OBV'].iloc[-5] else "Bearish"
    st.metric("OBV", f"{df['OBV'].iloc[-1]:,.0f}", obv_trend)

# Risk Metrics
st.markdown("---")
st.subheader(" Risk Assessment")

risk_col1, risk_col2, risk_col3 = st.columns(3)

with risk_col1:
    st.info(f"**Current ATR**: ${df['ATR'].iloc[-1]:.2f}")
    st.progress(min(df['ATR'].iloc[-1] / df['ATR'].max(), 1.0))
    st.caption("Volatility indicator")

with risk_col2:
    max_drawdown = ((df['Close'].cummax() - df['Close']) / df['Close'].cummax() * 100).iloc[-1]
    st.warning(f"**Max Drawdown**: {max_drawdown:.2f}%")
    st.progress(min(max_drawdown / 20, 1.0))
    st.caption("Peak to trough decline")

with risk_col3:
    sharpe_ratio = df['Close'].pct_change().mean() / df['Close'].pct_change().std() * np.sqrt(252)
    if pd.notna(sharpe_ratio):
        st.success(f"**Sharpe Ratio**: {sharpe_ratio:.2f}")
        color = "green" if sharpe_ratio > 1 else "orange" if sharpe_ratio > 0 else "red"
        st.markdown(f'<span style="color:{color}">{"Good" if sharpe_ratio > 1 else "Average" if sharpe_ratio > 0 else "Poor"}</span>', unsafe_allow_html=True)

# Prediction Section
if use_ml:
    st.markdown("---")
    st.subheader(" Machine Learning Predictions")
    
    # Simple prediction model (for demonstration)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    # Prepare features
    features = pd.DataFrame()
    features['returns'] = df['Close'].pct_change()
    features['volume_change'] = df['Volume'].pct_change()
    features['rsi'] = df['RSI']
    features['macd'] = df['MACD']
    features['bb_position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Create target (next period return)
    features['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    features = features.dropna()
    
    if len(features) > 100:
        X = features.drop('target', axis=1)
        y = features['target']
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X[:-1], y[:-1])
        
        # Predict
        prediction = model.predict_proba(X.iloc[-1:].values)[0]
        confidence = max(prediction) * 100
        
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        
        with pred_col1:
            if prediction[1] > prediction[0]:
                st.success(f"**Prediction**: UP")
            else:
                st.error(f"**Prediction**: DOWN")
        
        with pred_col2:
            st.metric("Confidence", f"{confidence:.1f}%")
        
        with pred_col3:
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            with st.expander("Feature Importance"):
                for _, row in feature_importance.head().iterrows():
                    st.write(f"{row['feature']}: {row['importance']:.3f}")

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with footer_col2:
    st.caption(f"Data Points: {len(df)}")
with footer_col3:
    st.caption("For educational purposes only")