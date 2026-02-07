import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="MarketMind Hedge Fund AI", layout="wide")
st.title("MarketMind AI - Hedge Fund Dashboard")

# auto refresh 10 sec
st_autorefresh(interval=10000, key="refresh")

# -------- MULTI ASSET ----------
symbols = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "BTC/USD": "BTC-USD",
    "ETH/USD": "ETH-USD",
    "NASDAQ": "^IXIC",
    "S&P 500": "^GSPC"
}

asset = st.selectbox("Select Asset", list(symbols.keys()))
symbol = symbols[asset]

df = yf.download(symbol, period="1d", interval="1m", progress=False)

if df.empty:
    st.error("No data")
    st.stop()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.dropna()
close = df["Close"]
latest = df.iloc[-1]

# -------- PRICE PANEL ----------
st.subheader(f" Live Price : {asset}")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Open", round(float(latest["Open"]),2))
c2.metric("High", round(float(latest["High"]),2))
c3.metric("Low", round(float(latest["Low"]),2))
c4.metric("Close", round(float(latest["Close"]),2))

# -------- INDICATORS ----------
df["SMA"] = SMAIndicator(close, window=20).sma_indicator()
df["EMA"] = EMAIndicator(close, window=20).ema_indicator()
df["RSI"] = RSIIndicator(close, window=14).rsi()

macd = MACD(close)
df["MACD"] = macd.macd()
df["MACD Signal"] = macd.macd_signal()

bb = BollingerBands(close)
df["BB Upper"] = bb.bollinger_hband()
df["BB Lower"] = bb.bollinger_lband()

# -------- TREND DETECTION AI ----------
def trend_ai(df):
    last = df.iloc[-1]
    if last["EMA"] > last["SMA"]:
        return "UPTREND "
    elif last["EMA"] < last["SMA"]:
        return "DOWNTREND"
    else:
        return "SIDEWAYS"

trend = trend_ai(df)

# -------- SUPPORT RESISTANCE ----------
support = df["Low"].tail(50).min()
resistance = df["High"].tail(50).max()

# -------- AI SIGNAL ----------
def ai_signal(df):
    last = df.iloc[-1]
    score = 0
    
    if last["RSI"] < 30: score += 2
    if last["RSI"] > 70: score -= 2
    if last["MACD"] > last["MACD Signal"]: score += 2
    else: score -= 2
    if last["EMA"] > last["SMA"]: score += 1
    else: score -= 1
    
    if score >= 3:
        return "STRONG BUY "
    elif score <= -3:
        return "STRONG SELL "
    elif score > 0:
        return "BUY "
    elif score < 0:
        return "SELL "
    else:
        return "WAIT "

signal = ai_signal(df)

# -------- PREDICTION MODEL ----------
df["Future"] = df["Close"].shift(-1)
df["Target"] = (df["Future"] > df["Close"]).astype(int)

prediction = df["Target"].iloc[-2]
pred_text = "NEXT CANDLE UP " if prediction==1 else "NEXT CANDLE DOWN "

# -------- DASHBOARD ----------
st.write("---")
col1,col2,col3 = st.columns(3)

with col1:
    st.subheader(" AI Signal")
    st.success(signal)

with col2:
    st.subheader(" Market Trend")
    st.info(trend)

with col3:
    st.subheader("AI Prediction")
    st.warning(pred_text)

# -------- SUPPORT RESISTANCE ----------
st.subheader("Support / Resistance")
c1,c2 = st.columns(2)
c1.metric("Support", round(float(support),2))
c2.metric("Resistance", round(float(resistance),2))

# -------- CHARTS ----------
st.subheader("Price + EMA + SMA")
st.line_chart(df[["Close","EMA","SMA"]])

st.subheader("RSI")
st.line_chart(df["RSI"])

st.subheader("MACD")
st.line_chart(df[["MACD","MACD Signal"]])

st.subheader("Bollinger Bands")
st.line_chart(df[["Close","BB Upper","BB Lower"]])
