import streamlit as st
import yfinance as yf
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from streamlit_autorefresh import st_autorefresh
st.set_page_config(page_title="Live Gold Dashboard", layout="wide")
st.title("LIVE Gold Trading Dashboard")
#  1 second refresh
st_autorefresh(interval=10000, key="refresh")
symbol = st.text_input("Symbol (Gold = GC=F)", "GC=F")
df = yf.download(symbol, period="1d", interval="1m", progress=False)
if df.empty:
    st.error("Data not found")
    st.stop()
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.dropna()
close = df["Close"]
latest = df.iloc[-1]
st.subheader("Live Price")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Open", round(float(latest["Open"]),2))
c2.metric("High", round(float(latest["High"]),2))
c3.metric("Low", round(float(latest["Low"]),2))
c4.metric("Close", round(float(latest["Close"]),2))
st.write("---")
st.subheader("Indicators")
df["SMA"] = SMAIndicator(close, window=20).sma_indicator()
df["EMA"] = EMAIndicator(close, window=20).ema_indicator()
df["RSI"] = RSIIndicator(close, window=14).rsi()
macd = MACD(close)
df["MACD"] = macd.macd()
df["MACD Signal"] = macd.macd_signal()
bb = BollingerBands(close)
df["BB Upper"] = bb.bollinger_hband()
df["BB Lower"] = bb.bollinger_lband()
st.subheader("Price + EMA + SMA")
st.line_chart(df[["Close","EMA","SMA"]])
st.subheader("RSI")
st.line_chart(df["RSI"])
st.subheader("MACD")
st.line_chart(df[["MACD","MACD Signal"]])
st.subheader("Bollinger Bands")
st.line_chart(df[["Close","BB Upper","BB Lower"]])
