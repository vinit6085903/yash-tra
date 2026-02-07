import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="MarketMind AI", layout="wide")
st.title(" MarketMind AI ")

# ===== SIDEBAR =====
with st.sidebar:
    st.header("⚙ Trading Settings")

    symbols = {
        "Gold (XAUUSD)": "GC=F",
        "Silver": "SI=F",
        "BTC/USD": "BTC-USD",
        "ETH/USD": "ETH-USD",
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "NASDAQ": "^IXIC",
        "S&P500": "^GSPC"
    }

    asset = st.selectbox("Select Asset", list(symbols.keys()))
    symbol = symbols[asset]

    tf = st.selectbox("Timeframe", ["1m","5m","15m","1h","1d"])
    capital = st.number_input("Capital ($)", value=100)
    risk = st.slider("Risk %",1,10,2)

    refresh = st.selectbox("Auto Refresh", ["Off","10s","30s"])
    if refresh!="Off":
        sec={"10s":10,"30s":30}[refresh]
        st_autorefresh(interval=sec*1000,key="refresh")

# ===== DATA FETCH =====
period_map={
"1m":"1d","5m":"5d","15m":"15d","1h":"60d","1d":"1y"
}

df = yf.download(symbol, period=period_map[tf], interval=tf, progress=False)

# ===== EMPTY DATA FIX =====
if df is None or df.empty:
    st.error("No market data available for the selected asset and timeframe.")
    st.stop()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.dropna()

if df.empty or len(df)<50:
    st.warning("Data loading...")
    st.stop()

close=df["Close"]

# ===== INDICATORS =====
df["EMA20"]=EMAIndicator(close,20).ema_indicator()
df["SMA50"]=SMAIndicator(close,50).sma_indicator()
df["RSI"]=RSIIndicator(close).rsi()

macd=MACD(close)
df["MACD"]=macd.macd()
df["MACD_signal"]=macd.macd_signal()

bb=BollingerBands(close)
df["BB_up"]=bb.bollinger_hband()
df["BB_low"]=bb.bollinger_lband()

last=df.iloc[-1]

# ===== AI SIGNAL =====
score=0
if last["RSI"]<30: score+=2
if last["RSI"]>70: score-=2
if last["MACD"]>last["MACD_signal"]: score+=2
else: score-=2
if last["EMA20"]>last["SMA50"]: score+=1
else: score-=1

if score>=3:
    signal="STRONG BUY"
elif score<=-3:
    signal="STRONG SELL"
elif score>0:
    signal="BUY"
elif score<0:
    signal="SELL"
else:
    signal="WAIT"

# ===== SUPPORT / RESIST =====
support=df["Low"].tail(50).min()
resistance=df["High"].tail(50).max()

# ===== RR + LOT =====
atr=(df["High"]-df["Low"]).rolling(14).mean().iloc[-1]
sl=last["Close"]-atr*1.2
tp=last["Close"]+atr*2.5
rr=abs(tp-last["Close"])/abs(last["Close"]-sl)

lot=(capital*(risk/100))/(abs(last["Close"]-sl)*100)
lot=round(max(lot,0.01),2)

# ===== TOP PANEL =====
c1,c2,c3,c4,c5=st.columns(5)
c1.metric("Price",round(last["Close"],2))
c2.metric("RSI",round(last["RSI"],1))
c3.metric("AI Signal",signal)
c4.metric("Risk/Reward",f"1:{round(rr,2)}")
c5.metric("Lot Size",lot)

st.success(f"AI Signal → {signal}")

# ===== CHART =====
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.6,0.2,0.2]
)

# Candle
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Price"
),row=1,col=1)

# EMA SMA
fig.add_trace(go.Scatter(x=df.index,y=df["EMA20"],name="EMA20"),row=1,col=1)
fig.add_trace(go.Scatter(x=df.index,y=df["SMA50"],name="SMA50"),row=1,col=1)

# Bollinger
fig.add_trace(go.Scatter(x=df.index,y=df["BB_up"],line=dict(width=1)),row=1,col=1)
fig.add_trace(go.Scatter(x=df.index,y=df["BB_low"],fill='tonexty',line=dict(width=1)),row=1,col=1)

# Support Resistance
fig.add_hline(y=support,line_dash="dot",line_color="green")
fig.add_hline(y=resistance,line_dash="dot",line_color="red")

# MACD
fig.add_trace(go.Scatter(x=df.index,y=df["MACD"],name="MACD"),row=2,col=1)
fig.add_trace(go.Scatter(x=df.index,y=df["MACD_signal"],name="Signal"),row=2,col=1)

# RSI
fig.add_trace(go.Scatter(x=df.index,y=df["RSI"],name="RSI"),row=3,col=1)
fig.add_hline(y=70,row=3,col=1,line_dash="dash")
fig.add_hline(y=30,row=3,col=1,line_dash="dash")

fig.update_layout(
height=850,
template="plotly_dark",
xaxis_rangeslider_visible=False,
title=f"{asset} AI Trading Terminal"
)

st.plotly_chart(fig,use_container_width=True)
