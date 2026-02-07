import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange
import time

st.set_page_config(page_title="Multi Asset AI Trading Bot", layout="wide")

# ===== SYMBOL LIST =====
SYMBOLS = {
    "Gold (XAUUSD)": "GC=F",
    "Silver": "SI=F",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "Crude Oil": "CL=F",
    "Nasdaq": "^IXIC"
}

MIN_RR = 2

st.title("💀 Multi Currency AI Trading Bot")
st.write("Live AI Signals using Yahoo Finance")

# ===== SIDEBAR =====
symbol_name = st.sidebar.selectbox("Select Trading Asset", list(SYMBOLS.keys()))
SYMBOL = SYMBOLS[symbol_name]

capital = st.sidebar.number_input("Capital ($)", value=100)
risk_percent = st.sidebar.slider("Risk % per Trade", 1, 10, 2)

st.sidebar.markdown("---")
st.sidebar.write("Selected Symbol:")
st.sidebar.success(symbol_name)

# ===== DATA =====
def get_data():
    df = yf.download(SYMBOL, period="1d", interval="5m", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['rsi'] = RSIIndicator(df['Close']).rsi()
    df['ema20'] = EMAIndicator(df['Close'], 20).ema_indicator()
    df['ema50'] = EMAIndicator(df['Close'], 50).ema_indicator()

    macd = MACD(df['Close'])
    df['macd'] = macd.macd_diff()

    df['atr'] = AverageTrueRange(
        df['High'], df['Low'], df['Close']
    ).average_true_range()

    df.dropna(inplace=True)
    return df

# ===== SIGNAL LOGIC =====
def signal(df):
    last = df.iloc[-1]

    price = float(last['Close'])
    atr = float(last['atr'])

    sig = "HOLD"
    sl = tp = rr = 0

    # BUY
    if price > last['ema20'] and last['rsi'] > 50 and last['macd'] > 0:
        sl = price - atr * 1.2
        tp = price + atr * 2.5
        rr = abs(tp - price) / abs(price - sl)
        if rr >= MIN_RR:
            sig = "BUY"

    # SELL
    if price < last['ema20'] and last['rsi'] < 50 and last['macd'] < 0:
        sl = price + atr * 1.2
        tp = price - atr * 2.5
        rr = abs(price - tp) / abs(price - sl)
        if rr >= MIN_RR:
            sig = "SELL"

    lot = (capital * (risk_percent / 100)) / (abs(price - sl) * 100) if sig != "HOLD" else 0
    lot = round(max(lot, 0.01), 2)

    return sig, price, sl, tp, lot, rr

# ===== LIVE DISPLAY =====
placeholder = st.empty()

while True:
    df = get_data()
    sig, price, sl, tp, lot, rr = signal(df)

    with placeholder.container():
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Price", round(price, 2))
        col2.metric("Signal", sig)
        col3.metric("Risk/Reward", f"1:{round(rr, 2)}")
        col4.metric("Lot Size", lot)

        if sig != "HOLD":
            st.success(f" TRADE SIGNAL: {sig}")
            st.write(f" Take Profit: {round(tp, 2)}")
            st.write(f"Stop Loss: {round(sl, 2)}")
        else:
            st.warning("No Trade Zone")

        st.line_chart(df['Close'])

    time.sleep(5)
