from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from prophet import Prophet

st.set_page_config(page_title="Stock Intelligence", page_icon="📈", layout="wide")
st.title("📈 Stock Intelligence Dashboard")
st.caption("Historical market data and statistical price forecasting")

STOCKS = {
    "Apple": "AAPL",
    "Alphabet (Google)": "GOOG",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "GameStop": "GME",
}

st.sidebar.header("Settings")
company = st.sidebar.selectbox("Choose a stock", list(STOCKS.keys()))
ticker = STOCKS[company]
years = st.sidebar.slider("Historical data (years)", 1, 10, 5)
forecast_days = st.sidebar.slider("Forecast horizon (business days)", 30, 252, 90, step=30)
show_averages = st.sidebar.checkbox("Show moving averages", value=True)
start_date = (date.today() - timedelta(days=365 * years)).isoformat()


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(symbol: str, start: str) -> pd.DataFrame:
    """Download and clean adjusted daily closing prices."""
    raw = yf.download(symbol, start=start, auto_adjust=True, progress=False, threads=False)
    if raw.empty:
        return pd.DataFrame(columns=["Date", "Close"])
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if "Close" not in raw.columns:
        return pd.DataFrame(columns=["Date", "Close"])
    data = raw.reset_index()
    if "Date" not in data.columns:
        data = data.rename(columns={data.columns[0]: "Date"})
    data = data[["Date", "Close"]].copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna(subset=["Date", "Close"])
    data = data.sort_values("Date").drop_duplicates("Date")
    return data.reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def make_forecast(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Fit Prophet and return historical plus future predictions."""
    training = history.rename(columns={"Date": "ds", "Close": "y"}).copy()
    training["ds"] = pd.to_datetime(training["ds"])
    training["y"] = pd.to_numeric(training["y"], errors="coerce")
    training = training.dropna().sort_values("ds")
    if len(training) < 60:
        raise ValueError("At least 60 valid daily prices are required.")
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                    daily_seasonality=False, interval_width=0.95)
    model.fit(training)
    future = model.make_future_dataframe(periods=horizon, freq="B")
    return model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]]


try:
    with st.spinner(f"Downloading {ticker} data..."):
        data = load_data(ticker, start_date)
except Exception as exc:
    st.error(f"Could not download stock data: {exc}")
    st.stop()

if data.empty:
    st.error("No data was returned. Check your internet connection or choose another stock.")
    st.stop()

data["MA20"] = data["Close"].rolling(20).mean()
data["MA50"] = data["Close"].rolling(50).mean()
latest = float(data["Close"].iloc[-1])
previous = float(data["Close"].iloc[-2]) if len(data) > 1 else latest
change_pct = ((latest - previous) / previous * 100) if previous else 0.0

st.subheader(f"{company} ({ticker})")
c1, c2, c3 = st.columns(3)
c1.metric("Latest adjusted close", f"{latest:,.2f}", f"{change_pct:+.2f}%")
c2.metric("20-day average", f"{data['MA20'].iloc[-1]:,.2f}" if pd.notna(data["MA20"].iloc[-1]) else "N/A")
c3.metric("50-day average", f"{data['MA50'].iloc[-1]:,.2f}" if pd.notna(data["MA50"].iloc[-1]) else "N/A")

st.subheader("Historical prices")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Adjusted close"))
if show_averages:
    fig.add_trace(go.Scatter(x=data["Date"], y=data["MA20"], name="20-day average"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["MA50"], name="50-day average"))
fig.update_layout(template="plotly_white", height=480, hovermode="x unified",
                  xaxis_title="Date", yaxis_title="Adjusted price",
                  xaxis_rangeslider_visible=True)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Forecast")
if st.button("Generate forecast", type="primary"):
    try:
        with st.spinner("Training Prophet model..."):
            forecast = make_forecast(data[["Date", "Close"]], forecast_days)
        future = forecast[forecast["ds"] > data["Date"].max()].copy()
        if future.empty:
            st.warning("No future forecast dates were generated.")
        else:
            forecast_fig = go.Figure()
            forecast_fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Historical close"))
            forecast_fig.add_trace(go.Scatter(x=future["ds"], y=future["yhat_upper"],
                                              mode="lines", line=dict(width=0),
                                              showlegend=False, hoverinfo="skip"))
            forecast_fig.add_trace(go.Scatter(x=future["ds"], y=future["yhat_lower"],
                                              mode="lines", line=dict(width=0),
                                              fill="tonexty", name="95% model interval"))
            forecast_fig.add_trace(go.Scatter(x=future["ds"], y=future["yhat"],
                                              name="Forecast", line=dict(dash="dash", width=2)))
            forecast_fig.update_layout(template="plotly_white", height=480,
                                       hovermode="x unified", xaxis_title="Date",
                                       yaxis_title="Adjusted price")
            st.plotly_chart(forecast_fig, use_container_width=True)
            last = future.iloc[-1]
            a, b = st.columns(2)
            a.metric("Final forecast estimate", f"{last['yhat']:,.2f}")
            b.metric("Final model interval", f"{last['yhat_lower']:,.2f} – {last['yhat_upper']:,.2f}")
            st.download_button("Download forecast CSV",
                               data=future.to_csv(index=False).encode("utf-8"),
                               file_name=f"{ticker}_forecast.csv", mime="text/csv")
            with st.expander("View forecast data"):
                st.dataframe(future, use_container_width=True)
    except Exception as exc:
        st.error(f"Forecast failed: {exc}")

with st.expander("View historical data"):
    st.dataframe(data.tail(100), use_container_width=True)
    st.download_button("Download historical CSV",
                       data=data.to_csv(index=False).encode("utf-8"),
                       file_name=f"{ticker}_historical.csv", mime="text/csv")

st.divider()
st.caption("Educational tool only. Forecasts are uncertain statistical estimates, not investment advice or guaranteed future prices. Data is supplied by Yahoo Finance.")
