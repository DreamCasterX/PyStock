import streamlit as st
from datetime import date
import yfinance as yf
import numpy as np
np.float_ = np.float64 # fixed a bug in prophet
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


START = "2014-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


st.title("Stock Prediction App")


stock_symbols = {
    "台積電": "2330.TW",
    "GOOG": "GOOG",
    "MSFT": "MSFT",
    "AAPL": "AAPL",
    "AMZN": "AMZN",
    "NVDA": "NVDA",
}

stocks = sorted(list(stock_symbols.keys()))

selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365


@st.cache_data  # cache the data
def load_data(symbol):
    data = yf.download(symbol, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading data...")
data = load_data(stock_symbols[selected_stock])
data_load_state.text("Done!")


st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data()

# Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)

