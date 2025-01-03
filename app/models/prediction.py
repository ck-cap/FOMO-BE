import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load the LSTM model
model = load_model('lstm_stock_model.keras')


"""
    past_data = years pulled from yahoo finance
    forecast_range = how many days to forecast (one year around 250 days)
"""
tickers = ['META', 'AAPL', 'GOOG','MSFT', 'NVDA']
past_data = 10 # past data range in years
forecast_range = 60
end = datetime.now()
start = datetime(end.year - past_data, end.month, end.day)

data = {}
for ticker in tickers:
    try:
        data[ticker] = yf.download(ticker, start, end)['Close']
    except Exception as e:
        print(f"Failed to download data for {ticker}: {e}")
        data[ticker] = pd.Series()  # Assign an empty series if download fails


data_df = pd.concat(data.values(), axis=1, keys=data.keys())
data_df.columns = tickers
data_df.ffill(inplace=True)
data_df.dropna(inplace=True)
dataset = data_df.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)



def forecast_next_days(model, scaled_data, scaler, forecast_range):
    predictions = []
    current_input = scaled_data[-forecast_range:].reshape(1, forecast_range, scaled_data.shape[1])
    for _ in range(forecast_range): # forecast the next N days/time
        prediction = model.predict(current_input) # history data considered
        predictions.append(prediction[0])
        current_input = np.append(current_input[:, 1:, :], [prediction], axis=1) # Updating the rolling window
    predictions = np.array(predictions).reshape(forecast_range, -1)     # Inverse back transformed predictions
    return scaler.inverse_transform(predictions)


predictions = forecast_next_days(model, scaled_data, scaler, forecast_range)


# Prepare results for visualization
prediction_dates = pd.date_range(
    start=data_df.index[-1] + pd.Timedelta(days=1),
    periods=forecast_range,
    freq='B'  # Business day frequency
)
forecast_df = pd.DataFrame(predictions, index=prediction_dates, columns=data_df.columns)


# Plot the forecasted prices
color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
pre_color = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
fig = go.Figure()
for ticker, ticker_color, pre_color in zip(tickers, color, pre_color):
    fig.add_trace(go.Scatter(
        x=data_df.index, 
        y=data_df[ticker], 
        mode='lines', 
        name=f'{ticker} Prices',
        line=dict(color=ticker_color)
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df.index, 
        y=forecast_df[ticker], 
        mode='lines', 
        name=f'{ticker} Predictions',
        line=dict(color=pre_color)
    ))

fig.update_layout(
    title="Price Forecast for Multiple Tickers",
    xaxis_title="Date",
    yaxis_title="Closing Price USD ($)",
    legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0)', bordercolor='rgba(0,0,0,0)'),
    template='plotly_white',
    hovermode='x unified'
)

fig.show()

    