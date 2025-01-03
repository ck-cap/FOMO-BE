import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
# from app.utils.preprocessing import preprocess as p

# Load the LSTM model
model = load_model('lstm_stock_model.keras')

@staticmethod
class prediction:
    
    def preprocess(past_data = 10, forecast_range = 60, tickers=['META', 'AAPL', 'GOOG','MSFT', 'NVDA']):
        
        # tickers = ['META', 'AAPL', 'GOOG','MSFT', 'NVDA']
        # past_data = 10 # past data range in years
        # forecast_range = 60
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
        return data_df, forecast_range, scaled_data, scaler
    
    @staticmethod
    def predict_stock_price(scaled_data, scaler, forecast_range):
        predictions = []
        current_input = scaled_data[-forecast_range:].reshape(1, forecast_range, scaled_data.shape[1])
        for _ in range(forecast_range): # forecast the next N days/time
            prediction = model.predict(current_input) # history data considered
            predictions.append(prediction[0])
            current_input = np.append(current_input[:, 1:, :], [prediction], axis=1) # Updating the rolling window
        predictions = np.array(predictions).reshape(forecast_range, -1)     # Inverse back transformed predictions
        
        return scaler.inverse_transform(predictions)

    @staticmethod
    # Plot the forecasted prices
    def plot_forecast(predictions, data_df, forecast_range, selected_tickers='AAPL'):
        prediction_dates = pd.date_range(
        start=data_df.index[-1] + pd.Timedelta(days=1),
        periods=forecast_range,
        freq='B'  # Business day frequency
        )
        forecast_df = pd.DataFrame(predictions, index=prediction_dates, columns=data_df.columns)

        # color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        # pre_color = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
        fig = go.Figure()
        # for ticker_color, pre_color in zip(selected_tickers, color, pre_color):
        fig.add_trace(go.Scatter(
            x=data_df.index, 
            y=data_df[selected_tickers], 
            mode='lines', 
            name=f'{selected_tickers} Prices',
            line=dict(color='#1f77b4')
        ))

        fig.add_trace(go.Scatter(
            x=forecast_df.index, 
            y=forecast_df[selected_tickers], 
            mode='lines', 
            name=f'{selected_tickers} Predictions',
            line=dict(color="#aec7e8")
        ))
    
        fig.update_layout(
            title=f"Price Forecast for {selected_tickers} Tickers",
            xaxis_title="Date",
            yaxis_title="Closing Price USD ($)",
            legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0)', bordercolor='rgba(0,0,0,0)'),
            template='plotly_white',
            hovermode='x unified'
        )
    
        fig.show()


    @staticmethod
    def plot_forecast_html(predictions, data_df, forecast_range, selected_tickers='AAPL', div_id="plotly-graph"):
        prediction_dates = pd.date_range(
            start=data_df.index[-1] + pd.Timedelta(days=1),
            periods=forecast_range,
            freq='B'  # Business day frequency
        )
        forecast_df = pd.DataFrame(predictions, index=prediction_dates, columns=data_df.columns)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data_df.index, 
            y=data_df[selected_tickers], 
            mode='lines', 
            name=f'{selected_tickers} Prices',
            line=dict(color='#1f77b4')
        ))

        fig.add_trace(go.Scatter(
            x=forecast_df.index, 
            y=forecast_df[selected_tickers], 
            mode='lines', 
            name=f'{selected_tickers} Predictions',
            line=dict(color="#aec7e8")
        ))

        fig.update_layout(
            title=f"Price Forecast for {selected_tickers} Tickers",
            xaxis_title="Date",
            yaxis_title="Closing Price USD ($)",
            legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0)', bordercolor='rgba(0,0,0,0)'),
            template='plotly_white',
            hovermode='x unified'
        )
        return pio.to_html(fig, full_html=False, include_plotlyjs=False, div_id=div_id)
    

    # data_df, forecast_range, scaled_data, scaler = preprocess()
    # predictions = predict_stock_price(model, scaled_data, scaler, forecast_range)    


    # # Prepare results for visualization

    
    # # tickers = ['META', 'AAPL', 'GOOG','MSFT', 'NVDA'] # temp
    # plot_forecast(data_df, forecast_range, forecast_df,)

 