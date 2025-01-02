import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the LSTM model
model = load_model("app/models/lstm_stock_model.keras")

class Prediction:
    # Function to preprocess the data for LSTM model
    @staticmethod
    def preprocess_data(data: np.ndarray, lookback: int = 60):
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)  # Apply scaling to the whole dataset (all tickers)

        # Prepare the input data for the LSTM
        X = []
        for i in range(lookback, len(data)):
            X.append(data_scaled[i - lookback:i, :])  # Include all tickers in the window
        X = np.array(X)
        
        # Reshape for LSTM input (samples, time steps, features)
        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
        
        return X, scaler

    # Function to predict stock prices
    @staticmethod
    def predict_stock_price(ticker: str, stock_data: pd.Series, forecast_days: int = 30):
        lookback = 60  # LSTM uses a lookback period of 60 days
        
        # Create a DataFrame with the correct columns: ['META', 'AAPL', 'GOOG', 'MSFT', 'NVDA']
        df = pd.DataFrame({
            'META': [np.nan] * len(stock_data),
            'AAPL': [np.nan] * len(stock_data),
            'GOOG': [np.nan] * len(stock_data),
            'MSFT': [np.nan] * len(stock_data),
            'NVDA': [np.nan] * len(stock_data)
        })

        df.index = stock_data.index
        df[ticker] = stock_data.values

        df[ticker] = df[ticker].apply(lambda x: None if (pd.isna(x) or np.isinf(x)) else x)
        
        # Preprocess the data for prediction
        X, scaler = Prediction.preprocess_data(df.values, lookback)

        # Make prediction for the next day (forecast)
        predictions = model.predict(X[-1:].reshape(1, lookback, 5))  # Use the last lookback period for prediction
        
        # Invert scaling to get actual prices
        predicted_prices = scaler.inverse_transform(predictions)

        # Debugging: Show predicted prices
        print(f"Predicted Prices for {ticker}: {predicted_prices}")

        # Clean up the predicted prices for JSON compatibility (remove NaN/Infinity)
        predicted_prices_clean = np.nan_to_num(predicted_prices)  # Convert NaNs to 0 and infinities to large numbers

        # Return the predicted prices (forecast_days will need to be handled separately if it's more than 1)
        return predicted_prices_clean[0].tolist()