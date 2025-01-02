from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from typing import Dict
from pydantic import BaseModel
from app.models.prediction import Prediction
import pandas as pd

app = FastAPI(title="Stock Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class StockRequest(BaseModel):
    symbol: str
    period: str = "1y"  # default to 1 year of data


@app.get("/")
async def root():
    return {"message": "Welcome to Stock Prediction API"}


@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/api/v1/stock/{symbol}")
async def get_stock_data(symbol: str):
    try:
        # Fetch stock data
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")

        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")

        return {
            "symbol": symbol,
            "prices": hist['Close'].tolist(),
            "dates": hist.index.strftime('%Y-%m-%d').tolist(),
            "volumes": hist['Volume'].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/predict")
async def predict_stock(symbol: str, lookback_years: int = 1, forecast_days: int = 30):
    try:
        # Ensure the lookback years is within the maximum limit of 10
        if lookback_years > 10:
            raise HTTPException(status_code=400, detail="Lookback years cannot exceed 10.")
        
        # Fetch historical data for the ticker
        stock_data = yf.Ticker(symbol).history(period=f"{lookback_years}y")
        if stock_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        # Only use the 'Close' price for prediction
        stock_data = stock_data['Close']

        # Use the Prediction class to predict stock prices
        predicted_prices = Prediction.predict_stock_price(symbol, stock_data, forecast_days)

        return {
            "ticker": symbol,
            "predicted_prices": predicted_prices,
            "forecast_days": forecast_days
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)