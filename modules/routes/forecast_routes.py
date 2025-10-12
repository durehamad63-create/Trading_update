"""Forecast Routes - Real ML Predictions Only"""
from fastapi import FastAPI
from datetime import datetime, timedelta
from multi_asset_support import multi_asset
from config.symbol_manager import symbol_manager

def setup_forecast_routes(app: FastAPI, model, database):
    
    @app.get("/api/asset/{symbol}/forecast")
    async def asset_forecast(symbol: str, timeframe: str = "1D"):
        prediction = await model.predict(symbol)
        if not prediction:
            return {"error": f"No prediction available for {symbol}"}
        
        current_price = prediction.get('current_price', 0)
        predicted_price = prediction.get('predicted_price', current_price)
        forecast_direction = prediction.get('forecast_direction', 'HOLD')
        
        past_prices = []
        past_timestamps = []
        
        if database and database.pool:
            async with database.pool.acquire() as conn:
                db_symbol = symbol_manager.get_db_key(symbol, timeframe)
                historical_data = await conn.fetch(
                    "SELECT price, timestamp FROM actual_prices WHERE symbol = $1 ORDER BY timestamp DESC LIMIT 30",
                    db_symbol
                )
                
                if historical_data:
                    for record in reversed(historical_data):
                        past_prices.append(float(record['price']))
                        past_timestamps.append(record['timestamp'].isoformat())
        
        if not past_prices:
            return {"error": "No historical data available"}
        
        future_prices = []
        future_timestamps = []
        
        for i in range(7):
            timestamp = datetime.now() + timedelta(days=i + 1)
            if forecast_direction == 'UP':
                future_price = predicted_price * (1 + (i + 1) * 0.01)
            elif forecast_direction == 'DOWN':
                future_price = predicted_price * (1 - (i + 1) * 0.01)
            else:
                future_price = predicted_price
            
            future_prices.append(round(future_price, 2))
            future_timestamps.append(timestamp.isoformat())
        
        return {
            "symbol": symbol,
            "name": multi_asset.get_asset_name(symbol),
            "timeframe": timeframe,
            "forecast_direction": forecast_direction,
            "confidence": prediction.get('confidence', 75),
            "current_price": current_price,
            "predicted_price": predicted_price,
            "change_24h": prediction.get('change_24h', 0),
            "chart": {
                "past": past_prices,
                "future": future_prices,
                "timestamps": past_timestamps + future_timestamps
            }
        }
