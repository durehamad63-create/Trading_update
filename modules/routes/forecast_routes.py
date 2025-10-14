"""Forecast Routes - Real ML Predictions Only"""
from fastapi import FastAPI
from datetime import datetime, timedelta
from multi_asset_support import multi_asset
from config.symbol_manager import symbol_manager

def setup_forecast_routes(app: FastAPI, model, database):
    
    @app.get("/api/asset/{symbol}/forecast")
    async def asset_forecast(symbol: str, timeframe: str = "1D"):
        # Check if macro indicator - they don't support timeframes
        macro_symbols = ['GDP', 'CPI', 'UNEMPLOYMENT', 'FED_RATE', 'CONSUMER_CONFIDENCE']
        is_macro = symbol in macro_symbols
        
        if is_macro and timeframe != "1D":
            return {"error": f"Macro indicator {symbol} does not support timeframe {timeframe}. Use default."}
        
        # Use 1D for macro indicators regardless of timeframe parameter
        actual_timeframe = "1D" if is_macro else timeframe
        
        prediction = await model.predict(symbol, actual_timeframe)
        if not prediction:
            return {"error": f"No prediction available for {symbol}"}
        
        current_price = prediction.get('current_price', 0)
        predicted_price = prediction.get('predicted_price', current_price)
        forecast_direction = prediction.get('forecast_direction', 'HOLD')
        
        past_prices = []
        past_timestamps = []
        
        if database and database.pool:
            async with database.pool.acquire() as conn:
                db_symbol = symbol_manager.get_db_key(symbol, actual_timeframe)
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
        
        # For macro indicators, use their actual update frequency for future projections
        if is_macro:
            macro_frequencies = {
                'GDP': {'days': 90, 'name': 'Quarterly'},
                'CPI': {'days': 30, 'name': 'Monthly'}, 
                'UNEMPLOYMENT': {'days': 30, 'name': 'Monthly'},
                'FED_RATE': {'days': 42, 'name': 'Every 6 weeks (FOMC meetings)'},
                'CONSUMER_CONFIDENCE': {'days': 30, 'name': 'Monthly'}
            }
            
            freq_info = macro_frequencies.get(symbol, {'days': 30, 'name': 'Monthly'})
            future_prices = [predicted_price]  # Only next expected value
            future_timestamps = [(datetime.now() + timedelta(days=freq_info['days'])).isoformat()]
        else:
            # Use model's predicted price only (no synthetic future projections)
            future_prices = [predicted_price]
            future_timestamps = [datetime.now().isoformat()]
        
        response = {
            "symbol": symbol,
            "name": multi_asset.get_asset_name(symbol),
            "timeframe": actual_timeframe,
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
        
        # Add change_frequency for macro indicators
        if is_macro:
            macro_frequencies = {
                'GDP': 'Quarterly',
                'CPI': 'Monthly', 
                'UNEMPLOYMENT': 'Monthly',
                'FED_RATE': 'Every 6 weeks (FOMC meetings)',
                'CONSUMER_CONFIDENCE': 'Monthly'
            }
            response['change_frequency'] = macro_frequencies.get(symbol, 'Monthly')
        
        return response
