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
            # Timeframe mapping: query stored predictions from finer timeframe models
            timeframe_mapping = {
                '1D': {'model_tf': '1h', 'steps': 12},  # 12 hourly predictions
                '1W': {'model_tf': '1D', 'steps': 4},   # 4 daily predictions
                '1M': {'model_tf': '1W', 'steps': 7},   # 7 weekly predictions
                '1h': {'model_tf': '1h', 'steps': 12},  # 12 hourly predictions
                '4h': {'model_tf': '4h', 'steps': 6}    # 6 4-hour predictions
            }
            
            mapping = timeframe_mapping.get(actual_timeframe, {'model_tf': actual_timeframe, 'steps': 1})
            model_timeframe = mapping['model_tf']
            num_steps = mapping['steps']
            
            # Query stored predictions from database for finer timeframe
            future_prices = []
            future_timestamps = []
            
            if database and database.pool and num_steps > 1:
                try:
                    async with database.pool.acquire() as conn:
                        db_key = symbol_manager.get_db_key(symbol, model_timeframe)
                        
                        # Get last N stored predictions from finer timeframe
                        # created_at is the future timestamp being predicted (from gap filling)
                        rows = await conn.fetch(
                            "SELECT predicted_price, created_at FROM forecasts WHERE symbol = $1 AND created_at >= NOW() - INTERVAL '24 hours' ORDER BY created_at DESC LIMIT $2",
                            db_key, num_steps
                        )
                        
                        if rows and len(rows) >= num_steps:
                            # Use stored predictions in chronological order
                            for row in reversed(rows):
                                future_prices.append(float(row['predicted_price']))
                                future_timestamps.append(row['created_at'].isoformat())
                        else:
                            # Fallback: not enough stored predictions
                            future_prices = [predicted_price]
                            future_timestamps = [datetime.now().isoformat()]
                except Exception as e:
                    future_prices = [predicted_price]
                    future_timestamps = [datetime.now().isoformat()]
            else:
                # Single step prediction
                future_prices = [predicted_price]
                future_timestamps = [datetime.now().isoformat()]
        
        # Get forecast direction from prediction
        if not is_macro:
            forecast_direction = prediction.get('forecast_direction', 'HOLD')
        
        response = {
            "symbol": symbol,
            "name": multi_asset.get_asset_name(symbol),
            "timeframe": actual_timeframe,
            "forecast_direction": forecast_direction,
            "confidence": prediction.get('confidence', 75),
            "current_price": current_price,
            "predicted_price": future_prices[-1] if future_prices else predicted_price,
            "change_24h": prediction.get('change_24h', 0),
            "chart": {
                "past": past_prices,
                "future": future_prices,
                "timestamps": past_timestamps + future_timestamps
            }
        }
        
        # Add metadata
        if is_macro:
            macro_frequencies = {
                'GDP': 'Quarterly',
                'CPI': 'Monthly', 
                'UNEMPLOYMENT': 'Monthly',
                'FED_RATE': 'Every 6 weeks (FOMC meetings)',
                'CONSUMER_CONFIDENCE': 'Monthly'
            }
            response['change_frequency'] = macro_frequencies.get(symbol, 'Monthly')
        else:
            # Add model metadata
            response['model_timeframe'] = model_timeframe
            response['prediction_steps'] = len(future_prices)
        
        return response
