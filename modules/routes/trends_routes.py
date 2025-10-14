"""Trends Routes - Real Database Data Only"""
from fastapi import FastAPI
from datetime import datetime
from config.symbol_manager import symbol_manager
from modules.data_validator import data_validator

def setup_trends_routes(app: FastAPI, model, database):
    
    @app.get("/api/asset/{symbol}/trends")
    async def asset_trends(symbol: str, timeframe: str = "1D"):
        if not database or not database.pool:
            return {'error': 'Database not available'}
        
        # Check if macro indicator - they don't support timeframes
        macro_symbols = ['GDP', 'CPI', 'UNEMPLOYMENT', 'FED_RATE', 'CONSUMER_CONFIDENCE']
        is_macro = symbol in macro_symbols
        
        if is_macro and timeframe != "1D":
            return {"error": f"Macro indicator {symbol} does not support timeframe {timeframe}. Use default."}
        
        # Use 1D for macro indicators regardless of timeframe parameter
        if is_macro:
            db_timeframe = "1D"
        else:
            timeframe_mapping = {'7D': '1W', '1Y': '1W', '5Y': '1M'}
            db_timeframe = timeframe_mapping.get(timeframe, timeframe)
        
        try:
            prediction = await model.predict(symbol, db_timeframe)
            current_price = prediction.get('current_price', 0)
            
            if not data_validator.validate_price(symbol, current_price):
                return {'error': f'Invalid price for {symbol}'}
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}
        
        actual_prices = []
        predicted_prices = []
        timestamps = []
        
        async with database.pool.acquire() as conn:
            db_symbol = symbol_manager.get_db_key(symbol, db_timeframe)
            
            actual_data = await conn.fetch(
                "SELECT price, timestamp FROM actual_prices WHERE symbol = $1 ORDER BY timestamp DESC LIMIT 50",
                db_symbol
            )
            
            pred_data = await conn.fetch(
                "SELECT predicted_price, created_at FROM forecasts WHERE symbol = $1 AND predicted_price IS NOT NULL ORDER BY created_at DESC LIMIT 50",
                db_symbol
            )
            
            if not actual_data:
                return {'error': 'No historical data available'}
            
            for i, record in enumerate(reversed(actual_data)):
                price = float(record['price'])
                if data_validator.validate_price(symbol, price):
                    actual_prices.append(price)
                    timestamps.append(record['timestamp'].isoformat())
                    
                    if i < len(pred_data) and pred_data[i]['predicted_price']:
                        predicted_prices.append(float(pred_data[i]['predicted_price']))
                    else:
                        predicted_prices.append(None)
        
        validation = data_validator.validate_accuracy_data(actual_prices, predicted_prices, symbol, timeframe)
        
        if not validation['valid']:
            return {'error': validation.get('error', 'Validation failed')}
        
        accuracy_history = []
        for i in range(min(len(actual_prices), len(predicted_prices))):
            actual = actual_prices[i]
            predicted = predicted_prices[i]
            
            if predicted is None:
                continue
            
            error_pct = abs(actual - predicted) / actual * 100 if actual > 0 else 0
            result = 'Hit' if error_pct < 5 else 'Miss'
            
            accuracy_history.append({
                'date': timestamps[i][:10],
                'actual': round(actual, 2),
                'predicted': round(predicted, 2),
                'result': result,
                'error_pct': round(error_pct, 1)
            })
        
        predicted_prices = [p for p in predicted_prices if p is not None]
        
        response = {
            'symbol': symbol,
            'timeframe': db_timeframe,
            'overall_accuracy': round(validation['mean_error_pct'], 1),
            'chart': {
                'actual': actual_prices,
                'predicted': predicted_prices,
                'timestamps': timestamps
            },
            'accuracy_history': accuracy_history,
            'validation': validation
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
