"""Trends Routes - Real Database Data Only"""
from fastapi import FastAPI
from datetime import datetime
from config.symbol_manager import symbol_manager
from modules.data_validator import data_validator

def setup_trends_routes(app: FastAPI, model, database):
    
    @app.get("/api/asset/{symbol}/trends")
    async def asset_trends(symbol: str, timeframe: str = "1D"):
        print(f"🔍 TRENDS API: symbol={symbol}, timeframe={timeframe}")
        
        if not database or not database.pool:
            return {'error': 'Database not available'}
        
        # Normalize timeframe to uppercase (1h -> 1H, 4h -> 4H)
        timeframe_normalized = timeframe.upper()
        
        # Check if macro indicator - they don't support timeframes
        macro_symbols = ['GDP', 'CPI', 'UNEMPLOYMENT', 'FED_RATE', 'CONSUMER_CONFIDENCE']
        is_macro = symbol in macro_symbols
        
        # Macro indicators: always use 1D, ignore timeframe parameter
        if is_macro:
            db_timeframe = "1D"
            print(f"📊 MACRO: Using db_timeframe=1D (ignored input timeframe={timeframe})")
        else:
            # Crypto/Stock: use normalized timeframe
            timeframe_mapping = {'7D': '1W', '1Y': '1W', '5Y': '1M'}
            db_timeframe = timeframe_mapping.get(timeframe_normalized, timeframe_normalized)
            print(f"📈 CRYPTO/STOCK: Using db_timeframe={db_timeframe} (input timeframe={timeframe})")
        
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
            
            # Query actual prices
            actual_rows = await conn.fetch("""
                SELECT price, timestamp
                FROM actual_prices
                WHERE symbol = $1
                ORDER BY timestamp ASC
                LIMIT 50
            """, db_symbol)
            
            # Query forecasts separately
            forecast_rows = await conn.fetch("""
                SELECT predicted_price, created_at
                FROM forecasts
                WHERE symbol = $1
                ORDER BY created_at ASC
                LIMIT 50
            """, db_symbol)
            
            # Build lookup for forecasts by date
            forecast_map = {}
            for f in forecast_rows:
                date_key = f['created_at'].date()
                forecast_map[date_key] = float(f['predicted_price'])
            
            # Combine actual and predicted
            rows = []
            for a in actual_rows:
                date_key = a['timestamp'].date()
                rows.append({
                    'actual_price': a['price'],
                    'timestamp': a['timestamp'],
                    'predicted_price': forecast_map.get(date_key)
                })
            
            if not rows:
                print(f"❌ No historical data found for {db_symbol}")
                return {'error': 'No historical data available'}
            
            print(f"✅ Found {len(rows)} historical records for {db_symbol}")
            
            for record in rows:
                price = float(record['actual_price'])
                if data_validator.validate_price(symbol, price):
                    actual_prices.append(price)
                    timestamps.append(record['timestamp'].isoformat())
                    
                    if record['predicted_price']:
                        predicted_prices.append(float(record['predicted_price']))
                    else:
                        predicted_prices.append(None)
        
        # Build accuracy history and filter valid pairs
        accuracy_history = []
        valid_actual = []
        valid_predicted = []
        valid_timestamps = []
        
        for i in range(len(actual_prices)):
            actual = actual_prices[i]
            predicted = predicted_prices[i] if i < len(predicted_prices) else None
            
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
            
            valid_actual.append(actual)
            valid_predicted.append(predicted)
            valid_timestamps.append(timestamps[i])
        
        # Check if we have any valid pairs
        if not valid_actual or not valid_predicted:
            print(f"❌ No valid prediction pairs found for {symbol}:{db_timeframe}")
            return {'error': 'No predictions available for this timeframe'}
        
        # Validate only the valid pairs
        validation = data_validator.validate_accuracy_data(valid_actual, valid_predicted, symbol, db_timeframe)
        
        if not validation['valid']:
            print(f"❌ Validation failed: {validation.get('error', 'Unknown error')}")
            return {'error': validation.get('error', 'Validation failed')}
        
        # Calculate accuracy as Hit rate (predictions within 5% error)
        hits = sum(1 for item in accuracy_history if item['result'] == 'Hit')
        total = len(accuracy_history)
        accuracy_pct = (hits / total * 100) if total > 0 else 0
        mean_error = validation['mean_error_pct'] if validation['valid'] else 0
        
        print(f"📊 ACCURACY: hits={hits}, total={total}, accuracy={accuracy_pct:.1f}%, mean_error={mean_error:.1f}%")
        print(f"📋 Valid pairs: actual={len(valid_actual)}, predicted={len(valid_predicted)}")
        
        # Build response based on asset type
        print(f"🎯 Building response: is_macro={is_macro}, timeframe={timeframe}, db_timeframe={db_timeframe}")
        
        if is_macro:
            macro_frequencies = {
                'GDP': 'Quarterly',
                'CPI': 'Monthly', 
                'UNEMPLOYMENT': 'Monthly',
                'FED_RATE': 'Every 6 weeks',
                'CONSUMER_CONFIDENCE': 'Monthly'
            }
            response = {
                'symbol': symbol,
                'change_frequency': macro_frequencies.get(symbol, 'Monthly'),
                'overall_accuracy': round(accuracy_pct, 1),
                'mean_error_pct': round(mean_error, 1),
                'chart': {
                    'actual': valid_actual,
                    'predicted': valid_predicted,
                    'timestamps': valid_timestamps
                },
                'accuracy_history': accuracy_history,
                'validation': validation
            }
        else:
            response = {
                'symbol': symbol,
                'timeframe': timeframe_normalized,  # Return normalized timeframe
                'overall_accuracy': round(accuracy_pct, 1),
                'mean_error_pct': round(mean_error, 1),
                'chart': {
                    'actual': valid_actual,
                    'predicted': valid_predicted,
                    'timestamps': valid_timestamps
                },
                'accuracy_history': accuracy_history,
                'validation': validation
            }
        
        return response
