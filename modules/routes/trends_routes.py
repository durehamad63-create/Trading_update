"""Trends Routes - Real Database Data Only"""
from fastapi import FastAPI
from datetime import datetime
from config.symbol_manager import symbol_manager
from modules.data_validator import data_validator
from utils.interval_formatter import interval_formatter

def setup_trends_routes(app: FastAPI, model, database):
    
    @app.get("/api/asset/{symbol}/trends")
    async def asset_trends(symbol: str, timeframe: str = "1D"):
        # Special handling for 'ALL' timeframe - combined accuracy
        if timeframe.upper() == 'ALL':
            return await get_combined_accuracy(symbol, database, model)
    
    async def get_combined_accuracy(symbol: str, database, model):
        """Calculate combined accuracy across all timeframes"""
        if not database or not database.pool:
            return {'error': 'Database not available'}
        
        macro_symbols = ['GDP', 'CPI', 'UNEMPLOYMENT', 'FED_RATE', 'CONSUMER_CONFIDENCE']
        is_macro = symbol in macro_symbols
        
        # Define timeframes to check
        if is_macro:
            timeframes = ['1D']
        else:
            timeframes = ['1H', '4H', '1D', '1W', '1M']
        
        all_actual = []
        all_predicted = []
        all_timestamps = []
        timeframe_stats = []
        
        async with database.pool.acquire() as conn:
            for tf in timeframes:
                db_symbol = symbol_manager.get_db_key(symbol, tf)
                
                rows = await conn.fetch("""
                    SELECT DISTINCT ON (DATE(ap.timestamp))
                        ap.price as actual_price,
                        ap.timestamp,
                        f.predicted_price
                    FROM actual_prices ap
                    LEFT JOIN forecasts f ON 
                        f.symbol = ap.symbol AND
                        DATE(f.created_at) = DATE(ap.timestamp)
                    WHERE ap.symbol = $1
                    ORDER BY DATE(ap.timestamp) DESC, ap.timestamp DESC
                    LIMIT 50
                """, db_symbol)
                
                if not rows:
                    continue
                
                tf_actual = []
                tf_predicted = []
                
                for record in reversed(rows):
                    price = float(record['actual_price'])
                    if data_validator.validate_price(symbol, price) and record['predicted_price']:
                        tf_actual.append(price)
                        tf_predicted.append(float(record['predicted_price']))
                
                if tf_actual and tf_predicted:
                    # Calculate accuracy for this timeframe
                    hits = sum(1 for i in range(len(tf_actual)) 
                              if abs(tf_actual[i] - tf_predicted[i]) / tf_actual[i] * 100 < 5)
                    tf_accuracy = (hits / len(tf_actual)) * 100
                    
                    timeframe_stats.append({
                        'timeframe': tf,
                        'accuracy': round(tf_accuracy, 1),
                        'total_predictions': len(tf_actual),
                        'hits': hits
                    })
                    
                    all_actual.extend(tf_actual)
                    all_predicted.extend(tf_predicted)
        
        if not all_actual or not all_predicted:
            return {'error': 'No predictions available'}
        
        # Calculate combined accuracy
        validation = data_validator.validate_accuracy_data(all_actual, all_predicted, symbol, 'ALL')
        
        if not validation['valid']:
            return {'error': validation.get('error', 'Validation failed')}
        
        total_hits = sum(1 for i in range(len(all_actual)) 
                        if abs(all_actual[i] - all_predicted[i]) / all_actual[i] * 100 < 5)
        combined_accuracy = (total_hits / len(all_actual)) * 100
        
        return {
            'symbol': symbol,
            'timeframe': 'ALL',
            'combined_accuracy': round(combined_accuracy, 1),
            'mean_error_pct': round(validation['mean_error_pct'], 1),
            'total_predictions': len(all_actual),
            'total_hits': total_hits,
            'timeframe_breakdown': timeframe_stats
        }
        
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
            
            # Query with JOIN - get one record per unique date/hour/period
            # Use DISTINCT ON to get latest record per time period
            if db_timeframe in ['1H', '4H']:
                # For hourly/4H: group by hour boundary
                rows = await conn.fetch("""
                    SELECT DISTINCT ON (DATE_TRUNC('hour', ap.timestamp))
                        ap.price as actual_price,
                        ap.timestamp,
                        f.predicted_price
                    FROM actual_prices ap
                    LEFT JOIN forecasts f ON 
                        f.symbol = ap.symbol AND
                        DATE_TRUNC('hour', f.created_at) = DATE_TRUNC('hour', ap.timestamp)
                    WHERE ap.symbol = $1
                    ORDER BY DATE_TRUNC('hour', ap.timestamp) DESC, ap.timestamp DESC
                    LIMIT 50
                """, db_symbol)
            else:
                # For daily/weekly/monthly: group by date
                rows = await conn.fetch("""
                    SELECT DISTINCT ON (DATE(ap.timestamp))
                        ap.price as actual_price,
                        ap.timestamp,
                        f.predicted_price
                    FROM actual_prices ap
                    LEFT JOIN forecasts f ON 
                        f.symbol = ap.symbol AND
                        DATE(f.created_at) = DATE(ap.timestamp)
                    WHERE ap.symbol = $1
                    ORDER BY DATE(ap.timestamp) DESC, ap.timestamp DESC
                    LIMIT 50
                """, db_symbol)
            
            print(f"📊 DEBUG: Found {len(rows)} unique time periods for {db_symbol}")
            
            if not rows:
                print(f"❌ No historical data found for {db_symbol}")
                return {'error': 'No historical data available'}
            
            # Reverse to chronological order and extract data
            for record in reversed(rows):
                price = float(record['actual_price'])
                if data_validator.validate_price(symbol, price):
                    actual_prices.append(price)
                    timestamps.append(record['timestamp'].isoformat())
                    
                    if record['predicted_price']:
                        predicted_prices.append(float(record['predicted_price']))
                    else:
                        predicted_prices.append(None)
            
            matched_count = sum(1 for p in predicted_prices if p is not None)
            print(f"📊 DEBUG: {matched_count} matched pairs out of {len(rows)} records")
        
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
            
            # Format timestamp based on timeframe interval
            formatted_time = interval_formatter.format_timestamp(timestamps[i], timeframe_normalized)
            
            accuracy_history.append({
                'date': formatted_time,
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
                'accuracy_history': accuracy_history
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
                'accuracy_history': accuracy_history
            }
        
        return response
