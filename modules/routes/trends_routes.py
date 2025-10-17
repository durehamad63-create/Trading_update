"""Trends Routes - Real Database Data Only"""
from fastapi import FastAPI
from datetime import datetime
from config.symbol_manager import symbol_manager
from modules.data_validator import data_validator
from utils.interval_formatter import interval_formatter

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
            
            # Check if stablecoin - hardcode values
            is_stablecoin = symbol in ['USDT', 'USDC']
            
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
                # Force stablecoin prices to $1.00
                if is_stablecoin:
                    price = 1.0
                    predicted = 1.0
                else:
                    price = float(record['actual_price'])
                    predicted = float(record['predicted_price']) if record['predicted_price'] else None
                
                if data_validator.validate_price(symbol, price):
                    actual_prices.append(price)
                    timestamps.append(record['timestamp'].isoformat())
                    predicted_prices.append(predicted)
            
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
            
            # Stablecoins always have 0% error
            if is_stablecoin:
                error_pct = 0.0
                result = 'Hit'
            else:
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
        
        # Calculate combined accuracy across all timeframes
        all_timeframes = ['1D'] if is_macro else ['1H', '4H', '1D', '1W', '1M']
        combined_hits = 0
        combined_total = 0
        
        # Stablecoins always have 100% accuracy
        if is_stablecoin:
            combined_hits = len(accuracy_history)
            combined_total = len(accuracy_history)
        else:
            async with database.pool.acquire() as conn:
                for tf in all_timeframes:
                    tf_db_symbol = symbol_manager.get_db_key(symbol, tf)
                    tf_rows = await conn.fetch("""
                        SELECT ap.price as actual_price, f.predicted_price
                        FROM actual_prices ap
                        LEFT JOIN forecasts f ON f.symbol = ap.symbol AND DATE(f.created_at) = DATE(ap.timestamp)
                        WHERE ap.symbol = $1 AND f.predicted_price IS NOT NULL
                        LIMIT 50
                    """, tf_db_symbol)
                    
                    for row in tf_rows:
                        actual = float(row['actual_price'])
                        predicted = float(row['predicted_price'])
                        error_pct = abs(actual - predicted) / actual * 100 if actual > 0 else 0
                        if error_pct < 5:
                            combined_hits += 1
                        combined_total += 1
        
        # Use combined accuracy if available, otherwise use current timeframe
        if combined_total > 0:
            accuracy_pct = (combined_hits / combined_total * 100)
            hits = combined_hits
            total = combined_total
        else:
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
