#!/usr/bin/env python3
"""
Monthly Data Collection & ML Prediction Service
Collects 1 month of historical data and generates ML predictions for all 25 assets
"""
import asyncio
import aiohttp
import logging
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from config.symbols import CRYPTO_SYMBOLS, STOCK_SYMBOLS, MACRO_SYMBOLS
from config.symbol_manager import symbol_manager
from utils.error_handler import ErrorHandler
from utils.startup_api_client import startup_api
from typing import List, Dict, Any

class GapFillingService:
    def __init__(self, model=None):
        self.model = model
        # All 25 assets from configuration
        self.crypto_symbols = list(CRYPTO_SYMBOLS.keys())  # 10 crypto
        self.stock_symbols = list(STOCK_SYMBOLS.keys())    # 10 stocks  
        self.macro_symbols = list(MACRO_SYMBOLS.keys())    # 5 macro
        self.all_symbols = self.crypto_symbols + self.stock_symbols + self.macro_symbols
        
        # Only native API intervals - no synthetic aggregation
        self.crypto_timeframes = ['1h', '4h', '1D', '1W', '1M']  # Binance native
        self.stock_timeframes = ['1h', '1D', '1W', '1M']  # Yahoo native (removed 4H, 7D)
        self.macro_timeframes = ['1D']  # FRED native - macro only supports daily (real release dates)
        
        # Maintain exactly 200 records per timeframe
        self.max_records = 200
        
        # Binance mapping for stablecoins
        self.binance_mapping = {'USDT': 'BTCUSDT', 'USDC': 'BTCUSDT'}
        self.binance_intervals = {
            '1h': '1h', '4h': '4h', '1D': '1d', '7D': '1w', '1W': '1w', '1M': '1M'
        }
        
        # Time intervals for proper data storage (avoid storing every second)
        self.storage_intervals = {
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4), 
            '1D': timedelta(days=1),
            '7D': timedelta(days=7),
            '1W': timedelta(weeks=1),
            '1M': timedelta(days=30)
        }
        
        print(f"🚀 Efficient Data Collector initialized for {len(self.all_symbols)} assets:")
        print(f"   📈 Crypto: {len(self.crypto_symbols)} symbols")
        print(f"   📊 Stocks: {len(self.stock_symbols)} symbols") 
        print(f"   🏛️ Macro: {len(self.macro_symbols)} symbols")
        print(f"   ⏱️ Crypto timeframes: {self.crypto_timeframes}")
        print(f"   ⏱️ Stock timeframes: {self.stock_timeframes}")
        print(f"   ⏱️ Macro timeframes: {self.macro_timeframes}")
        print(f"   📊 Max records per timeframe: {self.max_records}")
    
    async def fill_missing_data(self, db_instance):
        """Collect historical data efficiently with proper time intervals"""
        if not db_instance or not db_instance.pool:
            print("❌ Data collection: No database available")
            return
        
        self.db = db_instance
        print("🔍 Checking for missing data across all 25 assets")
        
        # Check if data already exists
        data_exists = await self._check_existing_data()
        if data_exists:
            print("✅ Historical data already exists - skipping collection")
            return
        
        print("🚀 Gap filling enabled - collecting missing historical data")
        print("ℹ️ Database cleanup disabled - preserving existing data")
        
        total_processed = 0
        total_accuracy = 0
        
        # Process all asset classes with correct timeframes
        for asset_class, symbols, timeframes in [
            ("crypto", self.crypto_symbols, self.crypto_timeframes),
            ("stocks", self.stock_symbols, self.stock_timeframes), 
            ("macro", self.macro_symbols, self.macro_timeframes)
        ]:
            print(f"📊 Processing {asset_class} assets: {len(symbols)} symbols")
            
            for i, symbol in enumerate(symbols):
                print(f"  🔄 {symbol} ({i+1}/{len(symbols)})...")
                
                for timeframe in timeframes:
                    try:
                        # Get historical data
                        data = await self._get_monthly_data(symbol, timeframe, asset_class)
                        
                        if not data or len(data) < 20:
                            print(f"    ❌ Insufficient data for {symbol} {timeframe}")
                            continue
                        
                        # Generate ML predictions
                        predictions = await self._generate_ml_predictions(data, symbol, timeframe, asset_class)
                        
                        if not predictions:
                            print(f"    ❌ No predictions for {symbol} {timeframe}")
                            continue
                        
                        # Calculate accuracy
                        results = self._calculate_accuracy(predictions)
                        
                        # Store data with proper intervals and maintain record limit
                        await self._store_historical_data_efficient(symbol, timeframe, data)
                        await self._store_predictions(symbol, timeframe, predictions, results)
                        
                        # Maintain record limit using centralized key
                        db_key = symbol_manager.get_db_key(symbol, timeframe)
                        await self._maintain_record_limit(db_key, timeframe)
                        
                        # Calculate accuracy stats
                        if results:
                            hits = sum(1 for r in results if r['result'] == 'Hit')
                            accuracy = (hits / len(results)) * 100
                            total_accuracy += accuracy
                            print(f"    ✅ {symbol} {timeframe}: {accuracy:.1f}% accuracy ({hits}/{len(results)} hits)")
                        
                        total_processed += 1
                        await asyncio.sleep(2.0)  # Longer delay to avoid rate limiting
                        
                    except Exception as e:
                        print(f"    ❌ Error processing {symbol} {timeframe}: {e}")
        
        if total_processed > 0:
            avg_accuracy = total_accuracy / total_processed
            print(f"✅ Gap filling completed: {total_processed} timeframes processed, {avg_accuracy:.1f}% average accuracy")
        else:
            print("⚠️ Gap filling completed with no data processed")
        
        # Close API client session
        try:
            await startup_api.close()
        except Exception:
            pass
    
    async def _check_existing_data(self):
        """Check if sufficient data already exists in database for all 25 symbols"""
        try:
            async with self.db.pool.acquire() as conn:
                # Check all 25 symbols across key timeframes
                all_symbols = self.crypto_symbols + self.stock_symbols + self.macro_symbols
                key_timeframes = ['1D', '1H']  # Check main timeframes
                
                symbols_with_data = 0
                total_checks = len(all_symbols) * len(key_timeframes)
                
                for symbol in all_symbols:
                    symbol_has_data = False
                    for timeframe in key_timeframes:
                        db_key = f"{symbol}_{timeframe}"
                        count = await conn.fetchval(
                            "SELECT COUNT(*) FROM actual_prices WHERE symbol = $1", db_key
                        )
                        if count >= 50:  # At least 50 records per timeframe
                            symbol_has_data = True
                            break
                    
                    if symbol_has_data:
                        symbols_with_data += 1
                
                coverage_pct = (symbols_with_data / len(all_symbols)) * 100
                print(f"📊 Data coverage: {symbols_with_data}/{len(all_symbols)} symbols ({coverage_pct:.1f}%)")
                
                # Require at least 80% coverage to skip collection
                if coverage_pct >= 80:
                    print("✅ Sufficient data coverage - skipping collection")
                    return True
                else:
                    print(f"ℹ️ Insufficient coverage ({coverage_pct:.1f}%) - will collect missing data")
                    return False
                    
        except Exception as e:
            print(f"⚠️ Error checking existing data: {e}")
            return False

    async def _clear_database(self):
        """Database cleanup disabled to preserve existing data"""
        print("ℹ️ Database cleanup disabled - preserving existing data")
        pass
    
    async def _maintain_record_limit(self, symbol_tf: str, timeframe: str):
        """Maintain exactly 1000 records per symbol-timeframe"""
        try:
            async with self.db.pool.acquire() as conn:
                # Keep only latest 200 records
                await conn.execute("""
                    DELETE FROM actual_prices 
                    WHERE symbol = $1 AND id NOT IN (
                        SELECT id FROM actual_prices 
                        WHERE symbol = $1 
                        ORDER BY timestamp DESC 
                        LIMIT 200
                    )
                """, symbol_tf)
        except Exception as e:
            print(f"      ⚠️ Record limit maintenance failed for {symbol_tf}: {e}")
    
    async def _get_monthly_data(self, symbol: str, timeframe: str, asset_class: str) -> List[Dict]:
        """Get 1 month of data for symbol and timeframe"""
        if asset_class == "crypto":
            return await self._get_crypto_data(symbol, timeframe)
        elif asset_class == "macro":
            return await self._get_macro_data(symbol, timeframe)
        else:
            return await self._get_stock_data(symbol, timeframe)
    
    async def _get_crypto_data(self, symbol: str, timeframe: str) -> List[Dict]:
        """Get crypto data from Binance with correct intervals"""
        for retry in range(2):
            try:
                interval = self.binance_intervals[timeframe]
                binance_symbol = symbol_manager.get_binance_symbol(symbol)
                
                if retry > 0:
                    await asyncio.sleep(30)
                
                klines = await startup_api.get_binance_historical(binance_symbol, 200, interval)
                data = []
                
                for kline in klines:
                    timestamp = datetime.fromtimestamp(kline[0] / 1000)
                    data.append({
                        'timestamp': timestamp,
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5])
                    })
                
                # Debug: Show recent prices from Binance
                if data:
                    recent_prices = [d['close'] for d in data[-5:]]
                    recent_dates = [d['timestamp'].strftime('%Y-%m-%d') for d in data[-5:]]
                    print(f"    📊 {symbol} recent Binance prices: {list(zip(recent_dates, recent_prices))}")
                
                print(f"    ✅ {symbol} {timeframe}: Got {len(data)} records from Binance ({interval})")
                return data
                            
            except Exception as e:
                if "418" in str(e):
                    print(f"    ⚠️ Binance rate limited for {symbol} {timeframe}, skipping")
                    return []
                elif retry < 1:
                    print(f"    ⚠️ Retry {retry+1}/2 for {symbol} {timeframe}: {e}")
                else:
                    print(f"    ❌ Failed for {symbol} {timeframe}: {e}")
        return []
    
    async def _get_stock_data(self, symbol: str, timeframe: str) -> List[Dict]:
        """Get stock data from Yahoo Finance with proper intervals"""
        for retry in range(3):
            try:
                # Get appropriate range and interval - NATIVE ONLY
                if timeframe == '1h':
                    yahoo_interval = '1h'
                    yahoo_range = '730d'
                elif timeframe == '1D':
                    yahoo_interval = '1d'
                    yahoo_range = '5y'
                elif timeframe == '1W':
                    yahoo_interval = '1wk'
                    yahoo_range = '10y'
                elif timeframe == '1M':
                    yahoo_interval = '1mo'
                    yahoo_range = '10y'
                else:
                    yahoo_interval = '1d'
                    yahoo_range = '2y'
                
                # Use custom URL with proper parameters
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={yahoo_interval}&range={yahoo_range}"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=60), headers=headers) as response:
                        if response.status == 200:
                            data_json = await response.json()
                        else:
                            raise Exception(f"Yahoo API failed: {response.status}")
                if 'chart' in data_json and data_json['chart']['result']:
                    result = data_json['chart']['result'][0]
                    timestamps = result['timestamp']
                    indicators = result['indicators']['quote'][0]
                    
                    data = []
                    for i in range(len(timestamps)):
                        if (i < len(indicators['close']) and 
                            indicators['close'][i] is not None):
                            timestamp = datetime.fromtimestamp(timestamps[i])
                            data.append({
                                'timestamp': timestamp,
                                'open': float(indicators['open'][i]),
                                'high': float(indicators['high'][i]),
                                'low': float(indicators['low'][i]),
                                'close': float(indicators['close'][i]),
                                'volume': float(indicators['volume'][i]) if indicators['volume'][i] else 0
                            })
                    
                    # Take last 200 records only
                    data = data[-200:] if len(data) > 200 else data
                    
                    filtered_data = data
                    
                    print(f"    ✅ {symbol} {timeframe}: Got {len(filtered_data)} records from Yahoo")
                    return filtered_data
                else:
                    raise Exception("No chart data available")
                            
            except Exception as e:
                if retry < 2:
                    print(f"    ⚠️ Retry {retry+1}/3 for {symbol} {timeframe}: {e}")
                    await asyncio.sleep(2 ** retry)
                else:
                    print(f"    ❌ Failed after 3 retries for {symbol} {timeframe}: {e}")
        return []
    

    
    async def _get_macro_data(self, symbol: str, timeframe: str) -> List[Dict]:
        """Get real macro economic data from FRED API"""
        try:
            from fredapi import Fred
            fred_api_key = os.getenv('FRED_API_KEY')
            
            if not fred_api_key:
                raise Exception("FRED_API_KEY not configured")
            
            fred = Fred(api_key=fred_api_key)
            
            # FRED series IDs
            fred_series = {
                'GDP': 'GDP',
                'CPI': 'CPIAUCSL',
                'UNEMPLOYMENT': 'UNRATE',
                'FED_RATE': 'FEDFUNDS',
                'CONSUMER_CONFIDENCE': 'UMCSENT'
            }
            
            series_id = fred_series.get(symbol)
            if not series_id:
                raise Exception(f"No FRED series for {symbol}")
            
            # Get real data from FRED
            interval = self.storage_intervals.get(timeframe, timedelta(days=1))
            days_back = self.max_records * (interval.days if interval.days > 0 else 1)
            
            fred_data = fred.get_series(
                series_id,
                observation_start=datetime.now() - timedelta(days=days_back)
            )
            
            if fred_data is None or len(fred_data) == 0:
                raise Exception(f"No FRED data for {symbol}")
            
            # Convert to required format - REAL DATA ONLY
            data = []
            for timestamp, value in fred_data.items():
                current_value = float(value)
                data.append({
                    'timestamp': timestamp,
                    'open': current_value,
                    'high': current_value,  # Same as close - no fake data
                    'low': current_value,   # Same as close - no fake data
                    'close': current_value,
                    'volume': 0             # No volume for macro indicators
                })
            
            # Take last max_records
            data = data[-self.max_records:] if len(data) > self.max_records else data
            
            print(f"    ✅ {symbol} {timeframe}: Got {len(data)} real FRED data points")
            return data
            
        except Exception as e:
            print(f"    ❌ FRED API failed for {symbol}: {e}")
            return []
    
    async def _generate_ml_predictions(self, data: List[Dict], symbol: str, timeframe: str, asset_class: str) -> List[Dict]:
        """Generate ML predictions using raw models for all asset types"""
        predictions = []
        
        if not self.model or len(data) < 20:
            return predictions
        
        # Use raw models for all asset types (crypto/stock/macro)
        use_raw_models = True
        
        for i in range(30, len(data)):
            try:
                hist_prices = np.array([d['close'] for d in data[max(0, i-30):i]])
                if len(hist_prices) < 20:
                    continue
                
                current_price = hist_prices[-1]
                
                # Use raw models for all asset types
                prediction_result = await self._predict_with_raw_models_historical(
                    symbol, timeframe, current_price, hist_prices, asset_class
                )
                if not prediction_result:
                    continue
                
                # Skip if no range predictions from model
                if 'range_low' not in prediction_result or 'range_high' not in prediction_result:
                    continue
                
                predictions.append({
                    'timestamp': data[i]['timestamp'],
                    'actual_price': current_price,
                    'predicted_price': prediction_result['predicted_price'],
                    'range_low': prediction_result['range_low'],
                    'range_high': prediction_result['range_high'],
                    'forecast_direction': prediction_result['forecast_direction'],
                    'confidence': prediction_result['confidence'],
                    'trend_score': int((prediction_result['predicted_price'] - current_price) / current_price * 100),
                    'rsi': 50,
                    'volatility': 0.02
                })
                
                # Fallback if raw model fails
                continue
                
            except Exception as e:
                continue
        
        return predictions
    
    async def _predict_with_raw_models_historical(self, symbol, timeframe, current_price, hist_prices, asset_class):
        """Use raw models for historical predictions (crypto/stock/macro)"""
        try:
            macro_symbols = ['GDP', 'CPI', 'UNEMPLOYMENT', 'FED_RATE', 'CONSUMER_CONFIDENCE']
            
            # Select appropriate model
            if asset_class == 'crypto' and self.model.crypto_raw_models:
                models = self.model.crypto_raw_models
            elif asset_class == 'stocks' and self.model.stock_raw_models:
                models = self.model.stock_raw_models
                # Map gap filling timeframes to trained model keys
                timeframe_map = {'1h': '60m', '1D': '1d', '1W': '1d', '1M': '1mo'}
                timeframe = timeframe_map.get(timeframe, timeframe)
            elif symbol in macro_symbols and self.model.macro_models:
                models = self.model.macro_models
                timeframe = '1D'  # Macro only supports 1D
            else:
                return None
            
            if symbol not in models or timeframe not in models[symbol]:
                return None
            
            model_data = models[symbol][timeframe]
            
            # Calculate features based on asset type
            df = pd.DataFrame({'close': hist_prices})
            
            if asset_class == 'macro':
                # Macro features
                df['change_1'] = df['close'].pct_change(1)
                df['change_4'] = df['close'].pct_change(4)
                df['ma_4'] = df['close'].rolling(4).mean()
                df['ma_12'] = df['close'].rolling(12).mean()
                df['trend'] = (df['close'] - df['ma_12']) / df['ma_12']
                df['volatility'] = df['change_1'].rolling(12).std()
                df['lag_1'] = df['close'].shift(1)
                df['lag_4'] = df['close'].shift(4)
                df['change_lag_1'] = df['change_1'].shift(1)
                
                latest = df.iloc[-1]
                features = {
                    'lag_1': float(latest['lag_1']) if pd.notna(latest['lag_1']) else current_price,
                    'lag_4': float(latest['lag_4']) if pd.notna(latest['lag_4']) else current_price,
                    'ma_4': float(latest['ma_4']) if pd.notna(latest['ma_4']) else current_price,
                    'ma_12': float(latest['ma_12']) if pd.notna(latest['ma_12']) else current_price,
                    'change_1': float(latest['change_1']) if pd.notna(latest['change_1']) else 0.0,
                    'change_4': float(latest['change_4']) if pd.notna(latest['change_4']) else 0.0,
                    'change_lag_1': float(latest['change_lag_1']) if pd.notna(latest['change_lag_1']) else 0.0,
                    'trend': float(latest['trend']) if pd.notna(latest['trend']) else 0.0,
                    'volatility': float(latest['volatility']) if pd.notna(latest['volatility']) else 0.01,
                    'quarter': 1  # Default quarter
                }
            else:
                # Crypto/Stock features
                df['sma_5'] = df['close'].rolling(5).mean()
                df['sma_20'] = df['close'].rolling(20).mean()
                df['price_sma5_ratio'] = df['close'] / df['sma_5']
                df['price_sma20_ratio'] = df['close'] / df['sma_20']
                df['returns'] = df['close'].pct_change()
                df['returns_5'] = df['close'].pct_change(5)
                
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                
                df['momentum_7'] = df['close'] / df['close'].shift(7)
                df['volatility'] = df['returns'].rolling(10).std()
                
                latest = df.iloc[-1]
                features = {
                    'price_sma5_ratio': float(latest['price_sma5_ratio']) if pd.notna(latest['price_sma5_ratio']) else 1.0,
                    'price_sma20_ratio': float(latest['price_sma20_ratio']) if pd.notna(latest['price_sma20_ratio']) else 1.0,
                    'returns': float(latest['returns']) if pd.notna(latest['returns']) else 0.0,
                    'returns_5': float(latest['returns_5']) if pd.notna(latest['returns_5']) else 0.0,
                    'rsi': float(latest['rsi']) if pd.notna(latest['rsi']) else 50.0,
                    'momentum_7': float(latest['momentum_7']) if pd.notna(latest['momentum_7']) else 1.0,
                    'volatility': float(latest['volatility']) if pd.notna(latest['volatility']) else 0.02
                }
            
            # Create feature vector
            feature_vector = np.zeros(len(model_data['features']))
            for i, feature_name in enumerate(model_data['features']):
                feature_vector[i] = features.get(feature_name, 0.0)
            
            # Scale and predict
            features_scaled = model_data['scaler'].transform(feature_vector.reshape(1, -1))
            price_change = model_data['price_model'].predict(features_scaled)[0]
            
            if asset_class == 'macro':
                range_low = model_data['lower_model'].predict(features_scaled)[0]
                range_high = model_data['upper_model'].predict(features_scaled)[0]
            else:
                range_high = model_data['high_model'].predict(features_scaled)[0]
                range_low = model_data['low_model'].predict(features_scaled)[0]
            
            # Clip predictions
            if asset_class == 'crypto':
                price_change = np.clip(price_change, -0.15, 0.15)
                range_high = np.clip(range_high, -0.1, 0.2)
                range_low = np.clip(range_low, -0.2, 0.1)
            elif asset_class == 'stocks':
                price_change = np.clip(price_change, -0.1, 0.1)
                range_high = np.clip(range_high, -0.05, 0.1)
                range_low = np.clip(range_low, -0.1, 0.05)
            else:  # macro
                price_change = np.clip(price_change, -0.05, 0.05)
                range_low = np.clip(range_low, -0.08, 0.08)
                range_high = np.clip(range_high, -0.08, 0.08)
            
            # Calculate confidence
            range_width = abs(range_high - range_low)
            model_r2 = model_data['metrics'].get('price_r2', 0)
            
            if model_r2 > 0.2:
                base_confidence = 75
            elif model_r2 > 0.1:
                base_confidence = 70
            elif model_r2 > 0:
                base_confidence = 65
            else:
                base_confidence = 60
            
            confidence = base_confidence - (range_width * 100)
            confidence = np.clip(confidence, 50, 90)
            
            predicted_price = current_price * (1 + price_change)
            high_price = current_price * (1 + range_high)
            low_price = current_price * (1 + range_low)
            
            if low_price > high_price:
                low_price, high_price = high_price, low_price
            
            if asset_class == 'crypto':
                threshold = 0.005
            elif asset_class == 'stocks':
                threshold = 0.003
            else:  # macro
                threshold = 0.001
            
            direction = 'UP' if price_change > threshold else 'DOWN' if price_change < -threshold else 'HOLD'
            
            return {
                'predicted_price': predicted_price,
                'range_low': low_price,
                'range_high': high_price,
                'forecast_direction': direction,
                'confidence': int(confidence)
            }
            
        except Exception as e:
            return None
    
    def _calculate_accuracy(self, predictions: List[Dict]) -> List[Dict]:
        """Calculate accuracy by comparing predictions with next actual prices"""
        results = []
        
        for i in range(len(predictions) - 1):
            current = predictions[i]
            next_actual = predictions[i + 1]['actual_price']
            
            # Calculate actual direction
            price_change = (next_actual - current['actual_price']) / current['actual_price']
            
            if abs(price_change) > 0.005:
                actual_direction = 'UP' if price_change > 0 else 'DOWN'
            else:
                actual_direction = 'HOLD'
            
            # Compare with prediction
            predicted_direction = current['forecast_direction']
            result = 'Hit' if predicted_direction == actual_direction else 'Miss'
            
            results.append({
                'timestamp': predictions[i + 1]['timestamp'],
                'predicted_direction': predicted_direction,
                'actual_direction': actual_direction,
                'result': result,
                'actual_price': next_actual,
                'predicted_price': current['predicted_price'],
                'confidence': current['confidence']
            })
        
        return results
    
    async def _store_historical_data_efficient(self, symbol: str, timeframe: str, data: List[Dict]):
        """Store all historical data without aggressive filtering"""
        try:
            db_key = symbol_manager.get_db_key(symbol, timeframe)
            
            # Store all data with minimal filtering - just normalize timestamps
            filtered_data = []
            
            for item in sorted(data, key=lambda x: x['timestamp']):
                current_time = item['timestamp']
                
                # Timestamp normalization using centralized utility
                from utils.timestamp_utils import TimestampUtils
                normalized_timestamp = TimestampUtils.adjust_for_timeframe(current_time, timeframe)
                
                filtered_data.append({
                    **item,
                    'timestamp': normalized_timestamp
                })
            
            # Store all filtered data
            async with self.db.pool.acquire() as conn:
                for item in filtered_data:
                    await conn.execute("""
                        INSERT INTO actual_prices (symbol, timeframe, open_price, high, low, 
                                                 close_price, price, volume, timestamp)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (symbol, timestamp) DO UPDATE SET
                            price = EXCLUDED.price,
                            close_price = EXCLUDED.close_price,
                            volume = EXCLUDED.volume
                    """, db_key, timeframe, item['open'], item['high'], item['low'],
                         item['close'], item['close'], item['volume'], item['timestamp'])
            
            print(f"    ✅ {db_key}: Stored {len(filtered_data)} records (filtered from {len(data)})")
                
        except Exception as e:
            print(f"      ❌ Error storing historical data for {db_key}: {e}")
    
    async def _store_predictions(self, symbol: str, timeframe: str, predictions: List[Dict], results: List[Dict]):
        """Store ML predictions and accuracy results with conflict handling"""
        try:
            db_key = symbol_manager.get_db_key(symbol, timeframe)
            async with self.db.pool.acquire() as conn:
                # Store predictions with conflict handling
                for pred in predictions:
                    await conn.execute("""
                        INSERT INTO forecasts (symbol, predicted_price, confidence, 
                                             forecast_direction, trend_score, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT DO NOTHING
                    """, db_key, pred['predicted_price'], pred['confidence'],
                         pred['forecast_direction'], pred['trend_score'], pred['timestamp'])
                
                # Store accuracy results with conflict handling
                for result in results:
                    await conn.execute("""
                        INSERT INTO forecast_accuracy (symbol, actual_direction, result, evaluated_at)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT DO NOTHING
                    """, db_key, result['actual_direction'], result['result'], result['timestamp'])
                
        except Exception as e:
            print(f"      ❌ Error storing predictions for {db_key}: {e}")

# Global monthly data collection service
gap_filler = GapFillingService()