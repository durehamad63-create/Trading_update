#!/usr/bin/env python3
"""
Monthly Data Collection & ML Prediction Service
Collects 1 month of historical data and generates ML predictions for all 25 assets
"""
import asyncio
import aiohttp
import logging
import numpy as np
import random
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
        
        # Only store higher timeframes - no 1m, 5m, 15m
        self.crypto_stock_timeframes = ['1h', '4H', '1D', '7D', '1W', '1M']
        self.macro_timeframes = ['1D', '7D', '1W', '1M']
        
        # Maintain exactly 200 records per timeframe
        self.max_records = 200
        
        # Binance mapping for stablecoins
        self.binance_mapping = {'USDT': 'BTCUSDT', 'USDC': 'BTCUSDT'}
        self.binance_intervals = {
            '1h': '1h', '4H': '4h', '1D': '1d', '7D': '1w', '1W': '1w', '1M': '1M'
        }
        
        # Time intervals for proper data storage (avoid storing every second)
        self.storage_intervals = {
            '1h': timedelta(hours=1),
            '4H': timedelta(hours=4), 
            '1D': timedelta(days=1),
            '7D': timedelta(days=7),
            '1W': timedelta(weeks=1),
            '1M': timedelta(days=30)
        }
        
        print(f"🚀 Efficient Data Collector initialized for {len(self.all_symbols)} assets:")
        print(f"   📈 Crypto: {len(self.crypto_symbols)} symbols")
        print(f"   📊 Stocks: {len(self.stock_symbols)} symbols") 
        print(f"   🏛️ Macro: {len(self.macro_symbols)} symbols")
        print(f"   ⏱️ Timeframes: {self.crypto_stock_timeframes}")
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
        
        # Process all asset classes
        for asset_class, symbols, timeframes in [
            ("crypto", self.crypto_symbols, self.crypto_stock_timeframes),
            ("stocks", self.stock_symbols, self.crypto_stock_timeframes), 
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
                key_timeframes = ['1D', '4H', '1h']  # Check main timeframes
                
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
                # Get appropriate range and interval for real data
                if timeframe == '1h':
                    yahoo_interval = '1h'
                    yahoo_range = '730d'  # 2 years of hourly data
                elif timeframe == '4H':
                    yahoo_interval = '1h'  # Get hourly, aggregate to 4H
                    yahoo_range = '730d'
                elif timeframe == '1D':
                    yahoo_interval = '1d'
                    yahoo_range = '5y'  # 5 years of daily data
                elif timeframe == '7D':
                    yahoo_interval = '1d'  # Get daily, aggregate to weekly
                    yahoo_range = '5y'
                elif timeframe == '1W':
                    yahoo_interval = '1wk'
                    yahoo_range = '10y'  # 10 years of weekly data
                elif timeframe == '1M':
                    yahoo_interval = '1mo'
                    yahoo_range = '10y'  # 10 years of monthly data
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
                    
                    # Aggregate if needed for higher timeframes
                    if timeframe == '4H' and yahoo_interval == '1h':
                        data = self._real_aggregate_to_4h(data)
                    elif timeframe == '7D' and yahoo_interval == '1d':
                        data = self._real_aggregate_to_weekly(data)
                    
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
    
    def _real_aggregate_to_4h(self, hourly_data: List[Dict]) -> List[Dict]:
        """Aggregate real hourly data to 4-hour candles"""
        if not hourly_data:
            return []
        
        four_hour_data = []
        i = 0
        
        while i < len(hourly_data) - 3:
            chunk = hourly_data[i:i+4]
            if len(chunk) == 4:
                four_hour_data.append({
                    'timestamp': chunk[-1]['timestamp'],
                    'open': chunk[0]['open'],
                    'high': max(d['high'] for d in chunk),
                    'low': min(d['low'] for d in chunk),
                    'close': chunk[-1]['close'],
                    'volume': sum(d['volume'] for d in chunk)
                })
            i += 4
        
        return four_hour_data
    
    def _real_aggregate_to_weekly(self, daily_data: List[Dict]) -> List[Dict]:
        """Aggregate real daily data to weekly candles"""
        if not daily_data:
            return []
        
        weekly_data = []
        i = 0
        
        while i < len(daily_data) - 4:
            week_end = min(i + 7, len(daily_data))
            week_data = daily_data[i:week_end]
            
            if len(week_data) >= 5:
                weekly_data.append({
                    'timestamp': week_data[-1]['timestamp'],
                    'open': week_data[0]['open'],
                    'high': max(d['high'] for d in week_data),
                    'low': min(d['low'] for d in week_data),
                    'close': week_data[-1]['close'],
                    'volume': sum(d['volume'] for d in week_data)
                })
            i += 7
        
        return weekly_data
    
    def _aggregate_to_4h(self, data: List[Dict]) -> List[Dict]:
        """Aggregate hourly data to 4-hour candles"""
        if not data:
            return []
        
        aggregated = []
        i = 0
        
        while i < len(data) - 3:
            chunk = data[i:i+4]
            if len(chunk) >= 2:
                aggregated.append({
                    'timestamp': chunk[-1]['timestamp'],
                    'open': chunk[0]['open'],
                    'high': max(d['high'] for d in chunk),
                    'low': min(d['low'] for d in chunk),
                    'close': chunk[-1]['close'],
                    'volume': sum(d['volume'] for d in chunk)
                })
            i += 4
        
        return aggregated
    
    def _aggregate_to_7d(self, data: List[Dict]) -> List[Dict]:
        """Aggregate daily data to 7-day periods"""
        if not data:
            return []
        
        aggregated = []
        i = 0
        
        while i < len(data) - 6:
            chunk = data[i:i+7]
            if len(chunk) >= 5:
                aggregated.append({
                    'timestamp': chunk[-1]['timestamp'],
                    'open': chunk[0]['open'],
                    'high': max(d['high'] for d in chunk),
                    'low': min(d['low'] for d in chunk),
                    'close': chunk[-1]['close'],
                    'volume': sum(d['volume'] for d in chunk)
                })
            i += 7
        
        return aggregated
    
    def _aggregate_to_1m(self, data: List[Dict]) -> List[Dict]:
        """Aggregate daily data to monthly periods"""
        if not data:
            return []
        
        aggregated = []
        i = 0
        
        while i < len(data) - 29:
            chunk = data[i:i+30]
            if len(chunk) >= 20:
                aggregated.append({
                    'timestamp': chunk[-1]['timestamp'],
                    'open': chunk[0]['open'],
                    'high': max(d['high'] for d in chunk),
                    'low': min(d['low'] for d in chunk),
                    'close': chunk[-1]['close'],
                    'volume': sum(d['volume'] for d in chunk)
                })
            i += 30
        
        return aggregated
    
    async def _get_macro_data(self, symbol: str, timeframe: str) -> List[Dict]:
        """Generate synthetic macro economic data with proper intervals"""
        try:
            base_values = {
                'GDP': 27000, 'CPI': 310.5, 'UNEMPLOYMENT': 3.7,
                'FED_RATE': 5.25, 'CONSUMER_CONFIDENCE': 102.3
            }
            
            total_needed = self.max_records
            data = []
            base_value = base_values.get(symbol, 100)
            
            # Generate data with proper time intervals
            interval = self.storage_intervals.get(timeframe, timedelta(days=1))
            
            for i in range(total_needed):
                timestamp = datetime.now() - (interval * (total_needed - i))
                
                if timeframe == '1D':
                    variation = random.uniform(-0.002, 0.002)
                elif timeframe == '7D':
                    variation = random.uniform(-0.008, 0.008)
                elif timeframe == '1W':
                    variation = random.uniform(-0.01, 0.01)
                elif timeframe == '1M':
                    variation = random.uniform(-0.02, 0.02)
                else:
                    variation = random.uniform(-0.005, 0.005)
                
                current_value = base_value * (1 + variation)
                
                data.append({
                    'timestamp': timestamp,
                    'open': current_value,
                    'high': current_value * 1.001,
                    'low': current_value * 0.999,
                    'close': current_value,
                    'volume': random.randint(800000, 1200000)
                })
            
            print(f"    ✅ {symbol} {timeframe}: Generated {len(data)} macro data points")
            return data
            
        except Exception as e:
            print(f"    ❌ Failed to generate macro data for {symbol} {timeframe}: {e}")
            return []
    
    async def _generate_ml_predictions(self, data: List[Dict], symbol: str, timeframe: str, asset_class: str) -> List[Dict]:
        """Generate ML predictions using real model"""
        predictions = []
        
        if not self.model or len(data) < 20:
            return predictions
        
        for i in range(10, len(data)):
            try:
                current_price = data[i-1]['close']
                
                # Use real ML model for prediction
                ml_prediction = await self.model.predict(symbol)
                
                predictions.append({
                    'timestamp': data[i]['timestamp'],
                    'actual_price': current_price,
                    'predicted_price': ml_prediction.get('predicted_price', current_price),
                    'forecast_direction': ml_prediction.get('forecast_direction', 'HOLD'),
                    'confidence': ml_prediction.get('confidence', 75),
                    'trend_score': 0,
                    'rsi': 0,
                    'volatility': 0
                })
                
            except Exception as e:
                continue
        
        return predictions
    
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
                
                # Light timestamp normalization only
                if timeframe == '1h':
                    normalized_timestamp = current_time.replace(minute=0, second=0, microsecond=0)
                elif timeframe == '4H':
                    hour = (current_time.hour // 4) * 4
                    normalized_timestamp = current_time.replace(hour=hour, minute=0, second=0, microsecond=0)
                elif timeframe == '1D':
                    normalized_timestamp = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                else:
                    normalized_timestamp = current_time.replace(second=0, microsecond=0)
                
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