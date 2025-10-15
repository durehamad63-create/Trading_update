"""
Real-time Stock Data Stream Service
"""
import asyncio
import json
import logging
import aiohttp
from datetime import datetime, timedelta, timezone
from typing import Dict
import os
from dotenv import load_dotenv
from utils.error_handler import ErrorHandler
from utils.websocket_security import WebSocketSecurity

logger = logging.getLogger(__name__)

class StockRealtimeService:
    def __init__(self, model=None, database=None):
        self.model = model
        self.database = database
        self.active_connections = {}
        self.price_cache = {}
        self.candle_cache = {}
        
        # Stock symbols from requirements
        self.stock_symbols = {
            'NVDA': 'NVIDIA', 'MSFT': 'Microsoft', 'AAPL': 'Apple',
            'GOOGL': 'Alphabet', 'AMZN': 'Amazon', 'META': 'Meta',
            'AVGO': 'Broadcom', 'TSLA': 'Tesla', 'BRK-B': 'Berkshire Hathaway',
            'JPM': 'JPMorgan Chase'
        }
        
        # API keys from environment
        load_dotenv()
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'YOUR_API_KEY')
        self.iex_token = os.getenv('IEX_CLOUD_TOKEN', 'YOUR_TOKEN')
        
        # Update intervals per timeframe (seconds)
        self.update_intervals = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '4H': 14400, '1D': 86400, '1W': 604800
        }
        
        self.last_update = {}
        self.session = None
        
        # Use centralized cache manager with priority system
        from utils.cache_manager import CacheManager, CacheKeys, CacheTTL, PredictionPriority
        self.cache_manager = CacheManager
        self.cache_keys = CacheKeys
        self.cache_ttl = CacheTTL
        self.prediction_priority = PredictionPriority
    
    async def start_stock_streams(self):
        """Start real-time data collection for all stocks"""
        self.session = aiohttp.ClientSession()
        
        # Start single rotating stream for all stocks
        print(f"🚀 Starting stock streams for {len(self.stock_symbols)} symbols...")
        asyncio.create_task(self._rotating_stock_stream())
    
    async def _rotating_stock_stream(self):
        """Dual stream that processes two stocks simultaneously"""
        symbols = list(self.stock_symbols.keys())
        current_index = 0
        
        while True:
            try:
                # Get two symbols for parallel processing
                symbol1 = symbols[current_index]
                symbol2 = symbols[(current_index + 1) % len(symbols)]
                
                # Process two stocks simultaneously
                tasks = [
                    self._process_single_stock(symbol1),
                    self._process_single_stock(symbol2)
                ]
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Move to next pair of symbols
                current_index = (current_index + 2) % len(symbols)
                
                # Wait 1 second before next pair (completes full cycle in 5 seconds)
                await asyncio.sleep(1)
                
            except Exception as e:
                ErrorHandler.log_stream_error('stock_dual', 'ALL', str(e))
                await asyncio.sleep(5)
    
    async def _process_single_stock(self, symbol):
        """Process a single stock update"""
        try:
            # Get real-time price data
            price_data = await self._get_realtime_stock_data(symbol)
            if price_data:

                # Update cache with timezone-aware timestamp
                cache_data = {
                    **price_data,
                    'timestamp': WebSocketSecurity.get_utc_now()
                }
                self.price_cache[symbol] = cache_data
                
                # Cache using centralized manager with standard TTL
                cache_key = self.cache_keys.price(symbol, 'stock')
                self.cache_manager.set_cache(cache_key, cache_data, ttl=self.cache_ttl.PRICE_STOCK)
                # Update candles and broadcast only if connections exist
                if symbol in self.active_connections and self.active_connections[symbol]:
                    # Immediate price broadcast
                    asyncio.create_task(self._broadcast_stock_price_update(symbol, price_data))
                    # Update candles and forecasts with priority-based rate limiting
                    asyncio.create_task(self._update_stock_candles_and_forecast(symbol, price_data))
                else:
                    # Store data for all timeframes even without active connections
                    asyncio.create_task(self._store_stock_data_all_timeframes(symbol, price_data))
        except Exception as e:
            logger.error(f"Stock {symbol} error: {e}", exc_info=True)
            ErrorHandler.log_stream_error('stock', symbol, str(e))
    

    
    async def _get_realtime_stock_data(self, symbol) -> Dict:
        """Get real-time stock data with rate limiting and fallback"""
        
        # No delay needed in rotating pattern
        
        # 1. Try Yahoo Finance direct API with rate limit handling
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            async with self.session.get(url, timeout=15, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'chart' in data and data['chart']['result']:
                        result = data['chart']['result'][0]
                        meta = result['meta']
                        current_price = meta['regularMarketPrice']
                        prev_close = meta['previousClose']
                        change_pct = ((current_price - prev_close) / prev_close) * 100
                        
                        return {
                            'current_price': float(current_price),
                            'change_24h': float(change_pct),
                            'volume': float(meta.get('regularMarketVolume', 0)),
                            'high': float(meta.get('regularMarketDayHigh', current_price)),
                            'low': float(meta.get('regularMarketDayLow', current_price)),
                            'data_source': 'Yahoo Finance API'
                        }
                elif response.status == 429:
                    return None
        except Exception as e:
            pass
        
        # 2. Try Alpha Vantage if available
        try:
            if self.alpha_vantage_key != "YOUR_API_KEY":
                url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.alpha_vantage_key}"
                async with self.session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "Global Quote" in data:
                            quote = data["Global Quote"]
                            return {
                                'current_price': float(quote["05. price"]),
                                'change_24h': float(quote["10. change percent"].replace('%', '')),
                                'volume': float(quote["06. volume"]),
                                'high': float(quote["03. high"]),
                                'low': float(quote["04. low"]),
                                'data_source': 'Alpha Vantage'
                            }
        except Exception as e:
            pass
        
        # 3. Try IEX Cloud if available
        try:
            if self.iex_token != "YOUR_TOKEN":
                url = f"https://cloud.iexapis.com/stable/stock/{symbol}/quote?token={self.iex_token}"
                async with self.session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'current_price': float(data["latestPrice"]),
                            'change_24h': float(data["changePercent"]) * 100,
                            'volume': float(data["latestVolume"]),
                            'high': float(data["high"]),
                            'low': float(data["low"]),
                            'data_source': 'IEX Cloud'
                        }
        except Exception as e:
            pass
        
        return None
    
    async def _update_stock_candles_and_forecast(self, symbol, price_data):
        """Update candle data and generate forecasts for active timeframes"""
        try:
            current_time = datetime.now()
            
            # Get unique timeframes from active connections
            timeframes = set()
            if symbol in self.active_connections:
                for conn_data in self.active_connections[symbol].values():
                    timeframes.add(conn_data['timeframe'])
            
            # Update candles for each timeframe
            for timeframe in timeframes:
                await self._update_stock_candle_data(symbol, timeframe, price_data, current_time)
                
                # Rate limit predictions per timeframe
                rate_key = f"{symbol}_{timeframe}"
                if rate_key in self.last_update:
                    time_diff = (current_time - self.last_update[rate_key]).total_seconds()
                    if time_diff < self.update_intervals.get(timeframe, 300):
                        continue
                
                # Generate timeframe-specific forecast
                await self._generate_stock_forecast(symbol, timeframe, current_time)
                self.last_update[rate_key] = current_time
                
        except Exception as e:
            logger.error(f"Stock candle error for {symbol}: {e}", exc_info=True)
            ErrorHandler.log_stream_error('stock_candle', symbol, str(e))
    
    async def _update_stock_candle_data(self, symbol, timeframe, price_data, timestamp):
        """Update candle data for specific timeframe"""
        try:
            if symbol not in self.candle_cache:
                self.candle_cache[symbol] = {}
            if timeframe not in self.candle_cache[symbol]:
                self.candle_cache[symbol][timeframe] = []
            
            # Get timeframe interval in minutes
            interval_map = {
                '1m': 1, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '4H': 240, '1D': 1440, '1W': 10080
            }
            interval_minutes = interval_map.get(timeframe, 60)
            
            # Round timestamp to candle boundary
            candle_start = timestamp.replace(second=0, microsecond=0)
            if interval_minutes >= 60:
                candle_start = candle_start.replace(minute=0)
                if interval_minutes >= 1440:
                    candle_start = candle_start.replace(hour=0)
            else:
                minute_boundary = (candle_start.minute // interval_minutes) * interval_minutes
                candle_start = candle_start.replace(minute=minute_boundary)
            
            candles = self.candle_cache[symbol][timeframe]
            current_price = price_data['current_price']
            volume = price_data['volume']
            
            # Update or create current candle
            if candles and candles[-1]['timestamp'] == candle_start:
                candle = candles[-1]
                candle['high'] = max(candle['high'], current_price)
                candle['low'] = min(candle['low'], current_price)
                candle['close'] = current_price
                candle['volume'] += volume
            else:
                new_candle = {
                    'timestamp': candle_start,
                    'open': current_price,
                    'high': current_price,
                    'low': current_price,
                    'close': current_price,
                    'volume': volume
                }
                candles.append(new_candle)
                
                if len(candles) > 100:
                    candles.pop(0)
                    
        except Exception as e:
            ErrorHandler.log_stream_error('stock_candle_data', 'ALL', str(e))
    
    async def _broadcast_stock_price_update(self, symbol, price_data):
        """Immediate stock price broadcast without ML predictions"""
        try:
            current_time = datetime.now()
            
            # Get unique timeframes from active connections
            timeframes = set()
            if symbol in self.active_connections:
                for conn_data in self.active_connections[symbol].values():
                    timeframes.add(conn_data['timeframe'])
            
            # Broadcast to all timeframes immediately
            for timeframe in timeframes:
                stock_price_data = {
                    "type": "stock_price_update",
                    "symbol": str(symbol),
                    "timeframe": str(timeframe),
                    "current_price": float(price_data['current_price']),
                    "change_24h": float(price_data['change_24h']),
                    "volume": float(price_data['volume']),
                    "data_source": price_data.get('data_source', 'Stock API'),
                    "timestamp": current_time.strftime("%H:%M:%S"),
                    "last_updated": current_time.isoformat()
                }
                
                await self._broadcast_to_timeframe(symbol, timeframe, stock_price_data)
                
        except Exception as e:
            ErrorHandler.log_websocket_error('stock_broadcast', str(e))
    
    def _adjust_timestamp_for_timeframe(self, timestamp, timeframe):
        """Adjust timestamp to prevent duplicates for different timeframes"""
        if timeframe == '1W':
            # Weekly: round to start of week (Monday)
            days_since_monday = timestamp.weekday()
            week_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            week_start = week_start - timedelta(days=days_since_monday)
            return week_start
        elif timeframe == '1D':
            # Daily: round to start of day
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        elif timeframe == '4H':
            # 4-hour: round to 4-hour boundaries (0, 4, 8, 12, 16, 20)
            hour_boundary = (timestamp.hour // 4) * 4
            return timestamp.replace(hour=hour_boundary, minute=0, second=0, microsecond=0)
        elif timeframe == '1h':
            # Hourly: round to start of hour
            return timestamp.replace(minute=0, second=0, microsecond=0)
        elif timeframe in ['15m', '30m']:
            # 15/30 minute: round to appropriate boundaries
            interval = 15 if timeframe == '15m' else 30
            minute_boundary = (timestamp.minute // interval) * interval
            return timestamp.replace(minute=minute_boundary, second=0, microsecond=0)
        elif timeframe == '5m':
            # 5 minute: round to 5-minute boundaries
            minute_boundary = (timestamp.minute // 5) * 5
            return timestamp.replace(minute=minute_boundary, second=0, microsecond=0)
        else:
            # 1m and others: round to minute
            return timestamp.replace(second=0, microsecond=0)
    
    async def _store_stock_data_all_timeframes(self, symbol, price_data):
        """Store stock price data for all timeframes with priority-based prediction storage"""
        try:
            from config.symbol_manager import symbol_manager
            from utils.timestamp_utils import TimestampUtils
            
            current_time = datetime.now()
            timeframes = ['1h', '4H', '1D', '1W']
            
            for timeframe in timeframes:
                db_key = symbol_manager.get_db_key(symbol, timeframe)
                adjusted_time = TimestampUtils.adjust_for_timeframe(current_time, timeframe)
                
                adjusted_price_data = {
                    **price_data,
                    'timestamp': adjusted_time,
                    'data_source': 'yahoo'
                }
                
                if self.database and self.database.pool:
                    # Always store price data
                    await self.database.store_actual_price(db_key, adjusted_price_data, timeframe)
                    
                    # Store prediction only if update is due (based on priority)
                    if self.cache_manager.should_update_prediction(symbol, timeframe):
                        try:
                            # Get cached prediction (already generated)
                            cache_key = self.cache_keys.prediction(symbol, timeframe)
                            prediction = self.cache_manager.get_cache(cache_key)
                            
                            if prediction:
                                await self.database.store_forecast(db_key, prediction, timeframe)
                        except Exception:
                            pass
                        
        except Exception as e:
            ErrorHandler.log_database_error('stock_store_timeframes', 'ALL', str(e))
    
    async def _generate_stock_forecast(self, symbol, timeframe, current_time):
        """Generate ML forecast for stock symbol and timeframe with priority-based caching"""
        try:
            # Get candle data
            if (symbol not in self.candle_cache or 
                timeframe not in self.candle_cache[symbol] or 
                not self.candle_cache[symbol][timeframe]):
                return
            
            candles = self.candle_cache[symbol][timeframe]
            current_candle = candles[-1]
            current_price = float(current_candle['close'])
            
            # Check cache first (unified caching)
            cache_key = self.cache_keys.prediction(symbol, timeframe)
            cached_pred = self.cache_manager.get_cache(cache_key)
            
            if cached_pred:
                # Use cached prediction
                predicted_price = float(cached_pred.get('predicted_price', current_price))
                forecast_direction = cached_pred.get('forecast_direction', 'HOLD')
                confidence = cached_pred.get('confidence', 75)
            else:
                # Check if we should generate fresh prediction based on priority
                if self.cache_manager.should_update_prediction(symbol, timeframe):
                    # Generate fresh prediction (blocking - ensures data consistency)
                    try:
                        prediction = await self.model.predict(symbol, timeframe)
                        predicted_price = float(prediction.get('predicted_price', current_price))
                        forecast_direction = prediction.get('forecast_direction', 'HOLD')
                        confidence = prediction.get('confidence', 75)
                        
                        # Mark as updated
                        self.cache_manager.mark_prediction_updated(symbol, timeframe)
                    except Exception as e:
                        # Fallback to current price
                        predicted_price = current_price
                        forecast_direction = 'HOLD'
                        confidence = 75
                else:
                    # Use current price as fallback (prediction not due yet)
                    predicted_price = current_price
                    forecast_direction = 'HOLD'
                    confidence = 75
            
            # Chart data
            chart_points = 50
            past_prices = [float(c['close']) for c in candles[-chart_points:]]
            future_prices = [predicted_price]
            timestamps = [c['timestamp'].strftime("%H:%M" if timeframe in ['1m', '5m', '15m', '30m', '1h'] else "%m-%d") for c in candles[-chart_points:]]
            
            forecast_data = {
                "type": "stock_forecast",
                "symbol": str(symbol),
                "timeframe": str(timeframe),
                "name": str(self.stock_symbols.get(symbol, symbol)),
                "forecast_direction": str(forecast_direction),
                "confidence": int(confidence),
                "chart": {
                    "past": past_prices,
                    "future": future_prices,
                    "timestamps": timestamps
                },
                "last_updated": current_time.isoformat(),
                "current_price": float(current_price),
                "change_24h": float(self.price_cache.get(symbol, {}).get('change_24h', 0)),
                "volume": float(current_candle['volume'])
            }
            
            # Broadcast to connections
            await self._broadcast_to_timeframe(symbol, timeframe, forecast_data)
            
        except Exception as e:
            ErrorHandler.log_prediction_error('stock_forecast', str(e))
    
    async def _broadcast_to_timeframe(self, symbol, timeframe, data):
        """Efficient broadcast with connection pooling"""
        if symbol not in self.active_connections:
            return
        
        # Pre-serialize message once
        message = json.dumps(data, default=str)
        
        # Get matching connections
        matching_connections = [
            (conn_id, conn_data['websocket']) 
            for conn_id, conn_data in self.active_connections[symbol].items() 
            if conn_data['timeframe'] == timeframe
        ]
        
        if not matching_connections:
            return
        
        # Batch send with concurrency control
        semaphore = asyncio.Semaphore(15)  # Higher limit for stocks
        
        async def send_to_connection(conn_id, websocket):
            async with semaphore:
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    if "connectionclosed" in str(e).lower():
                        self.active_connections[symbol].pop(conn_id, None)
        
        await asyncio.gather(
            *[send_to_connection(conn_id, ws) for conn_id, ws in matching_connections],
            return_exceptions=True
        )
    

    
    async def add_connection(self, websocket, symbol, connection_id, timeframe='1D'):
        """Add connection with pooling optimization"""
        print(f"📈 Stock service adding connection for {symbol} with ID {connection_id}")
        
        try:
            if symbol not in self.active_connections:
                self.active_connections[symbol] = {}
                print(f"🆕 Created new stock symbol entry for {symbol}")
            
            self.active_connections[symbol][connection_id] = {
                'websocket': websocket,
                'timeframe': timeframe,
                'connected_at': datetime.now()
            }
            logging.debug("Stock connection stored for %s", symbol)
            
            # Send historical data
            print(f"📊 Sending stock historical data for {symbol}")
            await self._send_stock_historical_data(websocket, symbol, timeframe)
            print(f"✅ Stock historical data sent for {symbol}")
            
        except Exception as e:
            print(f"❌ Error in stock add_connection for {symbol}: {e}")
            raise
    
    async def _send_stock_historical_data(self, websocket, symbol, timeframe):
        """Send historical stock data with caching"""
        print(f"📊 _send_stock_historical_data called for {symbol} {timeframe}")
        try:
            # Check cache first
            cache_key = f"stock_history:{symbol}:{timeframe}"
            if hasattr(self, 'data_cache') and cache_key in self.data_cache:
                cached_item = self.data_cache[cache_key]
                if (datetime.now() - cached_item['timestamp']).total_seconds() < 300:
                    print(f"💾 Using cached stock data for {symbol}")
                    await websocket.send_text(cached_item['data'])
                    return
                else:
                    print(f"⏰ Stock cache expired for {symbol}")
            else:
                print(f"🚫 No stock cache for {symbol}")
            
            # Use database from constructor or fallback to global
            db = self.database
            if not db or not db.pool:
                try:
                    from database import db as global_db
                    if global_db and global_db.pool:
                        db = global_db
                    else:
                        return
                except Exception:
                    return
                
            # Try multiple database query formats for stock historical data
            chart_data = None
            actual_data = []
            forecast_data = []
            timestamps = []
            
            # Use centralized key generation
            from config.symbol_manager import symbol_manager
            db_key = symbol_manager.get_db_key(symbol, timeframe)
            query_attempts = [db_key]
            
            for attempt, query_symbol in enumerate(query_attempts):
                print(f"📊 Stock DB Query attempt {attempt+1}: {query_symbol}")
                try:
                    chart_data = await db.get_chart_data(query_symbol, timeframe)
                    if chart_data and chart_data.get('actual') and chart_data.get('forecast'):
                        print(f"✅ Found stock data: {len(chart_data['actual'])} actual, {len(chart_data['forecast'])} forecast")
                        
                        # Process successful data
                        min_length = min(len(chart_data['actual']), len(chart_data['forecast']), len(chart_data['timestamps']))
                        points = min(50, min_length)
                        
                        actual_data = [float(x) for x in chart_data['actual'][-points:]]
                        forecast_data = [float(x) for x in chart_data['forecast'][-points:]]
                        timestamps = [str(x) for x in chart_data['timestamps'][-points:]]
                        break
                except Exception as e:
                    print(f"❌ Stock query {attempt+1} failed: {e}")
                    continue
            
            # If no database data, return error - DO NOT generate synthetic data
            if not actual_data or not forecast_data:
                print(f"❌ No historical data available for stock {symbol} {timeframe}")
                error_data = {
                    "type": "error",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "message": f"No historical data available for {symbol}. Please wait for data collection to complete."
                }
                await websocket.send_text(json.dumps(error_data))
                return
            
            historical_message = {
                "type": "historical_data",  # Standardize message type
                "symbol": str(symbol),
                "timeframe": str(timeframe),
                "name": str(self.stock_symbols.get(symbol, symbol)),
                "chart": {
                    "actual": actual_data,  # Standardize field names
                    "predicted": forecast_data,
                    "timestamps": timestamps
                },
                "last_updated": datetime.now().isoformat()
            }
            
            message_json = json.dumps(historical_message)
            
            # Cache using centralized manager with standard TTL
            self.cache_manager.set_cache(cache_key, message_json, ttl=self.cache_ttl.WEBSOCKET_HISTORY)
            
            await websocket.send_text(message_json)
            
        except Exception as e:
            print(f"❌ _send_stock_historical_data failed for {symbol}: {e}")
            # Send error response instead of silent failure
            try:
                error_data = {
                    "type": "error",
                    "symbol": symbol,
                    "message": f"Stock historical data error: {str(e)}"
                }
                await websocket.send_text(json.dumps(error_data))
            except:
                pass
    

    
    def remove_connection(self, symbol, connection_id):
        """Remove WebSocket connection"""
        if symbol in self.active_connections and connection_id in self.active_connections[symbol]:
            del self.active_connections[symbol][connection_id]
    
    async def close(self):
        """Close the service and cleanup"""
        if self.session:
            await self.session.close()

# Global stock service instance
stock_realtime_service = None