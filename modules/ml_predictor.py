"""
ML Prediction Module
"""
import asyncio
import logging
import time
import pickle
import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import warnings
import joblib
from dotenv import load_dotenv
from multi_asset_support import multi_asset
from database import db
from config.symbols import CRYPTO_SYMBOLS, STOCK_SYMBOLS
from utils.api_client import APIClient
from utils.error_handler import ErrorHandler
from utils.cache_manager import CacheKeys

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available, using enhanced fallback predictions")

# Load environment variables
load_dotenv()

class MobileMLModel:
    def __init__(self):
        self.last_request_time = {}
        self.min_request_interval = 0.5  # 500ms to reduce rate limiting
        self.xgb_model = None
        self.prediction_cache = {}
        self.cache_ttl = 1  # 1 second cache for real-time updates
        self.prediction_semaphore = asyncio.Semaphore(5)  # Limit concurrent predictions
        
        # Load macro models
        self.macro_models = None
        macro_model_path = 'models/macro/macro_range_models.pkl'
        
        if not os.path.exists(macro_model_path):
            macro_url = os.getenv('MACRO_MODEL_URL')
            if macro_url:
                print(f"📥 Downloading macro models...")
                self._download_raw_model(macro_url, macro_model_path, 'macro')
        
        if os.path.exists(macro_model_path):
            try:
                self.macro_models = joblib.load(macro_model_path)
                print(f"✅ Macro models loaded")
            except Exception as e:
                print(f"⚠️ Macro models failed to load: {e}")
        
        # Always use raw models for all asset types
        self.use_raw_models = True
        
        self.crypto_raw_models = None
        self.stock_raw_models = None
        
        # Load crypto raw models (REQUIRED for crypto predictions)
        crypto_model_path = 'models/crypto_raw/crypto_raw_models.pkl'
        if not os.path.exists(crypto_model_path):
            crypto_url = os.getenv('CRYPTO_RAW_MODEL_URL')
            if crypto_url:
                print(f"📥 Downloading crypto raw models...")
                self._download_raw_model(crypto_url, crypto_model_path, 'crypto')
            else:
                raise Exception(f"❌ CRITICAL: Crypto raw models not found")
        
        try:
            self.crypto_raw_models = joblib.load(crypto_model_path)
            print(f"✅ Crypto raw models loaded")
        except Exception as e:
            raise Exception(f"❌ CRITICAL: Crypto raw models failed: {e}")
        
        # Load stock raw models (REQUIRED for stock predictions)
        stock_model_path = 'models/stock_raw/stock_raw_models.pkl'
        if not os.path.exists(stock_model_path):
            stock_url = os.getenv('STOCK_RAW_MODEL_URL')
            if stock_url:
                print(f"📥 Downloading stock raw models...")
                self._download_raw_model(stock_url, stock_model_path, 'stock')
            else:
                raise Exception(f"❌ CRITICAL: Stock raw models not found")
        
        try:
            self.stock_raw_models = joblib.load(stock_model_path)
            print(f"✅ Stock raw models loaded")
        except Exception as e:
            raise Exception(f"❌ CRITICAL: Stock raw models failed: {e}")
        
        print(f"🔧 Model Configuration: Crypto={bool(self.crypto_raw_models)}, Stock={bool(self.stock_raw_models)}, Macro={bool(self.macro_models)}")
        
        # Use centralized cache manager with priority system
        from utils.cache_manager import CacheManager, CacheKeys, CacheTTL, PredictionPriority
        self.cache_manager = CacheManager
        self.cache_keys = CacheKeys
        self.cache_ttl = CacheTTL
        self.prediction_priority = PredictionPriority
        
        print(f"🔧 ML Predictor initialized: min_interval={self.min_request_interval*1000:.0f}ms, cache_ttl={self.cache_ttl}s")
    
    def _download_model_from_drive_old(self, model_path):
        """Download model from Google Drive with virus scan bypass"""
        try:
            file_id = "10uBJLKsijJHDFBOhCsyFFi-1DAhamjez"
            
            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            print(f"Downloading model from Google Drive...")
            
            # Use direct download URL with confirmation for large files
            download_url = f"https://drive.usercontent.google.com/download?id={file_id}&confirm=t"
            
            session = requests.Session()
            response = session.get(download_url, stream=True, timeout=300, allow_redirects=True)
            response.raise_for_status()
            
            # Check if we got HTML (virus scan page) instead of binary data
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                raise Exception("Got HTML page instead of file - virus scan blocking download")
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"Model downloaded successfully to {model_path}")
            
            # Verify file size
            file_size = os.path.getsize(model_path)
            if file_size < 1000000:  # Less than 1MB indicates error (model should be ~161MB)
                raise Exception(f"Downloaded file too small ({file_size} bytes), expected ~161MB")
            
        except Exception as e:
            raise Exception(f"Failed to download model from Google Drive: {str(e)}")
    
    def _download_raw_model(self, drive_url, model_path, model_type):
        """Download raw model from Google Drive"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            print(f"📥 Downloading {model_type} raw model from Google Drive...")
            
            # Extract file ID from URL
            if '/file/d/' in drive_url:
                file_id = drive_url.split('/file/d/')[1].split('/')[0]
            elif 'id=' in drive_url:
                file_id = drive_url.split('id=')[1].split('&')[0]
            else:
                raise Exception(f"Invalid URL: {drive_url}")
            
            # Use direct download URL with confirmation
            download_url = f"https://drive.usercontent.google.com/download?id={file_id}&confirm=t"
            
            session = requests.Session()
            response = session.get(download_url, stream=True, timeout=300, allow_redirects=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                raise Exception(f"Got HTML page instead of file - virus scan may be blocking download")
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Verify file size
            file_size = os.path.getsize(model_path)
            if file_size < 1000:
                raise Exception(f"Downloaded file too small ({file_size} bytes)")
            
            print(f"✅ {model_type.title()} raw model downloaded successfully ({file_size:,} bytes)")
            
        except Exception as e:
            if os.path.exists(model_path):
                os.remove(model_path)
            raise Exception(f"Failed to download {model_type} raw model: {str(e)}")
    
    async def predict(self, symbol, timeframe='1D'):
        """Generate real model prediction with unified priority-based caching"""
        import time
        
        # Check unified cache first
        cache_key = self.cache_keys.prediction(symbol, timeframe)
        cached_data = self.cache_manager.get_cache(cache_key)
        
        if cached_data:
            return cached_data
        
        async with self.prediction_semaphore:
            
            # Use raw models for all asset types
            try:
                result = await self._predict_with_raw_models(symbol, timeframe)
                if result:
                    ttl = self.prediction_priority.get_cache_ttl(symbol)
                    self.cache_manager.set_cache(cache_key, result, ttl)
                    return result
            except Exception as e:
                raise Exception(f"Prediction failed for {symbol}: {e}")
            
            raise Exception(f"No model available for {symbol}")
    
    def predict_for_timestamp(self, symbol, timestamp):
        """Generate ML prediction for specific timestamp"""
        try:
            # Get current prediction as baseline
            current_pred = self.predict(symbol)
            current_price = current_pred['current_price']
            
            # Calculate time difference from now
            from datetime import datetime
            now = datetime.now()
            time_diff_hours = (timestamp - now).total_seconds() / 3600
            
            # Apply time-based prediction model using real trend
            trend_factor = current_pred['change_24h'] / 100
            time_decay = max(0.1, 1 - abs(time_diff_hours) * 0.02)
            
            # Deterministic prediction based on trend only
            predicted_price = current_price * (1 + trend_factor * time_decay)
            
            return max(0.01, predicted_price)
            
        except Exception as e:
            logging.error(f"Timestamp prediction failed for {symbol}: {e}")
            raise
    
    async def get_historical_predictions(self, symbol, num_points=50):
        """Get historical predictions from database"""
        try:
            from database import db
            if not db or not db.pool:
                logging.error("Database not available for historical predictions")
                return []
            
            # Get recent forecasts from database
            async with db.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT predicted_price, created_at, confidence
                    FROM forecasts
                    WHERE symbol = $1 AND predicted_price IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT $2
                """, symbol, num_points)
                
                if rows:
                    predictions = []
                    for row in reversed(rows):  # Reverse to get chronological order
                        predictions.append({
                            'timestamp': row['created_at'].isoformat(),
                            'predicted_price': float(row['predicted_price']),
                            'confidence': row['confidence']
                        })
                    
                    pass
                    return predictions
                else:
                    pass
                    return []
                    
        except Exception as e:
            pass
            return []
    
    async def _get_real_price(self, symbol):
        """Get real price from cached services - FAST"""
        try:
            # Try cached prices from realtime services first (FAST)
            from utils.cache_manager import CacheManager, CacheKeys
            
            # Check crypto cache
            if symbol in CRYPTO_SYMBOLS:
                cache_key = CacheKeys.price(symbol, 'crypto')
                cached = CacheManager.get_cache(cache_key)
                if cached and 'current_price' in cached:
                    return cached['current_price']
                
                # Try realtime service cache
                try:
                    import realtime_websocket_service as rws
                    if rws.realtime_service and symbol in rws.realtime_service.price_cache:
                        return rws.realtime_service.price_cache[symbol]['current_price']
                except:
                    pass
                
                # Handle stablecoins
                if CRYPTO_SYMBOLS[symbol].get('fixed_price'):
                    return CRYPTO_SYMBOLS[symbol]['fixed_price']
            
            # Check stock cache
            if symbol in STOCK_SYMBOLS:
                cache_key = CacheKeys.price(symbol, 'stock')
                cached = CacheManager.get_cache(cache_key)
                if cached and 'current_price' in cached:
                    return cached['current_price']
                
                # Try stock service cache
                try:
                    import stock_realtime_service as stock
                    if stock.stock_realtime_service and symbol in stock.stock_realtime_service.price_cache:
                        return stock.stock_realtime_service.price_cache[symbol]['current_price']
                except:
                    pass
            
            # Check macro cache
            macro_symbols = ['GDP', 'CPI', 'UNEMPLOYMENT', 'FED_RATE', 'CONSUMER_CONFIDENCE']
            if symbol in macro_symbols:
                cache_key = CacheKeys.price(symbol, 'macro')
                cached = CacheManager.get_cache(cache_key)
                if cached and 'current_price' in cached:
                    return cached['current_price']
                
                # Try macro service cache
                try:
                    import macro_realtime_service as macro
                    if macro.macro_realtime_service and symbol in macro.macro_realtime_service.price_cache:
                        return macro.macro_realtime_service.price_cache[symbol]['current_price']
                except:
                    pass
                
                # No fallback - fail if real data unavailable
                raise Exception(f"No real price data available for macro indicator {symbol}")
            
            raise Exception(f"No cached price for {symbol}")
        except Exception as e:
            ErrorHandler.log_prediction_error(symbol, f"Price fetch failed: {e}")
            raise
    
    async def _get_real_change(self, symbol):
        """Get real 24h change from cached services - FAST"""
        try:
            from utils.cache_manager import CacheManager, CacheKeys
            
            # Check crypto cache
            if symbol in CRYPTO_SYMBOLS:
                cache_key = CacheKeys.price(symbol, 'crypto')
                cached = CacheManager.get_cache(cache_key)
                if cached and 'change_24h' in cached:
                    return cached['change_24h']
                
                # Try realtime service
                try:
                    import realtime_websocket_service as rws
                    if rws.realtime_service and symbol in rws.realtime_service.price_cache:
                        return rws.realtime_service.price_cache[symbol].get('change_24h', 0.0)
                except:
                    pass
                
                if CRYPTO_SYMBOLS[symbol].get('fixed_price'):
                    return 0.0
            
            # Check stock cache
            if symbol in STOCK_SYMBOLS:
                cache_key = CacheKeys.price(symbol, 'stock')
                cached = CacheManager.get_cache(cache_key)
                if cached and 'change_24h' in cached:
                    return cached['change_24h']
                
                try:
                    import stock_realtime_service as stock
                    if stock.stock_realtime_service and symbol in stock.stock_realtime_service.price_cache:
                        return stock.stock_realtime_service.price_cache[symbol].get('change_24h', 0.0)
                except:
                    pass
            
            # Check macro cache
            macro_symbols = ['GDP', 'CPI', 'UNEMPLOYMENT', 'FED_RATE', 'CONSUMER_CONFIDENCE']
            if symbol in macro_symbols:
                cache_key = CacheKeys.price(symbol, 'macro')
                cached = CacheManager.get_cache(cache_key)
                if cached and 'change_24h' in cached:
                    return cached['change_24h']
                
                try:
                    import macro_realtime_service as macro
                    if macro.macro_realtime_service and symbol in macro.macro_realtime_service.price_cache:
                        return macro.macro_realtime_service.price_cache[symbol].get('change_24h', 0.0)
                except:
                    pass
            
        except Exception as e:
            ErrorHandler.log_prediction_error(symbol, f"Change fetch failed: {e}")
        return 0.0
    
    def _get_real_historical_volumes(self, symbol):
        """Get real historical volumes for stocks"""
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1mo"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, timeout=15, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if 'chart' in data and data['chart']['result']:
                    result = data['chart']['result'][0]
                    indicators = result['indicators']['quote'][0]
                    volumes = [x if x is not None else 0 for x in indicators['volume']]
                    return volumes[-30:] if len(volumes) > 30 else volumes
        except:
            pass
        return []
    
    def _get_real_historical_prices(self, symbol):
        """Get real historical prices - Binance for crypto, YFinance for stocks, FRED for macro"""
        crypto_symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOGE', 'USDT', 'USDC', 'TRX']
        macro_symbols = ['GDP', 'CPI', 'UNEMPLOYMENT', 'FED_RATE', 'CONSUMER_CONFIDENCE']
        
        try:
            # Stablecoins return fixed price history
            if symbol in ['USDT', 'USDC']:
                return [1.0] * 30
            
            if symbol in crypto_symbols:
                # Use Binance for crypto historical data
                binance_map = {
                    'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'BNB': 'BNBUSDT',
                    'SOL': 'SOLUSDT', 'ADA': 'ADAUSDT', 'XRP': 'XRPUSDT', 
                    'DOGE': 'DOGEUSDT', 'TRX': 'TRXUSDT'
                }
                
                if symbol in binance_map:
                    url = f"https://api.binance.com/api/v3/klines?symbol={binance_map[symbol]}&interval=1d&limit=30"
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        klines = response.json()
                        prices = [float(k[4]) for k in klines]  # Close prices
                        return prices
            elif symbol in macro_symbols:
                # Use FRED for macro historical data
                from fredapi import Fred
                fred_api_key = os.getenv('FRED_API_KEY')
                if not fred_api_key:
                    raise Exception("FRED_API_KEY not found")
                
                fred = Fred(api_key=fred_api_key)
                fred_series = {
                    'GDP': 'GDP', 'CPI': 'CPIAUCSL', 'UNEMPLOYMENT': 'UNRATE',
                    'FED_RATE': 'FEDFUNDS', 'CONSUMER_CONFIDENCE': 'UMCSENT'
                }
                
                series_id = fred_series.get(symbol)
                if series_id:
                    data = fred.get_series(series_id)
                    if data is not None and len(data) > 0:
                        prices = [float(x) for x in data.values if pd.notna(x)]
                        return prices[-30:] if len(prices) > 30 else prices
            else:
                # Use direct Yahoo Finance API for stock historical data
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1mo"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, timeout=15, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    if 'chart' in data and data['chart']['result']:
                        result = data['chart']['result'][0]
                        indicators = result['indicators']['quote'][0]
                        closes = [x for x in indicators['close'] if x is not None]
                        if closes:
                            return closes[-30:]  # Last 30 days
                return []
                    
        except Exception as e:
            raise Exception(f"Failed to get real historical prices for {symbol}: {e}")
        
        return []
    
    def _enhanced_technical_forecast(self, current_price, change_24h, symbol):
        """Enhanced technical analysis for better predictions"""
        try:
            # Get more historical data for better analysis
            prices, volumes = multi_asset.get_historical_data(symbol, 50)
            
            # Calculate multiple technical indicators
            sma_short = np.mean(prices[-5:]) if len(prices) >= 5 else current_price
            sma_long = np.mean(prices[-20:]) if len(prices) >= 20 else current_price
            
            # RSI calculation (simplified)
            price_changes = np.diff(prices[-14:]) if len(prices) >= 14 else [change_24h]
            gains = np.mean([x for x in price_changes if x > 0]) if any(x > 0 for x in price_changes) else 0
            losses = abs(np.mean([x for x in price_changes if x < 0])) if any(x < 0 for x in price_changes) else 1
            rsi = 100 - (100 / (1 + gains / losses)) if losses > 0 else 50
            
            # Volatility
            volatility = np.std(prices[-10:]) / np.mean(prices[-10:]) * 100 if len(prices) >= 10 else abs(change_24h)
            
            # Enhanced forecast logic
            trend_strength = (sma_short - sma_long) / sma_long * 100 if sma_long > 0 else 0
            momentum_score = change_24h + trend_strength
            
            # Confidence based on multiple factors
            confidence = min(95, max(55, 
                70 + abs(trend_strength) * 2 - volatility * 0.5 + 
                (10 if 30 < rsi < 70 else 0)  # RSI in normal range
            ))
            
            # Direction based on multiple signals
            if momentum_score > 1 and rsi < 70:
                direction = 'UP'
                trend_score = min(10, max(1, momentum_score))
            elif momentum_score < -1 and rsi > 30:
                direction = 'DOWN'
                trend_score = max(-10, min(-1, momentum_score))
            else:
                direction = 'HOLD'
                trend_score = 0
            
            return {
                'forecast_direction': direction,
                'confidence': int(confidence),
                'trend_score': int(trend_score)
            }
            
        except Exception as e:
            pass
            # Basic fallback
            return {
                'forecast_direction': 'UP' if change_24h > 0 else 'DOWN' if change_24h < 0 else 'HOLD',
                'confidence': min(90, max(50, 60 + abs(change_24h) * 2)),
                'trend_score': int(change_24h / 2) if abs(change_24h) > 1 else 0
            }
    
    async def _predict_with_raw_models(self, symbol, timeframe='1D'):
        """Predict using raw models (crypto/stock/macro)"""
        try:
            # Stablecoins: hardcoded $1.00 prediction
            if symbol in ['USDT', 'USDC']:
                return {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'current_price': 1.00,
                    'predicted_price': 1.00,
                    'predicted_range': '$1.00 - $1.00',
                    'range_low': 1.00,
                    'range_high': 1.00,
                    'forecast_direction': 'HOLD',
                    'confidence': 99,
                    'change_24h': 0.0,
                    'data_source': 'Stablecoin'
                }
            
            # Determine asset type and select appropriate model
            macro_symbols = ['GDP', 'CPI', 'UNEMPLOYMENT', 'FED_RATE', 'CONSUMER_CONFIDENCE']
            
            if symbol in CRYPTO_SYMBOLS and self.crypto_raw_models:
                models = self.crypto_raw_models
                asset_type = 'crypto'
                # Crypto models use lowercase for short timeframes: 1h, 4h
                # and uppercase for long timeframes: 1D, 1W, 1M
                timeframe_map = {
                    '1H': '1h', '4H': '4h',  # Normalize uppercase to lowercase for hourly
                    '1D': '1D', '1W': '1W', '1M': '1M'  # Keep uppercase for daily+
                }
                timeframe = timeframe_map.get(timeframe, timeframe)
            elif symbol in STOCK_SYMBOLS and self.stock_raw_models:
                models = self.stock_raw_models
                asset_type = 'stock'
                # Stock models available: ['5m', '15m', '30m', '60m', '4h', '1d', '1wk', '1mo']
                timeframe_map = {
                    '1H': '60m',  # Map 1H to 60m for stocks
                    '4H': '4h',   # Map 4H to 4h for stocks
                    '1D': '1d',   # Map 1D to 1d for stocks
                    '1W': '1wk',  # Map 1W to 1wk for stocks
                    '1M': '1mo'   # Map 1M to 1mo for stocks
                }
                timeframe = timeframe_map.get(timeframe, timeframe)
            elif symbol in macro_symbols and self.macro_models:
                models = self.macro_models
                asset_type = 'macro'
                timeframe = '1D'  # Macro only supports 1D
            else:
                raise Exception(f"No models available for {symbol}")
            
            # Check if symbol and timeframe exist in models
            if symbol not in models:
                raise Exception(f"Symbol {symbol} not found in {asset_type} raw models")
            
            # Handle SOL 1M fallback to ETH 1M
            if symbol == 'SOL' and timeframe == '1M' and timeframe not in models[symbol]:
                if 'ETH' in models and '1M' in models['ETH']:
                    print(f"🔄 Using ETH 1M model as fallback for SOL 1M")
                    model_data = models['ETH']['1M']
                else:
                    available = list(models[symbol].keys())
                    raise Exception(f"SOL 1M model missing and ETH 1M fallback unavailable. Available: {available}")
            elif timeframe not in models[symbol]:
                available = list(models[symbol].keys())
                raise Exception(f"Timeframe {timeframe} not found for {symbol} in {asset_type} raw models. Available: {available}")
            else:
                model_data = models[symbol][timeframe]
            
            # Check required keys - all asset types use same structure
            required_keys = ['price_model', 'high_model', 'low_model', 'confidence_model', 'scaler', 'features']
            
            if not all(key in model_data for key in required_keys):
                raise Exception(f"Incomplete model data for {symbol}:{timeframe}")
            
            # Get current market data
            current_price = await self._get_real_price(symbol)
            change_24h = await self._get_real_change(symbol)
            
            if not current_price:
                raise Exception(f"No current price available for {symbol}")
            
            # Calculate features based on asset type
            if asset_type == 'macro':
                features = await self._calculate_macro_features(symbol, current_price)
            else:
                features = await self._calculate_raw_features(symbol, current_price, asset_type)
            
            if features is None:
                raise Exception(f"Failed to calculate features for {symbol}")
            
            # Create feature vector (exact match to test files)
            feature_vector = np.zeros(len(model_data['features']))
            print(f"\n🔍 Feature Debug for {symbol}:")
            print(f"  Model expects {len(model_data['features'])} features: {model_data['features']}")
            print(f"  We calculated {len(features)} features: {list(features.keys())}")
            
            for i, feature_name in enumerate(model_data['features']):
                if feature_name in features:
                    value = features[feature_name]
                    feature_vector[i] = float(value) if not pd.isna(value) else 0.0
                else:
                    print(f"  ⚠️ Missing feature: {feature_name}")
                    feature_vector[i] = 0.0
            
            # Check for NaN in feature vector
            if np.any(np.isnan(feature_vector)):
                raise Exception(f"NaN values in feature vector for {symbol}")
            
            # Scale features and predict
            features_scaled = model_data['scaler'].transform(feature_vector.reshape(1, -1))
            
            price_change = model_data['price_model'].predict(features_scaled)[0]
            range_high = model_data['high_model'].predict(features_scaled)[0]
            range_low = model_data['low_model'].predict(features_scaled)[0]
            
            # Clip predictions based on asset type
            if asset_type == 'crypto':
                price_change = np.clip(price_change, -0.15, 0.15)
                range_high = np.clip(range_high, -0.1, 0.2)
                range_low = np.clip(range_low, -0.2, 0.1)
            elif asset_type == 'stock':
                price_change = np.clip(price_change, -0.1, 0.1)
                range_high = np.clip(range_high, -0.05, 0.1)
                range_low = np.clip(range_low, -0.1, 0.05)
            else:  # macro
                price_change = np.clip(price_change, -0.05, 0.05)
                range_low = np.clip(range_low, -0.08, 0.08)
                range_high = np.clip(range_high, -0.08, 0.08)
            
            # Get confidence from model for all asset types
            confidence = model_data['confidence_model'].predict(features_scaled)[0]
            if asset_type == 'crypto':
                confidence = np.clip(confidence, 60, 95)
            elif asset_type == 'stock':
                confidence = np.clip(confidence, 60, 95)
            else:  # macro
                confidence = np.clip(confidence, 60, 90)
            
            predicted_price = current_price * (1 + price_change)
            high_price = current_price * (1 + range_high)
            low_price = current_price * (1 + range_low)
            
            # Ensure range makes sense
            if low_price > high_price:
                low_price, high_price = high_price, low_price
            
            # Direction threshold
            if asset_type == 'crypto':
                threshold = 0.005
            elif asset_type == 'stock':
                threshold = 0.003
            else:  # macro
                threshold = 0.001
            
            if price_change > threshold:
                direction = 'UP'
            elif price_change < -threshold:
                direction = 'DOWN'
            else:
                direction = 'HOLD'
            
            # Format predicted range
            predicted_range = multi_asset.format_predicted_range(symbol, low_price, high_price)
            
            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': round(current_price, 2),
                'predicted_price': round(predicted_price, 2),
                'predicted_range': predicted_range,
                'range_low': round(low_price, 2),
                'range_high': round(high_price, 2),
                'forecast_direction': direction,
                'confidence': int(confidence),
                'change_24h': round(change_24h, 2),
                'data_source': f'{asset_type.title()} ML Model'
            }
            
            return result
            
        except Exception as e:
            raise
    
    async def _calculate_raw_features(self, symbol, current_price, asset_type):
        """Calculate raw features matching training pattern"""
        try:
            historical_prices = self._get_real_historical_prices(symbol)
            if not historical_prices or len(historical_prices) < 20:
                raise Exception(f"Insufficient historical data: {len(historical_prices) if historical_prices else 0} points")
            
            df = pd.DataFrame({'close': historical_prices})
            
            df['sma_short'] = df['close'].rolling(5).mean()
            df['sma_long'] = df['close'].rolling(20).mean()
            df['price_sma_short_ratio'] = df['close'] / df['sma_short']
            df['price_sma_long_ratio'] = df['close'] / df['sma_long']
            df['returns'] = df['close'].pct_change()
            df['returns_multi'] = df['close'].pct_change(5)
            
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            df['momentum'] = df['close'] / df['close'].shift(5)
            df['volatility'] = df['returns'].rolling(5).std()
            
            # Add volume ratio for stocks (not available for crypto from historical prices)
            if asset_type == 'stock':
                # Get volume data if available
                try:
                    volumes = self._get_real_historical_volumes(symbol)
                    if volumes and len(volumes) == len(hist_prices):
                        df['volume'] = volumes
                        df['volume_sma'] = df['volume'].rolling(5).mean()
                        df['volume_ratio'] = df['volume'] / df['volume_sma']
                    else:
                        df['volume_ratio'] = 1.0
                except:
                    df['volume_ratio'] = 1.0
            
            latest = df.iloc[-1]
            
            features = {
                'price_sma_short_ratio': float(latest['price_sma_short_ratio']) if pd.notna(latest['price_sma_short_ratio']) else 1.0,
                'price_sma_long_ratio': float(latest['price_sma_long_ratio']) if pd.notna(latest['price_sma_long_ratio']) else 1.0,
                'returns': float(latest['returns']) if pd.notna(latest['returns']) else 0.0,
                'returns_multi': float(latest['returns_multi']) if pd.notna(latest['returns_multi']) else 0.0,
                'rsi': float(latest['rsi']) if pd.notna(latest['rsi']) else 50.0,
                'momentum': float(latest['momentum']) if pd.notna(latest['momentum']) else 1.0,
                'volatility': float(latest['volatility']) if pd.notna(latest['volatility']) else 0.02
            }
            
            # Add volume_ratio for stocks
            if asset_type == 'stock':
                features['volume_ratio'] = float(latest.get('volume_ratio', 1.0)) if pd.notna(latest.get('volume_ratio', 1.0)) else 1.0
            
            return features
            
        except Exception as e:
            raise Exception(f"Feature calculation failed for {symbol}: {e}")
    
    async def _calculate_macro_features(self, symbol, current_price):
        """Calculate macro features matching training pattern"""
        try:
            from fredapi import Fred
            from datetime import datetime
            
            fred_series = {
                'GDP': 'GDP',
                'CPI': 'CPIAUCSL',
                'UNEMPLOYMENT': 'UNRATE',
                'FED_RATE': 'FEDFUNDS',
                'CONSUMER_CONFIDENCE': 'UMCSENT'
            }
            
            series_id = fred_series.get(symbol)
            if not series_id:
                raise Exception(f"Unknown macro symbol: {symbol}")
            
            fred_api_key = os.getenv('FRED_API_KEY')
            if not fred_api_key:
                raise Exception("FRED_API_KEY not found")
            
            fred = Fred(api_key=fred_api_key)
            data = fred.get_series(series_id)
            
            if data is None or len(data) < 20:
                raise Exception(f"Insufficient FRED data for {symbol}")
            
            df = pd.DataFrame({'timestamp': data.index, 'close': data.values})
            df = df.dropna().tail(20)
            
            df['change_1'] = df['close'].pct_change(1)
            df['change_4'] = df['close'].pct_change(4)
            df['ma_4'] = df['close'].rolling(4).mean()
            df['ma_12'] = df['close'].rolling(12).mean()
            df['trend'] = (df['close'] - df['ma_12']) / df['ma_12']
            df['volatility'] = df['change_1'].rolling(12).std()
            df['quarter'] = df['timestamp'].dt.quarter
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
                'quarter': int(latest['quarter'])
            }
            
            return features
            
        except Exception as e:
            raise Exception(f"Macro feature calculation failed for {symbol}: {e}")