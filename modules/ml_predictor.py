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
        
        # Load models directly - REQUIRED
        try:
            import joblib
            import yfinance as yf
            import requests
            
            # Use new specialized model
            model_path = os.path.join('models', 'specialized_trading_model.pkl')
            
            # Download model from Google Drive if not exists locally
            if not os.path.exists(model_path):
                self._download_model_from_drive(model_path)
            
            # Verify model file exists and is valid
            if not os.path.exists(model_path):
                raise Exception(f"Model file not found: {model_path}")
            
            file_size = os.path.getsize(model_path)
            if file_size < 1000:
                raise Exception(f"Model file too small ({file_size} bytes), download may have failed")
            
            self.mobile_model = joblib.load(model_path)
            print(f"✅ Specialized model loaded from {model_path} ({file_size} bytes)")
            
            # Store all timeframe models
            if hasattr(self.mobile_model, 'models'):
                self.timeframe_models = self.mobile_model.models
                # Default to 1D model for backward compatibility
                if 'Crypto' in self.timeframe_models:
                    crypto_1d = self.timeframe_models['Crypto'].get('1D', {})
                    self.xgb_model = crypto_1d.get('model')
                    self.model_features = crypto_1d.get('features', [])
            else:
                raise Exception("Specialized model structure not found")
        except Exception as e:
            raise Exception(f"❌ CRITICAL: Specialized model failed to load: {str(e)}")
        
        # Load new raw models - ALWAYS use raw models for crypto/stocks
        self.use_legacy_model = os.getenv('USE_LEGACY_MODEL', 'false').lower() == 'true'
        self.use_raw_models = True  # Always enabled
        self.raw_model_priority = True  # Always prioritize raw models
        
        self.crypto_raw_models = None
        self.stock_raw_models = None
        
        # Load crypto raw models (REQUIRED for crypto predictions)
        crypto_model_path = 'models/crypto_raw/crypto_raw_models.pkl'
        if not os.path.exists(crypto_model_path):
            raise Exception(f"❌ CRITICAL: Crypto raw models not found at {crypto_model_path}\n🛠️ Run 'python train_crypto_model_raw.py' to train models")
        
        try:
            self.crypto_raw_models = joblib.load(crypto_model_path)
            print(f"✅ Crypto raw models loaded from {crypto_model_path}")
        except Exception as e:
            raise Exception(f"❌ CRITICAL: Crypto raw models failed to load: {e}")
        
        # Load stock raw models (REQUIRED for stock predictions)
        stock_model_path = 'models/stock_raw/stock_raw_models.pkl'
        if not os.path.exists(stock_model_path):
            raise Exception(f"❌ CRITICAL: Stock raw models not found at {stock_model_path}\n🛠️ Run 'python train_stock_model_raw.py' to train models")
        
        try:
            self.stock_raw_models = joblib.load(stock_model_path)
            print(f"✅ Stock raw models loaded from {stock_model_path}")
        except Exception as e:
            raise Exception(f"❌ CRITICAL: Stock raw models failed to load: {e}")
        
        print(f"🔧 Model Configuration: Legacy={self.use_legacy_model}, Raw={self.use_raw_models}, Priority={'Raw' if self.raw_model_priority else 'Legacy'}")
        
        # Use centralized cache manager
        from utils.cache_manager import CacheManager, CacheKeys, CacheTTL
        self.cache_manager = CacheManager
        self.cache_keys = CacheKeys
        self.cache_ttl = CacheTTL
        
        print(f"🔧 ML Predictor initialized: min_interval={self.min_request_interval*1000:.0f}ms, cache_ttl={self.cache_ttl}s")
    
    def _download_model_from_drive(self, model_path):
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
            if 'drive.google.com/uc?id=' in drive_url:
                file_id = drive_url.split('id=')[1].split('&')[0]
            else:
                raise Exception(f"Invalid Google Drive URL format: {drive_url}")
            
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
        """Generate real model prediction with Redis caching"""
        import time
        start_time = time.time()
        
        cache_key = self.cache_keys.prediction(symbol, timeframe)
        cached_data = self.cache_manager.get_cache(cache_key)
        
        if cached_data:
            return cached_data
        
        # Rate limiting for real-time updates
        if symbol in self.last_request_time:
            time_since_last = start_time - self.last_request_time[symbol]
            if time_since_last < self.min_request_interval:
                if symbol in self.prediction_cache:
                    cache_time, cached_result = self.prediction_cache[symbol]
                    if start_time - cache_time < 2.0:
                        return cached_result
                await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time[symbol] = start_time
        
        async with self.prediction_semaphore:
            
            # Determine asset class
            asset_class = 'Crypto' if symbol in CRYPTO_SYMBOLS else 'Stocks' if symbol in STOCK_SYMBOLS else 'Macro'
            
            # Use raw models for crypto and stocks
            if asset_class in ['Crypto', 'Stocks']:
                try:
                    raw_result = await self._predict_with_raw_models(symbol, timeframe)
                    if raw_result:
                        ttl = self.cache_ttl.PREDICTION_HOT if symbol in ['BTC', 'ETH', 'NVDA', 'AAPL'] else self.cache_ttl.PREDICTION_NORMAL
                        self.cache_manager.set_cache(cache_key, raw_result, ttl)
                        self.prediction_cache[symbol] = (start_time, raw_result)
                        return raw_result
                except Exception as raw_e:
                    raise Exception(f"Raw model prediction failed for {symbol}: {raw_e}")
            
            # Use specialized model only for macro indicators
            if asset_class != 'Macro':
                raise Exception(f"No model available for {symbol} ({asset_class})")
            
            try:
                # Get real data with faster timeout
                price_start = time.time()
                try:
                    real_price = await asyncio.wait_for(self._get_real_price(symbol), timeout=2.0)
                    if not real_price:
                        raise Exception("No price data available")
                except (asyncio.TimeoutError, Exception) as e:
                    raise Exception(f"Failed to get real price data for {symbol}: {e}")
                
                current_price = real_price
                try:
                    change_24h = await asyncio.wait_for(self._get_real_change(symbol), timeout=1.0)
                    data_source = 'ML Analysis'
                except (asyncio.TimeoutError, Exception) as e:
                    raise Exception(f"Failed to get 24h change for {symbol}: {e}")
            
                real_prices = self._get_real_historical_prices(symbol)
                if not real_prices or len(real_prices) < 10:
                    raise Exception(f"Insufficient historical price data for {symbol}: need at least 10 points, got {len(real_prices) if real_prices else 0}")
            
                # Ensure current price is the latest
                real_prices.append(current_price)
                real_prices = real_prices[-30:]  # Keep last 30 days
            
                # Create feature vector using real market data
                features = np.zeros(len(self.model_features))
            
                # Calculate real returns from historical prices
                if len(real_prices) >= 2:
                    log_return = np.log(real_prices[-1] / real_prices[-2])
                else:
                    log_return = 0.0
            
                # Calculate real technical indicators from historical data
                returns = np.diff(np.log(real_prices)) if len(real_prices) >= 2 else [0]
                volatility = np.std(returns) if len(returns) >= 2 else 0.015
            
                # Calculate RSI from real price movements
                def calculate_rsi(prices, period=14):
                    if len(prices) < period + 1:
                        return 50.0
                    deltas = np.diff(prices)
                    gains = np.where(deltas > 0, deltas, 0)
                    losses = np.where(deltas < 0, -deltas, 0)
                    avg_gain = np.mean(gains[-period:])
                    avg_loss = np.mean(losses[-period:])
                    if avg_loss == 0:
                        return 100.0
                    rs = avg_gain / avg_loss
                    return 100 - (100 / (1 + rs))
                
                rsi = calculate_rsi(real_prices)
            
                # Fill features with real market-based values
                feature_idx = 0
                for feature_name in self.model_features:
                    if feature_idx >= len(features):
                        break
                        
                    if 'Return_Lag_1' in feature_name:
                        features[feature_idx] = returns[-1] if len(returns) >= 1 else 0
                    elif 'Return_Lag_3' in feature_name:
                        features[feature_idx] = returns[-3] if len(returns) >= 3 else 0
                    elif 'Return_Lag_5' in feature_name:
                        features[feature_idx] = returns[-5] if len(returns) >= 5 else 0
                    elif 'Log_Return' in feature_name:
                        features[feature_idx] = log_return
                    elif 'Volatility' in feature_name:
                        features[feature_idx] = volatility
                    elif 'RSI' in feature_name:
                        features[feature_idx] = rsi
                    elif 'Price_Momentum' in feature_name:
                        features[feature_idx] = np.mean(returns[-5:]) if len(returns) >= 5 else 0
                    elif feature_name == 'High':
                        features[feature_idx] = np.max(real_prices[-5:]) if len(real_prices) >= 5 else current_price
                    elif 'Close_Lag_1' in feature_name:
                        features[feature_idx] = real_prices[-2] if len(real_prices) >= 2 else current_price
                    elif 'BB_Width' in feature_name:
                        features[feature_idx] = volatility * 2
                    elif 'VIX' in feature_name:
                        features[feature_idx] = volatility * 100
                    elif 'SPY' in feature_name:
                        features[feature_idx] = np.mean(returns[-5:]) if len(returns) >= 5 else 0
                    else:
                        if feature_idx == 0:
                            features[feature_idx] = current_price
                        elif feature_idx == 1:
                            features[feature_idx] = change_24h
                        elif feature_idx == 2:
                            features[feature_idx] = np.mean(real_prices[-5:]) if len(real_prices) >= 5 else current_price
                        elif feature_idx == 3:
                            features[feature_idx] = np.mean(real_prices[-10:]) if len(real_prices) >= 10 else current_price
                        elif feature_idx == 4:
                            features[feature_idx] = volatility
                        else:
                            idx = min(feature_idx - 4, len(real_prices) - 1)
                            features[feature_idx] = real_prices[-idx-1] if idx >= 0 else current_price
                    
                    feature_idx += 1
            
                # Get timeframe-specific specialized model (macro only)
                
                if hasattr(self, 'timeframe_models') and asset_class in self.timeframe_models:
                    timeframe_model_data = self.timeframe_models[asset_class].get(timeframe, {})
                    if timeframe_model_data and 'model' in timeframe_model_data:
                        model_to_use = timeframe_model_data['model']
                    else:
                        model_to_use = self.xgb_model
                else:
                    model_to_use = self.xgb_model
            
                # Real ML prediction with timeframe-specific model
                xgb_prediction = model_to_use.predict(features.reshape(1, -1))[0]
                predicted_price = current_price * (1 + xgb_prediction)
                
                # Dynamic confidence based on multiple factors
                volatility_factor = abs(change_24h) * 0.5
                prediction_strength = abs(xgb_prediction) * 200
                
                # Base confidence varies by symbol type
                if symbol in ['BTC', 'ETH']:
                    base_confidence = 75
                elif symbol in ['USDT', 'USDC']:
                    base_confidence = 90
                else:
                    base_confidence = 70
                
                # Confidence based on real volatility (lower volatility = higher confidence)
                confidence = base_confidence + prediction_strength - volatility_factor - (volatility * 100)
                confidence = min(95, max(50, confidence))
                
                # logging.info(f"🔥 REAL DATA: {symbol} price=${current_price} from API, ML prediction={xgb_prediction:.4f}")
                
                forecast = {
                    'forecast_direction': 'UP' if xgb_prediction > 0.01 else 'DOWN' if xgb_prediction < -0.01 else 'HOLD',
                    'confidence': int(confidence),
                    'trend_score': int(xgb_prediction * 100)
                }
                
                result = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'current_price': round(current_price, 2),
                    'predicted_price': round(predicted_price, 2),
                    'forecast_direction': forecast['forecast_direction'],
                    'confidence': forecast['confidence'],
                    'change_24h': round(change_24h, 2),
                    'data_source': data_source
                }
                
                ttl = self.cache_ttl.PREDICTION_HOT if symbol in ['BTC', 'ETH', 'NVDA', 'AAPL'] else self.cache_ttl.PREDICTION_NORMAL
                self.cache_manager.set_cache(cache_key, result, ttl)
                self.prediction_cache[symbol] = (start_time, result)
                return result
            
            except Exception as e:
                raise Exception(f"Specialized model prediction failed for {symbol}: {str(e)}")
    
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
    
    def _get_real_historical_prices(self, symbol):
        """Get real historical prices - Binance for crypto, YFinance for stocks"""
        crypto_symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOGE', 'USDT', 'USDC', 'TRX']
        
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
                        # logging.info(f"🔥 REAL BINANCE HISTORY: {symbol} got {len(prices)} price points")
                        return prices
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
        """Predict using new raw models (crypto/stock)"""
        try:
            # Determine asset type and select appropriate model
            if symbol in CRYPTO_SYMBOLS and self.crypto_raw_models:
                models = self.crypto_raw_models
                asset_type = 'crypto'
            elif symbol in STOCK_SYMBOLS and self.stock_raw_models:
                models = self.stock_raw_models
                asset_type = 'stock'
            else:
                raise Exception(f"No raw models available for {symbol}")
            
            # Check if symbol and timeframe exist in models
            if symbol not in models:
                raise Exception(f"Symbol {symbol} not found in {asset_type} raw models")
            
            if timeframe not in models[symbol]:
                raise Exception(f"Timeframe {timeframe} not found for {symbol} in {asset_type} raw models")
            
            model_data = models[symbol][timeframe]
            required_keys = ['price_model', 'high_model', 'low_model', 'confidence_model', 'scaler', 'features']
            if not all(key in model_data for key in required_keys):
                raise Exception(f"Incomplete model data for {symbol}:{timeframe}")
            
            # Get current market data
            current_price = await self._get_real_price(symbol)
            change_24h = await self._get_real_change(symbol)
            
            if not current_price:
                raise Exception(f"No current price available for {symbol}")
            
            # Calculate raw features for the model
            features = await self._calculate_raw_features(symbol, current_price, asset_type)
            if features is None:
                raise Exception(f"Failed to calculate features for {symbol}")
            
            # Create feature vector (exact match to test files)
            feature_vector = np.zeros(len(model_data['features']))
            for i, feature_name in enumerate(model_data['features']):
                if feature_name in features:
                    value = features[feature_name]
                    feature_vector[i] = float(value) if not pd.isna(value) else 0.0
                else:
                    feature_vector[i] = 0.0
            
            # Check for NaN in feature vector
            if np.any(np.isnan(feature_vector)):
                raise Exception(f"NaN values in feature vector for {symbol}")
            
            # Scale features and predict (exact match to test files)
            features_scaled = model_data['scaler'].transform(feature_vector.reshape(1, -1))
            
            price_change = model_data['price_model'].predict(features_scaled)[0]
            range_high = model_data['high_model'].predict(features_scaled)[0]
            range_low = model_data['low_model'].predict(features_scaled)[0]
            confidence = model_data['confidence_model'].predict(features_scaled)[0]
            
            # Clip predictions (exact match to test files)
            if asset_type == 'crypto':
                price_change = np.clip(price_change, -0.15, 0.15)  # ±15% for crypto
                range_high = np.clip(range_high, -0.1, 0.2)        # -10% to +20%
                range_low = np.clip(range_low, -0.2, 0.1)          # -20% to +10%
            else:  # stocks
                price_change = np.clip(price_change, -0.1, 0.1)    # ±10% for stocks
                range_high = np.clip(range_high, -0.05, 0.1)       # -5% to +10%
                range_low = np.clip(range_low, -0.1, 0.05)         # -10% to +5%
            
            confidence = np.clip(confidence, 60, 95)
            
            predicted_price = current_price * (1 + price_change)
            high_price = current_price * (1 + range_high)
            low_price = current_price * (1 + range_low)
            
            # Ensure range makes sense
            if low_price > high_price:
                low_price, high_price = high_price, low_price
            
            # Direction threshold (exact match to test files)
            threshold = 0.005 if asset_type == 'crypto' else 0.003
            if price_change > threshold:
                direction = 'UP'
            elif price_change < -threshold:
                direction = 'DOWN'
            else:
                direction = 'HOLD'
            
            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': round(current_price, 2),
                'predicted_price': round(predicted_price, 2),
                'range_low': round(low_price, 2),
                'range_high': round(high_price, 2),
                'forecast_direction': direction,
                'confidence': int(confidence),
                'change_24h': round(change_24h, 2),
                'data_source': f'Raw {asset_type.title()} Model'
            }
            
            return result
            
        except Exception as e:
            raise
    
    async def _calculate_raw_features(self, symbol, current_price, asset_type):
        """Calculate raw features matching training pattern"""
        try:
            # Get historical prices
            historical_prices = self._get_real_historical_prices(symbol)
            if not historical_prices or len(historical_prices) < 20:
                raise Exception(f"Insufficient historical data: {len(historical_prices) if historical_prices else 0} points")
            
            # Create DataFrame matching training format
            import pandas as pd
            df = pd.DataFrame({'close': historical_prices})
            
            # Calculate raw features (exact match to training)
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['price_sma5_ratio'] = df['close'] / df['sma_5']
            df['price_sma20_ratio'] = df['close'] / df['sma_20']
            
            df['returns'] = df['close'].pct_change()
            df['returns_5'] = df['close'].pct_change(5)
            
            # RSI calculation (exact match to training)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            df['momentum_7'] = df['close'] / df['close'].shift(7)
            df['volatility'] = df['returns'].rolling(10).std()
            
            # Get latest values
            latest = df.iloc[-1]
            
            # Handle NaN values (exact match to test files)
            features = {
                'price_sma5_ratio': float(latest['price_sma5_ratio']) if pd.notna(latest['price_sma5_ratio']) else 1.0,
                'price_sma20_ratio': float(latest['price_sma20_ratio']) if pd.notna(latest['price_sma20_ratio']) else 1.0,
                'returns': float(latest['returns']) if pd.notna(latest['returns']) else 0.0,
                'returns_5': float(latest['returns_5']) if pd.notna(latest['returns_5']) else 0.0,
                'rsi': float(latest['rsi']) if pd.notna(latest['rsi']) else 50.0,
                'momentum_7': float(latest['momentum_7']) if pd.notna(latest['momentum_7']) else 1.0,
                'volatility': float(latest['volatility']) if pd.notna(latest['volatility']) else 0.02
            }
            
            return features
            
        except Exception as e:
            raise Exception(f"Feature calculation failed for {symbol}: {e}")