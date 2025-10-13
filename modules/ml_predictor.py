"""
ML Prediction Module
"""
import asyncio
import logging
import time
import pickle
import os
import numpy as np
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
        self.min_request_interval = 0.001  # 1ms for real-time
        self.xgb_model = None
        self.prediction_cache = {}
        self.cache_ttl = 1  # 1 second cache for real-time updates
        
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
            print(f"Model loaded successfully from {model_path} ({file_size} bytes)")
            pass
            
            # Extract XGBoost model for compatibility
            if hasattr(self.mobile_model, 'models') and 'Crypto' in self.mobile_model.models:
                crypto_1d = self.mobile_model.models['Crypto'].get('1D', {})
                self.xgb_model = crypto_1d.get('model')
                self.model_features = crypto_1d.get('features', [])
                pass
            else:
                raise Exception("Specialized model structure not found")
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            if "timeout" in str(e).lower() or "60" in str(e):
                print("Timeout detected - trying gdown alternative...")
                try:
                    import subprocess
                    import sys
                    subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True, capture_output=True)
                    import gdown
                    file_id = "10uBJLKsijJHDFBOhCsyFFi-1DAhamjez"
                    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False, fuzzy=True)
                    self.mobile_model = joblib.load(model_path)
                    print("Model loaded successfully using gdown")
                except Exception as alt_error:
                    raise Exception(f"Cannot start: Both download methods failed - {str(e)}")
            else:
                raise Exception(f"Cannot start: {str(e)}")
        
        # Use centralized cache manager
        from utils.cache_manager import CacheManager, CacheKeys, CacheTTL
        self.cache_manager = CacheManager
        self.cache_keys = CacheKeys
        self.cache_ttl = CacheTTL
    
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
    
    async def predict(self, symbol):
        """Generate real model prediction with Redis caching"""
        import time
        current_time = time.time()
        cache_key = self.cache_keys.prediction(symbol)
        
        # Check cache using centralized manager
        cached_data = self.cache_manager.get_cache(cache_key)
        if cached_data:
            return cached_data
        
        # Minimal rate limiting for real-time updates
        if symbol in self.last_request_time:
            time_since_last = current_time - self.last_request_time[symbol]
            if time_since_last < 0.1:  # Only 100ms rate limit
                # Return memory cached result if available
                if symbol in self.prediction_cache:
                    cache_time, cached_result = self.prediction_cache[symbol]
                    if current_time - cache_time < 0.5:  # 500ms cache
                        return cached_result
        
        self.last_request_time[symbol] = current_time
        
        try:
            # Get real data with faster timeout
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
            
            # Get real historical prices for feature engineering
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
            
            # Real ML prediction
            xgb_prediction = self.xgb_model.predict(features.reshape(1, -1))[0]
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
                'current_price': round(current_price, 2),
                'predicted_price': round(predicted_price, 2),
                'forecast_direction': forecast['forecast_direction'],
                'confidence': forecast['confidence'],
                'change_24h': round(change_24h, 2),
                'data_source': data_source
            }
            
            # Cache using centralized manager with hot symbol priority
            ttl = self.cache_ttl.PREDICTION_HOT if symbol in ['BTC', 'ETH', 'NVDA', 'AAPL'] else self.cache_ttl.PREDICTION_NORMAL
            self.cache_manager.set_cache(cache_key, result, ttl)
            
            # Cache the result in memory
            self.prediction_cache[symbol] = (current_time, result)
            
            return result
            
        except Exception as e:
            raise Exception(f"PREDICTION FAILED: Cannot generate prediction without real market data for {symbol}: {str(e)}")
    
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