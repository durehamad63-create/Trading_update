"""
Multi-Step Prediction Service with Caching
"""
import asyncio
import numpy as np
from datetime import datetime, timedelta
from utils.cache_manager import CacheManager, CacheKeys

class MultiStepPredictor:
    def __init__(self, ml_model):
        self.ml_model = ml_model
        self.cache_ttl = 300  # 5 minutes for multi-step predictions
    
    async def get_multistep_forecast(self, symbol, timeframe, num_steps):
        """Get cached or generate multi-step predictions"""
        cache_key = f"multistep:{symbol}:{timeframe}:{num_steps}"
        cached = CacheManager.get_cache(cache_key)
        
        if cached:
            return cached
        
        # Generate new multi-step predictions
        result = await self._generate_multistep(symbol, timeframe, num_steps)
        
        if result:
            CacheManager.set_cache(cache_key, result, self.cache_ttl)
        
        return result
    
    async def _generate_multistep(self, symbol, timeframe, num_steps):
        """Generate autoregressive multi-step predictions"""
        try:
            # Get historical prices for feature calculation
            historical_prices = self.ml_model._get_real_historical_prices(symbol)
            if not historical_prices or len(historical_prices) < 20:
                return None
            
            # Time deltas (uppercase timeframes)
            time_deltas = {
                '1H': timedelta(hours=1),
                '4H': timedelta(hours=4),
                '1D': timedelta(days=1),
                '1W': timedelta(weeks=1),
                '1M': timedelta(days=30)
            }
            
            delta = time_deltas.get(timeframe, timedelta(hours=1))
            current_time = datetime.now()
            
            predictions = []
            timestamps = []
            
            # Start with real historical data
            price_history = list(historical_prices[-30:])  # Last 30 points
            
            # Autoregressive prediction: each step uses previous predictions
            for i in range(num_steps):
                try:
                    # Calculate features from current history (real + predicted)
                    import pandas as pd
                    df = pd.DataFrame({'close': price_history})
                    
                    df['sma_5'] = df['close'].rolling(5, min_periods=1).mean()
                    df['sma_20'] = df['close'].rolling(20, min_periods=1).mean()
                    df['price_sma5_ratio'] = df['close'] / df['sma_5']
                    df['price_sma20_ratio'] = df['close'] / df['sma_20']
                    df['returns'] = df['close'].pct_change()
                    df['returns_5'] = df['close'].pct_change(5)
                    
                    delta_col = df['close'].diff()
                    gain = (delta_col.where(delta_col > 0, 0)).rolling(14, min_periods=1).mean()
                    loss = (-delta_col.where(delta_col < 0, 0)).rolling(14, min_periods=1).mean()
                    rs = gain / loss
                    df['rsi'] = 100 - (100 / (1 + rs))
                    
                    df['momentum_7'] = df['close'] / df['close'].shift(7)
                    df['volatility'] = df['returns'].rolling(10, min_periods=1).std()
                    
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
                    
                    # Get model and predict
                    from config.symbols import CRYPTO_SYMBOLS, STOCK_SYMBOLS
                    if symbol in CRYPTO_SYMBOLS:
                        models = self.ml_model.crypto_raw_models
                    elif symbol in STOCK_SYMBOLS:
                        models = self.ml_model.stock_raw_models
                    else:
                        return None
                    
                    if not models or symbol not in models or timeframe not in models[symbol]:
                        return None
                    
                    model_data = models[symbol][timeframe]
                    
                    # Create feature vector
                    feature_vector = np.zeros(len(model_data['features']))
                    for j, feature_name in enumerate(model_data['features']):
                        feature_vector[j] = features.get(feature_name, 0.0)
                    
                    # Scale and predict
                    features_scaled = model_data['scaler'].transform(feature_vector.reshape(1, -1))
                    price_change = model_data['price_model'].predict(features_scaled)[0]
                    
                    # Clip prediction
                    if symbol in CRYPTO_SYMBOLS:
                        price_change = np.clip(price_change, -0.15, 0.15)
                    else:
                        price_change = np.clip(price_change, -0.1, 0.1)
                    
                    # Calculate next price
                    current_price = price_history[-1]
                    next_price = current_price * (1 + price_change)
                    
                    predictions.append(round(next_price, 2))
                    timestamps.append((current_time + delta * (i + 1)).isoformat())
                    
                    # Append prediction to history for next iteration
                    price_history.append(next_price)
                    
                except Exception as e:
                    # Fallback: use last prediction
                    if predictions:
                        predictions.append(predictions[-1])
                        timestamps.append((current_time + delta * (i + 1)).isoformat())
                    else:
                        return None
            
            return {
                'prices': predictions,
                'timestamps': timestamps,
                'base_confidence': 75
            }
            
        except Exception as e:
            return None

# Global instance
multistep_predictor = None

def init_multistep_predictor(ml_model):
    """Initialize multi-step predictor"""
    global multistep_predictor
    multistep_predictor = MultiStepPredictor(ml_model)
    return multistep_predictor
