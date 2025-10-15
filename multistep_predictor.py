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
        """Generate iterative forward predictions"""
        try:
            # Get base prediction
            base_pred = await self.ml_model.predict(symbol, timeframe)
            if not base_pred:
                return None
            
            current_price = base_pred['current_price']
            predictions = []
            timestamps = []
            
            # Time delta based on forecast route mapping
            # 1D forecast = 12 hourly steps, 1W = 7 daily steps, etc.
            time_deltas = {
                '1h': timedelta(hours=1),
                '4h': timedelta(hours=4),
                '1D': timedelta(hours=1),  # 1D forecast uses hourly steps
                '1W': timedelta(days=1),   # 1W forecast uses daily steps
                '1M': timedelta(weeks=1)   # 1M forecast uses weekly steps
            }
            
            delta = time_deltas.get(timeframe, timedelta(hours=1))
            current_time = datetime.now()
            
            # Use base prediction as first step
            predictions.append(base_pred['predicted_price'])
            timestamps.append(current_time.isoformat())
            
            # Generate remaining steps with decay
            for i in range(1, num_steps):
                # Apply decay to prediction confidence
                decay_factor = 0.95 ** i
                price_change = (predictions[-1] - current_price) / current_price
                
                # Dampen future predictions
                next_price = current_price * (1 + price_change * decay_factor)
                predictions.append(round(next_price, 2))
                
                current_time += delta
                timestamps.append(current_time.isoformat())
            
            return {
                'prices': predictions,
                'timestamps': timestamps,
                'base_confidence': base_pred['confidence']
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
