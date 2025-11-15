"""Centralized cache key management with Redis + Memory fallback"""
import os
import redis
import json
import time
from dotenv import load_dotenv

load_dotenv()

class CacheTTL:
    """Centralized TTL configuration with priority system"""
    PRICE_CRYPTO = 300      # 5 minutes (was 60)
    PRICE_STOCK = 300       # 5 minutes (was 60)
    PRICE_MACRO = 1800      # 30 minutes (was 600)
    
    # Prediction cache TTLs by priority
    PREDICTION_HOT = 120    # BTC, ETH, NVDA, AAPL - 2 minutes (was 30)
    PREDICTION_NORMAL = 300 # Other major assets - 5 minutes (was 60)
    PREDICTION_COLD = 600   # Less active assets - 10 minutes (was 120)
    
    CHART_DATA = 900
    WEBSOCKET_HISTORY = 600

class PredictionPriority:
    """Prediction priority and update intervals"""
    HOT_SYMBOLS = ['BTC', 'ETH', 'NVDA', 'AAPL']
    NORMAL_SYMBOLS = ['MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BNB', 'SOL', 'XRP']
    
    # Update intervals (how often to generate fresh predictions)
    HOT_UPDATE_INTERVAL = 60      # 60s for hot symbols
    NORMAL_UPDATE_INTERVAL = 120  # 120s for normal symbols
    COLD_UPDATE_INTERVAL = 300    # 300s for cold symbols
    
    @classmethod
    def get_cache_ttl(cls, symbol):
        """Get cache TTL for symbol"""
        if symbol in cls.HOT_SYMBOLS:
            return CacheTTL.PREDICTION_HOT
        elif symbol in cls.NORMAL_SYMBOLS:
            return CacheTTL.PREDICTION_NORMAL
        else:
            return CacheTTL.PREDICTION_COLD
    
    @classmethod
    def get_update_interval(cls, symbol):
        """Get update interval for symbol"""
        if symbol in cls.HOT_SYMBOLS:
            return cls.HOT_UPDATE_INTERVAL
        elif symbol in cls.NORMAL_SYMBOLS:
            return cls.NORMAL_UPDATE_INTERVAL
        else:
            return cls.COLD_UPDATE_INTERVAL

class CacheManager:
    _redis_client = None
    _memory_cache = {}  # Fallback memory cache
    _cache_timestamps = {}  # Track TTL for memory cache
    _max_memory_cache_size = 1000  # Limit memory cache size
    
    @classmethod
    def get_redis_client(cls):
        if cls._redis_client is None:
            try:
                redis_host = os.getenv('REDIS_HOST', 'localhost')
                redis_port = int(os.getenv('REDIS_PORT', '6379'))
                redis_password = os.getenv('REDIS_PASSWORD')
                
                cls._redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=0,
                    password=redis_password if redis_password else None,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                cls._redis_client.ping()
            except Exception:
                cls._redis_client = None
        return cls._redis_client
    
    @classmethod
    def set_cache(cls, key, value, ttl=60):
        """Set cache with Redis primary, memory fallback"""
        # Enforce memory cache size limit (LRU eviction)
        if len(cls._memory_cache) >= cls._max_memory_cache_size:
            # Remove oldest entry
            oldest_key = min(cls._cache_timestamps.keys(), 
                           key=lambda k: cls._cache_timestamps[k][0])
            cls._memory_cache.pop(oldest_key, None)
            cls._cache_timestamps.pop(oldest_key, None)
        
        # Store in memory cache as fallback
        cls._memory_cache[key] = value
        cls._cache_timestamps[key] = (time.time(), ttl)
        
        # Try Redis
        try:
            client = cls.get_redis_client()
            if client:
                serialized = json.dumps(value, default=str)
                client.setex(key, ttl, serialized)
        except Exception as e:
            import logging
            logging.warning(f"Redis cache set failed for {key}: {e}")
    
    @classmethod
    def get_cache(cls, key):
        """Get cache with Redis primary, memory fallback"""
        # Try Redis first
        try:
            client = cls.get_redis_client()
            if client:
                data = client.get(key)
                if data:
                    result = json.loads(data)
                    return result
        except Exception as e:
            import logging
            logging.debug(f"Redis cache get failed for {key}: {e}")
        
        # Fallback to memory cache
        if key in cls._memory_cache:
            timestamp, ttl = cls._cache_timestamps.get(key, (0, 0))
            if time.time() - timestamp < ttl:
                return cls._memory_cache[key]
            else:
                # Expired, remove
                cls._memory_cache.pop(key, None)
                cls._cache_timestamps.pop(key, None)
        
        return None
    
    @classmethod
    def should_update_prediction(cls, symbol, timeframe='1D'):
        """Check if prediction should be updated based on priority"""
        from utils.cache_manager import PredictionPriority
        
        timestamp_key = f"pred_time:{symbol}:{timeframe}"
        last_update = cls.get_cache(timestamp_key)
        
        if not last_update:
            return True
        
        import time
        time_since_update = time.time() - last_update
        update_interval = PredictionPriority.get_update_interval(symbol)
        
        return time_since_update >= update_interval
    
    @classmethod
    def mark_prediction_updated(cls, symbol, timeframe='1D'):
        """Mark prediction as updated"""
        import time
        timestamp_key = f"pred_time:{symbol}:{timeframe}"
        cls.set_cache(timestamp_key, time.time(), ttl=600)
    
    @classmethod
    def delete_cache(cls, key):
        """Delete cache entry from both Redis and memory"""
        # Remove from memory
        cls._memory_cache.pop(key, None)
        cls._cache_timestamps.pop(key, None)
        
        # Remove from Redis
        try:
            client = cls.get_redis_client()
            if client:
                client.delete(key)
        except Exception as e:
            import logging
            logging.warning(f"Redis cache delete failed for {key}: {e}")
    
    @classmethod
    def clear_pattern(cls, pattern):
        """Clear cache entries matching pattern (e.g., 'prediction:BTC:*')"""
        # Clear from memory
        keys_to_remove = [k for k in cls._memory_cache.keys() if pattern.replace('*', '') in k]
        for key in keys_to_remove:
            cls._memory_cache.pop(key, None)
            cls._cache_timestamps.pop(key, None)
        
        # Clear from Redis
        try:
            client = cls.get_redis_client()
            if client:
                for key in client.scan_iter(match=pattern):
                    client.delete(key)
        except Exception as e:
            import logging
            logging.warning(f"Redis pattern clear failed for {pattern}: {e}")

# Export PredictionPriority for use in other modules
__all__ = ['CacheManager', 'CacheKeys', 'CacheTTL', 'PredictionPriority']

class CacheKeys:
    # Unified key patterns
    @staticmethod
    def price(symbol, asset_type):
        return f"price:{asset_type}:{symbol}"
    
    @staticmethod
    def prediction(symbol, timeframe='1D'):
        return f"prediction:{symbol}:{timeframe}"
    
    @staticmethod
    def prediction_timestamp(symbol, timeframe='1D'):
        """Track last prediction generation time"""
        return f"pred_time:{symbol}:{timeframe}"
    
    @staticmethod
    def market_summary(class_filter="all"):
        return f"market:{class_filter}"
    
    @staticmethod
    def chart_data(symbol, timeframe):
        return f"chart:{symbol}:{timeframe}"
    
    @staticmethod
    def websocket_history(symbol, timeframe):
        return f"ws_history:{symbol}:{timeframe}"