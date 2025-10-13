"""Centralized cache key management with Redis + Memory fallback"""
import os
import redis
import json
import time
from dotenv import load_dotenv

load_dotenv()

class CacheTTL:
    """Centralized TTL configuration"""
    PRICE_CRYPTO = 30
    PRICE_STOCK = 30
    PRICE_MACRO = 300
    PREDICTION_HOT = 1
    PREDICTION_NORMAL = 3
    CHART_DATA = 600
    WEBSOCKET_HISTORY = 300

class CacheManager:
    _redis_client = None
    _memory_cache = {}  # Fallback memory cache
    _cache_timestamps = {}  # Track TTL for memory cache
    
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
        # Always store in memory cache as fallback
        cls._memory_cache[key] = value
        cls._cache_timestamps[key] = (time.time(), ttl)
        
        # Try Redis
        try:
            client = cls.get_redis_client()
            if client:
                client.setex(key, ttl, json.dumps(value, default=str))
        except Exception:
            pass  # Memory cache already set
    
    @classmethod
    def get_cache(cls, key):
        """Get cache with Redis primary, memory fallback"""
        # Try Redis first
        try:
            client = cls.get_redis_client()
            if client:
                data = client.get(key)
                if data:
                    return json.loads(data)
        except Exception:
            pass
        
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

class CacheKeys:
    # Unified key patterns
    @staticmethod
    def price(symbol, asset_type):
        return f"price:{asset_type}:{symbol}"
    
    @staticmethod
    def prediction(symbol, timeframe='1D'):
        return f"prediction:{symbol}:{timeframe}"
    
    @staticmethod
    def market_summary(class_filter="all"):
        return f"market:{class_filter}"
    
    @staticmethod
    def chart_data(symbol, timeframe):
        return f"chart:{symbol}:{timeframe}"
    
    @staticmethod
    def websocket_history(symbol, timeframe):
        return f"ws_history:{symbol}:{timeframe}"