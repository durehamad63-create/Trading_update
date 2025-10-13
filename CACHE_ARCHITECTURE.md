# Cache Architecture - Consistent Data Flow

## Overview
The Trading AI Platform uses a **3-tier cache hierarchy** with automatic fallback:
1. **Memory Cache** (fastest, in-process)
2. **Redis Cache** (fast, distributed)
3. **Database** (persistent, source of truth)

## Cache Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA FLOW HIERARCHY                       │
└─────────────────────────────────────────────────────────────┘

READ PATH (Fastest → Slowest):
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Memory     │ →  │    Redis     │ →  │   Database   │
│   Cache      │    │    Cache     │    │   (Source)   │
└──────────────┘    └──────────────┘    └──────────────┘
   ↓ Hit              ↓ Hit              ↓ Hit
   Return            Return             Return
                     + Cache             + Cache
                       Memory             Memory + Redis

WRITE PATH (All layers):
┌──────────────┐
│  New Data    │
└──────────────┘
       ↓
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Memory     │ ←  │    Redis     │ ←  │   Database   │
│   Cache      │    │    Cache     │    │   (Store)    │
└──────────────┘    └──────────────┘    └──────────────┘
```

## Centralized Cache Manager

### Location
`utils/cache_manager.py`

### Components

#### 1. CacheTTL (Time-To-Live Configuration)
```python
class CacheTTL:
    PRICE_CRYPTO = 30        # 30 seconds for crypto prices
    PRICE_STOCK = 30         # 30 seconds for stock prices
    PRICE_MACRO = 300        # 5 minutes for macro indicators
    PREDICTION_HOT = 1       # 1 second for hot symbols (BTC, ETH, NVDA, AAPL)
    PREDICTION_NORMAL = 3    # 3 seconds for normal symbols
    CHART_DATA = 600         # 10 minutes for chart data
    WEBSOCKET_HISTORY = 300  # 5 minutes for WebSocket historical data
```

#### 2. CacheManager (Unified Cache Operations)
```python
class CacheManager:
    # Redis primary, Memory fallback
    
    @classmethod
    def set_cache(cls, key, value, ttl):
        """
        1. Store in memory cache (always)
        2. Try to store in Redis (if available)
        """
    
    @classmethod
    def get_cache(cls, key):
        """
        1. Try Redis first (if available)
        2. Fallback to memory cache (if Redis fails)
        3. Return None if both miss
        """
```

#### 3. CacheKeys (Standardized Key Generation)
```python
class CacheKeys:
    @staticmethod
    def price(symbol, asset_type):
        return f"price:{asset_type}:{symbol}"
    
    @staticmethod
    def prediction(symbol):
        return f"prediction:{symbol}"
    
    @staticmethod
    def market_summary(class_filter="all"):
        return f"market:{class_filter}"
    
    @staticmethod
    def chart_data(symbol, timeframe):
        return f"chart:{symbol}:{timeframe}"
    
    @staticmethod
    def websocket_history(symbol, timeframe):
        return f"ws_history:{symbol}:{timeframe}"
```

## Data Flow by Service

### 1. Crypto Realtime Service (realtime_websocket_service.py)

**Price Update Flow:**
```
Binance WebSocket → price_cache (memory) → CacheManager.set_cache()
                                              ↓
                                    Memory Cache + Redis Cache
                                              ↓
                                         Database Store
```

**Price Read Flow:**
```
API Request → CacheManager.get_cache() → Redis (try) → Memory (fallback) → Database (last resort)
```

**Key Pattern:** `price:crypto:BTC`
**TTL:** 30 seconds (CacheTTL.PRICE_CRYPTO)

### 2. Stock Realtime Service (stock_realtime_service.py)

**Price Update Flow:**
```
Yahoo Finance API → price_cache (memory) → CacheManager.set_cache()
                                              ↓
                                    Memory Cache + Redis Cache
                                              ↓
                                         Database Store
```

**Key Pattern:** `price:stock:NVDA`
**TTL:** 30 seconds (CacheTTL.PRICE_STOCK)

### 3. Macro Realtime Service (macro_realtime_service.py)

**Price Update Flow:**
```
FRED API → price_cache (memory) → CacheManager.set_cache()
                                      ↓
                            Memory Cache + Redis Cache
                                      ↓
                                 Database Store
```

**Key Pattern:** `price:macro:GDP`
**TTL:** 300 seconds (CacheTTL.PRICE_MACRO)

### 4. ML Predictor (modules/ml_predictor.py)

**Prediction Flow:**
```
predict(symbol) → Check CacheManager.get_cache()
                       ↓ Cache Miss
                  Get cached prices (from realtime services)
                       ↓
                  XGBoost Model Inference
                       ↓
                  CacheManager.set_cache() → Memory + Redis
```

**Key Pattern:** `prediction:BTC`
**TTL:** 
- Hot symbols (BTC, ETH, NVDA, AAPL): 1 second
- Normal symbols: 3 seconds

### 5. WebSocket Historical Data

**Historical Data Flow:**
```
WebSocket Connect → CacheManager.get_cache(ws_history:BTC:1D)
                         ↓ Cache Miss
                    Database Query
                         ↓
                    CacheManager.set_cache() → Memory + Redis
                         ↓
                    Send to Client
```

**Key Pattern:** `ws_history:BTC:1D`
**TTL:** 300 seconds (CacheTTL.WEBSOCKET_HISTORY)

## Cache Consistency Rules

### 1. Write-Through Pattern
- All writes go to **Memory + Redis + Database** simultaneously
- Ensures consistency across all layers

### 2. Read-Through Pattern with Fallback
- Read order: **Redis → Memory → Database**
- If Redis fails, memory cache serves as backup
- If both fail, query database and populate caches

### 3. TTL-Based Expiration
- All cache entries have explicit TTL
- Memory cache tracks timestamps for expiration
- Redis handles expiration automatically

### 4. Hot Symbol Priority
- BTC, ETH, NVDA, AAPL get 1-second prediction cache
- Other symbols get 3-second prediction cache
- Ensures freshest data for most-traded assets

## Redis Configuration

### Connection Settings
```python
redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', '6379')),
    db=0,  # Single database for all cache types
    password=os.getenv('REDIS_PASSWORD'),
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
    retry_on_timeout=True,
    health_check_interval=30
)
```

### Database Selection
- **db=0**: All cache types (unified for Railway compatibility)
- Single database simplifies deployment and reduces connection overhead

## Memory Cache Implementation

### Storage
```python
_memory_cache = {}  # {key: value}
_cache_timestamps = {}  # {key: (timestamp, ttl)}
```

### Expiration Logic
```python
if key in _memory_cache:
    timestamp, ttl = _cache_timestamps[key]
    if time.time() - timestamp < ttl:
        return _memory_cache[key]  # Valid
    else:
        # Expired, remove
        _memory_cache.pop(key)
        _cache_timestamps.pop(key)
```

## Performance Characteristics

### Cache Hit Rates (Expected)
- **Memory Cache**: ~95% hit rate for hot symbols
- **Redis Cache**: ~90% hit rate overall
- **Database**: <5% queries (cache misses only)

### Response Times
- **Memory Cache Hit**: <1ms
- **Redis Cache Hit**: 1-5ms
- **Database Query**: 10-50ms

### Cache Sizes
- **Memory Cache**: Unbounded (auto-expires by TTL)
- **Redis Cache**: Limited by Redis memory (auto-eviction)
- **Database**: Unlimited (persistent storage)

## Monitoring & Debugging

### Cache Status Check
```python
# Check Redis availability
redis_client = CacheManager.get_redis_client()
if redis_client:
    print("✅ Redis available")
else:
    print("⚠️ Redis unavailable, using memory fallback")
```

### Cache Key Inspection
```bash
# Redis CLI
redis-cli
> KEYS price:*
> GET price:crypto:BTC
> TTL price:crypto:BTC
```

### Memory Cache Inspection
```python
# In Python
print(f"Memory cache size: {len(CacheManager._memory_cache)}")
print(f"Cached keys: {list(CacheManager._memory_cache.keys())}")
```

## Best Practices

### 1. Always Use CacheManager
❌ **Don't create new Redis connections:**
```python
redis_client = redis.Redis(...)  # BAD
```

✅ **Use centralized manager:**
```python
CacheManager.set_cache(key, value, ttl)  # GOOD
```

### 2. Use Standardized Keys
❌ **Don't hardcode keys:**
```python
cache_key = f"price_{symbol}"  # BAD
```

✅ **Use CacheKeys:**
```python
cache_key = CacheKeys.price(symbol, 'crypto')  # GOOD
```

### 3. Use Centralized TTL
❌ **Don't hardcode TTL:**
```python
CacheManager.set_cache(key, value, ttl=30)  # BAD
```

✅ **Use CacheTTL:**
```python
CacheManager.set_cache(key, value, ttl=CacheTTL.PRICE_CRYPTO)  # GOOD
```

### 4. Handle Cache Misses Gracefully
```python
cached_data = CacheManager.get_cache(key)
if cached_data:
    return cached_data
else:
    # Fetch from source
    fresh_data = fetch_from_source()
    # Populate cache
    CacheManager.set_cache(key, fresh_data, ttl)
    return fresh_data
```

## Troubleshooting

### Issue: Redis Connection Fails
**Symptom:** Application logs "Redis unavailable"
**Solution:** Memory cache automatically takes over, no action needed

### Issue: Stale Data in Cache
**Symptom:** Old prices displayed
**Solution:** Check TTL values, reduce if needed

### Issue: High Memory Usage
**Symptom:** Memory cache grows unbounded
**Solution:** Implement LRU eviction or reduce TTL values

### Issue: Cache Inconsistency
**Symptom:** Different data in Redis vs Memory
**Solution:** Both caches expire independently, this is expected behavior

## Future Improvements

1. **LRU Eviction for Memory Cache**: Limit memory cache size with LRU policy
2. **Cache Warming**: Pre-populate caches on startup
3. **Cache Metrics**: Track hit/miss rates, response times
4. **Distributed Locking**: Prevent cache stampede on popular keys
5. **Cache Versioning**: Invalidate caches on schema changes
