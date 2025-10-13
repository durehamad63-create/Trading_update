# Cache Quick Reference Guide

## Import Statement
```python
from utils.cache_manager import CacheManager, CacheKeys, CacheTTL
```

## Basic Usage

### Store Data in Cache
```python
# Crypto price
cache_key = CacheKeys.price('BTC', 'crypto')
CacheManager.set_cache(cache_key, price_data, ttl=CacheTTL.PRICE_CRYPTO)

# Stock price
cache_key = CacheKeys.price('NVDA', 'stock')
CacheManager.set_cache(cache_key, price_data, ttl=CacheTTL.PRICE_STOCK)

# Macro indicator
cache_key = CacheKeys.price('GDP', 'macro')
CacheManager.set_cache(cache_key, price_data, ttl=CacheTTL.PRICE_MACRO)

# Prediction (hot symbol)
cache_key = CacheKeys.prediction('BTC')
CacheManager.set_cache(cache_key, prediction, ttl=CacheTTL.PREDICTION_HOT)

# Prediction (normal symbol)
cache_key = CacheKeys.prediction('ADA')
CacheManager.set_cache(cache_key, prediction, ttl=CacheTTL.PREDICTION_NORMAL)
```

### Retrieve Data from Cache
```python
# Get cached data (Redis → Memory fallback → None)
cache_key = CacheKeys.price('BTC', 'crypto')
cached_data = CacheManager.get_cache(cache_key)

if cached_data:
    # Use cached data
    return cached_data
else:
    # Fetch fresh data
    fresh_data = fetch_from_source()
    CacheManager.set_cache(cache_key, fresh_data, ttl=CacheTTL.PRICE_CRYPTO)
    return fresh_data
```

## Cache Key Patterns

| Function | Pattern | Example |
|----------|---------|---------|
| `CacheKeys.price(symbol, type)` | `price:{type}:{symbol}` | `price:crypto:BTC` |
| `CacheKeys.prediction(symbol)` | `prediction:{symbol}` | `prediction:BTC` |
| `CacheKeys.market_summary(class)` | `market:{class}` | `market:crypto` |
| `CacheKeys.chart_data(symbol, tf)` | `chart:{symbol}:{tf}` | `chart:BTC:1D` |
| `CacheKeys.websocket_history(symbol, tf)` | `ws_history:{symbol}:{tf}` | `ws_history:BTC:1D` |

## TTL Values

| Constant | Value | Use Case |
|----------|-------|----------|
| `CacheTTL.PRICE_CRYPTO` | 30s | Crypto prices |
| `CacheTTL.PRICE_STOCK` | 30s | Stock prices |
| `CacheTTL.PRICE_MACRO` | 300s | Macro indicators |
| `CacheTTL.PREDICTION_HOT` | 1s | BTC, ETH, NVDA, AAPL |
| `CacheTTL.PREDICTION_NORMAL` | 3s | Other symbols |
| `CacheTTL.CHART_DATA` | 600s | Chart data |
| `CacheTTL.WEBSOCKET_HISTORY` | 300s | WebSocket history |

## Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                    WRITE OPERATION                       │
└─────────────────────────────────────────────────────────┘

CacheManager.set_cache(key, value, ttl)
            ↓
    ┌───────────────┐
    │ Memory Cache  │ ← Always written
    └───────────────┘
            ↓
    ┌───────────────┐
    │  Redis Cache  │ ← Try to write (silent fail if unavailable)
    └───────────────┘


┌─────────────────────────────────────────────────────────┐
│                    READ OPERATION                        │
└─────────────────────────────────────────────────────────┘

CacheManager.get_cache(key)
            ↓
    ┌───────────────┐
    │  Redis Cache  │ ← Try first (1-5ms)
    └───────────────┘
            ↓ Miss
    ┌───────────────┐
    │ Memory Cache  │ ← Fallback (<1ms)
    └───────────────┘
            ↓ Miss
         None
```

## Common Patterns

### Pattern 1: Price Caching (Realtime Services)
```python
# In realtime service
price_data = {
    'current_price': 50000.0,
    'change_24h': 2.5,
    'volume': 1000000,
    'timestamp': datetime.now()
}

# Cache it
cache_key = self.cache_keys.price(symbol, 'crypto')
self.cache_manager.set_cache(cache_key, price_data, ttl=self.cache_ttl.PRICE_CRYPTO)
```

### Pattern 2: Prediction Caching (ML Predictor)
```python
# In ML predictor
prediction = {
    'symbol': 'BTC',
    'predicted_price': 51000.0,
    'forecast_direction': 'UP',
    'confidence': 85
}

# Cache with hot symbol priority
cache_key = self.cache_keys.prediction(symbol)
ttl = self.cache_ttl.PREDICTION_HOT if symbol in ['BTC', 'ETH', 'NVDA', 'AAPL'] else self.cache_ttl.PREDICTION_NORMAL
self.cache_manager.set_cache(cache_key, prediction, ttl=ttl)
```

### Pattern 3: WebSocket History Caching
```python
# In WebSocket service
historical_data = {
    'type': 'historical_data',
    'symbol': 'BTC',
    'chart': {...}
}

# Cache for 5 minutes
cache_key = self.cache_keys.websocket_history(symbol, timeframe)
self.cache_manager.set_cache(cache_key, json.dumps(historical_data), ttl=self.cache_ttl.WEBSOCKET_HISTORY)
```

## Troubleshooting

### Redis Not Available
**Symptom:** Application logs "Redis unavailable"
**Solution:** Memory cache automatically takes over, no action needed

### Stale Data
**Symptom:** Old prices displayed
**Solution:** Reduce TTL value in `CacheTTL` class

### High Memory Usage
**Symptom:** Memory cache grows large
**Solution:** Reduce TTL values or implement LRU eviction

### Cache Miss Rate High
**Symptom:** Slow response times
**Solution:** Increase TTL values or check Redis connectivity

## Best Practices

✅ **DO:**
- Use `CacheManager` for all cache operations
- Use `CacheKeys` for standardized key generation
- Use `CacheTTL` for consistent TTL values
- Handle cache misses gracefully

❌ **DON'T:**
- Create new Redis connections directly
- Hardcode cache keys
- Hardcode TTL values
- Assume cache always hits

## Example: Complete Cache Flow

```python
from utils.cache_manager import CacheManager, CacheKeys, CacheTTL

async def get_price(symbol):
    """Get price with caching"""
    
    # 1. Check cache
    cache_key = CacheKeys.price(symbol, 'crypto')
    cached_price = CacheManager.get_cache(cache_key)
    
    if cached_price:
        print(f"✅ Cache hit for {symbol}")
        return cached_price
    
    # 2. Cache miss - fetch from source
    print(f"⚠️ Cache miss for {symbol}, fetching from API")
    fresh_price = await fetch_from_binance(symbol)
    
    # 3. Store in cache
    CacheManager.set_cache(cache_key, fresh_price, ttl=CacheTTL.PRICE_CRYPTO)
    
    return fresh_price
```

## Redis CLI Commands

```bash
# Connect to Redis
redis-cli

# Check if Redis is running
PING

# List all price keys
KEYS price:*

# Get specific price
GET price:crypto:BTC

# Check TTL
TTL price:crypto:BTC

# Delete specific key
DEL price:crypto:BTC

# Flush all cache (DANGER!)
FLUSHDB
```

## Performance Metrics

| Operation | Redis Available | Redis Unavailable |
|-----------|----------------|-------------------|
| Cache Write | Memory + Redis | Memory only |
| Cache Read (Hit) | 1-5ms (Redis) | <1ms (Memory) |
| Cache Read (Miss) | 10-50ms (DB) | 10-50ms (DB) |
| Availability | 99.9% | 99.99% (fallback) |
