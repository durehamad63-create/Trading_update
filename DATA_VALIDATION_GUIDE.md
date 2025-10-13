# Data Validation Guide

## Overview
All data stored in the Trading AI Platform must pass validation to ensure only real API data is persisted.

## Validation Rules

### 1. Price Data Validation
```python
from utils.data_validator import data_validator

price_data = {
    'current_price': 50000.0,
    'timestamp': datetime.now(),
    'change_24h': 2.5,
    'volume': 1000000
}

# Validate before storing
if data_validator.validate_price_data(price_data, 'binance'):
    await db.store_actual_price(symbol, price_data)
else:
    logging.warning("Invalid price data rejected")
```

**Requirements:**
- `current_price` must be positive number
- `timestamp` must be datetime object
- Source must be valid API: binance, yahoo, fred, iex, alpha_vantage

### 2. Historical Data Validation
```python
historical_data = [
    {'timestamp': datetime.now(), 'close': 50000, 'open': 49500, 'high': 50500, 'low': 49000},
    # ... more data points
]

if data_validator.validate_historical_data(historical_data, 'binance'):
    await db.store_historical_batch(symbol, historical_data)
```

**Requirements:**
- Non-empty list
- Each point must have `timestamp` and `close` fields
- All points must be dictionaries

### 3. Forecast Data Validation
```python
forecast_data = {
    'symbol': 'BTC',
    'forecast_direction': 'UP',
    'confidence': 75,
    'predicted_price': 51000
}

if data_validator.validate_forecast_data(forecast_data):
    await db.store_forecast(symbol, forecast_data)
```

**Requirements:**
- `forecast_direction` must be: UP, DOWN, or HOLD
- `confidence` must be 0-100
- Required fields: symbol, forecast_direction, confidence

### 4. Synthetic Data Detection
```python
# Automatically rejects data with synthetic markers
if data_validator.is_synthetic_data(data):
    logging.error("Synthetic data detected and rejected")
    return
```

**Detects:**
- Data source containing: synthetic, generated, simulated, random, sample, mock, fake, test
- Metadata flags: is_synthetic, is_simulated

## Integration Points

### Database Operations
All database storage methods now validate data:
- `store_actual_price()` - Validates price data
- `store_forecast()` - Validates forecast data
- `store_historical_batch()` - Validates historical data

### Real-Time Services
Services validate data before caching:
```python
# In realtime_websocket_service.py
price_data = {
    'current_price': float(data['c']),
    'change_24h': float(data['P']),
    'volume': float(data['v']),
    'timestamp': datetime.now()
}

# Validation happens automatically in database layer
await self._store_realtime_data(symbol, price_data, timeframe)
```

## Error Handling

### Invalid Data
When validation fails:
1. Log warning with symbol and reason
2. Skip storage operation
3. Continue processing other data
4. No exception thrown (graceful degradation)

### Synthetic Data
When synthetic data detected:
1. Log error with detection details
2. Reject data immediately
3. Return without storing
4. Alert monitoring system

## Testing Validation

### Valid Data Test
```python
valid_data = {
    'current_price': 50000.0,
    'timestamp': datetime.now(),
    'data_source': 'binance'
}
assert data_validator.validate_price_data(valid_data, 'binance') == True
```

### Invalid Data Test
```python
invalid_data = {
    'current_price': -100,  # Negative price
    'timestamp': datetime.now()
}
assert data_validator.validate_price_data(invalid_data, 'binance') == False
```

### Synthetic Data Test
```python
synthetic_data = {
    'current_price': 50000.0,
    'timestamp': datetime.now(),
    'data_source': 'synthetic_generator'
}
assert data_validator.is_synthetic_data(synthetic_data) == True
```

## Monitoring

### Validation Metrics
Track validation failures:
```python
# Log validation failures for monitoring
if not data_validator.validate_price_data(data, source):
    metrics.increment('validation.price.failed', tags=[f'source:{source}'])
```

### Synthetic Data Alerts
Alert when synthetic data detected:
```python
if data_validator.is_synthetic_data(data):
    alert.send('Synthetic data detected', severity='high')
```

## Best Practices

1. **Always validate before storing** - Never skip validation
2. **Use correct source identifier** - Match actual API source
3. **Log validation failures** - Track data quality issues
4. **Handle gracefully** - Don't crash on invalid data
5. **Monitor trends** - Watch for increasing validation failures

## Compliance

✅ All data validated before storage  
✅ Synthetic data automatically rejected  
✅ Invalid data logged and skipped  
✅ Real API sources enforced  
✅ Validation integrated in database layer
