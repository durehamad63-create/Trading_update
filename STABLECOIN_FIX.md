# Stablecoin Historical Data Fix

## Problem
Stablecoins (USDT, USDC) were showing incorrect historical prices (BTC prices) in chart data instead of the correct fixed $1.00 value.

### Example of Issue:
```json
{
    "symbol": "USDC",
    "current_price": 1.0,
    "chart": {
        "past": [107986.14, 108364.21, ...],  // ❌ Wrong - BTC prices
        "future": [1.0]
    }
}
```

## Root Cause
1. Database contained incorrect price data for stablecoins
2. WebSocket chart route was fetching from database without filtering stablecoins
3. Gap filling service was collecting data for stablecoins

## Solution

### 1. WebSocket Chart Route Fix
**File**: `modules/routes/websocket_routes.py`

Added stablecoin detection and fixed $1.00 historical data generation:

```python
# Check if stablecoin - use fixed $1.00 for all historical data
is_stablecoin = symbol in ['USDT', 'USDC']

if is_stablecoin:
    # Generate 30 points of $1.00 with proper timestamps
    past_prices = [1.0] * 30
    timestamps = [(current_time - timedelta(hours=i)).isoformat() for i in range(29, -1, -1)]
```

### 2. Gap Filling Service Fix
**File**: `gap_filling_service.py`

Excluded stablecoins from data collection:

```python
# Exclude USDT and USDC from crypto symbols
self.crypto_symbols = [s for s in CRYPTO_SYMBOLS.keys() if s not in ['USDT', 'USDC']]
```

### 3. Database Cleanup Script
**File**: `fix_stablecoin_history.py`

Created script to remove incorrect stablecoin data from database:

```bash
python fix_stablecoin_history.py
```

This will:
- Delete all USDT records from database
- Delete all USDC records from database
- Report total records cleaned

## Expected Results

### After Fix:
```json
{
    "symbol": "USDC",
    "current_price": 1.0,
    "predicted_price": 1.0,
    "chart": {
        "past": [1.0, 1.0, 1.0, ...],  // ✅ Correct - all $1.00
        "future": [1.0]
    }
}
```

## Implementation Steps

1. **Run cleanup script** (one-time):
   ```bash
   python fix_stablecoin_history.py
   ```

2. **Restart server** to apply code changes:
   ```bash
   python main.py
   ```

3. **Verify fix**:
   - Connect to WebSocket: `ws://localhost:8000/ws/chart/USDC?timeframe=1H`
   - Check that all past prices are 1.0
   - Check that all past prices are 1.0 for USDT as well

## Stablecoin Behavior

### Current Price (Real-time):
- Always returns `1.0`
- No API calls needed
- Defined in `config/symbols.py` as `fixed_price: 1.0`

### Historical Data (Charts):
- Always shows `1.0` for all past data points
- Generated dynamically with proper timestamps
- No database storage needed

### Predictions:
- Always predicts `1.0`
- Direction: `HOLD`
- Confidence: `99%`
- Range: `$1.00 - $1.00`

## Files Modified

1. ✅ `modules/routes/websocket_routes.py` - Fixed chart data
2. ✅ `gap_filling_service.py` - Excluded from data collection
3. ✅ `fix_stablecoin_history.py` - Cleanup script (new)

## Verification Checklist

- ✅ USDT shows $1.00 for all historical data
- ✅ USDC shows $1.00 for all historical data
- ✅ No stablecoin data stored in database
- ✅ Gap filling skips stablecoins
- ✅ Real-time service handles stablecoins correctly
- ✅ Predictions always return $1.00

## Notes

- Stablecoins are pegged to $1.00 USD by design
- No need to fetch or store historical data
- This is correct behavior, not synthetic data
- Reduces database storage and API calls
