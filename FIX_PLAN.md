# Complete Fix Plan - Deploy Once, Run Forever

## Goal
Create a self-sustaining system that runs indefinitely without manual intervention.

---

## Fix Strategy: 3 Core Changes

### **Fix 1: Increase Data Retention (CRITICAL)**
**Problem**: System deletes data after 200 records
**Solution**: Keep 2000 records per symbol-timeframe (covers ~3 months of hourly data)

### **Fix 2: Continuous Data Storage (CRITICAL)**
**Problem**: Data only stored when clients connected
**Solution**: Always store data, regardless of connections

### **Fix 3: Periodic Data Refresh (CRITICAL)**
**Problem**: Gap filling runs once at startup
**Solution**: Run gap filling every 7 days automatically

---

## Implementation Plan

### **Step 1: Fix Gap Filling Service**
**File**: `gap_filling_service.py`

**Changes**:
1. Increase max_records from 200 to 2000
2. Update record limit maintenance to keep 2000 records
3. Add periodic execution support

**Lines to change**:
- Line ~40: `self.max_records = 200` → `self.max_records = 2000`
- Line ~220: `LIMIT 200` → `LIMIT 2000`

---

### **Step 2: Fix Real-Time Services (All 3)**
**Files**: 
- `realtime_websocket_service.py` (Crypto)
- `stock_realtime_service.py` (Stocks)
- `macro_realtime_service.py` (Macro)

**Changes**:
1. Remove condition that checks for active connections before storing
2. Always store price data to database
3. Keep prediction storage priority-based (no change)

**Pattern to apply**:
```python
# BEFORE (WRONG):
if symbol in self.active_connections and self.active_connections[symbol]:
    await self._store_data(...)

# AFTER (CORRECT):
# Always store, regardless of connections
await self._store_data(...)
```

---

### **Step 3: Add Periodic Gap Filling**
**File**: `main.py`

**Changes**:
1. Add background task for periodic gap filling
2. Run every 7 days (604800 seconds)
3. Handle errors gracefully

**Add new function**:
```python
async def periodic_gap_filling():
    """Run gap filling every 7 days"""
    while True:
        try:
            await asyncio.sleep(604800)  # 7 days
            print("🔄 Running scheduled gap filling...")
            await gap_filler.fill_missing_data(db)
            print("✅ Scheduled gap filling completed")
        except Exception as e:
            print(f"⚠️ Scheduled gap filling failed: {e}")
```

**Add to lifespan**:
```python
background_tasks.append(
    asyncio.create_task(periodic_gap_filling())
)
```

---

### **Step 4: Increase Cache TTL (Optional but Recommended)**
**File**: `utils/cache_manager.py`

**Changes**:
1. Increase cache duration to reduce database load
2. Keep data fresh longer

**Lines to change**:
```python
# BEFORE:
PRICE_CRYPTO = 60
PRICE_STOCK = 60
PREDICTION_HOT = 30
PREDICTION_NORMAL = 60

# AFTER:
PRICE_CRYPTO = 300      # 5 minutes
PRICE_STOCK = 300       # 5 minutes
PREDICTION_HOT = 120    # 2 minutes
PREDICTION_NORMAL = 300 # 5 minutes
```

---

### **Step 5: Add Health Monitoring (Optional but Recommended)**
**File**: `main.py`

**Add new function**:
```python
async def health_monitor():
    """Monitor system health and auto-recover"""
    while True:
        try:
            await asyncio.sleep(3600)  # Check every hour
            
            # Check database has recent data
            if db and db.pool:
                async with db.pool.acquire() as conn:
                    count = await conn.fetchval(
                        "SELECT COUNT(*) FROM actual_prices WHERE timestamp > NOW() - INTERVAL '1 hour'"
                    )
                    if count < 10:
                        print("⚠️ Low data count, triggering gap filling...")
                        await gap_filler.fill_missing_data(db)
        except Exception as e:
            print(f"⚠️ Health monitor error: {e}")
```

---

## Detailed Code Changes

### **Change 1: gap_filling_service.py**

**Location 1** (Line ~40):
```python
# BEFORE:
self.max_records = 200

# AFTER:
self.max_records = 2000  # Keep 2000 records (~3 months hourly data)
```

**Location 2** (Line ~220 in `_maintain_record_limit`):
```python
# BEFORE:
await conn.execute("""
    DELETE FROM actual_prices 
    WHERE symbol = $1 AND id NOT IN (
        SELECT id FROM actual_prices 
        WHERE symbol = $1 
        ORDER BY timestamp DESC 
        LIMIT 200
    )
""", symbol_tf)

# AFTER:
await conn.execute("""
    DELETE FROM actual_prices 
    WHERE symbol = $1 AND id NOT IN (
        SELECT id FROM actual_prices 
        WHERE symbol = $1 
        ORDER BY timestamp DESC 
        LIMIT 2000
    )
""", symbol_tf)
```

---

### **Change 2: realtime_websocket_service.py**

**Location 1** (Line ~180 in `_update_candles_and_forecast`):
```python
# BEFORE:
if symbol in self.active_connections and self.active_connections[symbol]:
    asyncio.create_task(self._broadcast_price_update(...))
    asyncio.create_task(self._update_candles_and_forecast(...))
else:
    asyncio.create_task(self._store_all_timeframes(...))

# AFTER:
# Always store data first
asyncio.create_task(self._store_all_timeframes(symbol, current_price, volume, change_24h))

# Then broadcast if connections exist
if symbol in self.active_connections and self.active_connections[symbol]:
    asyncio.create_task(self._broadcast_price_update(...))
    asyncio.create_task(self._update_candles_and_forecast(...))
```

---

### **Change 3: stock_realtime_service.py**

**Location 1** (Line ~120 in `_process_single_stock`):
```python
# BEFORE:
if symbol in self.active_connections and self.active_connections[symbol]:
    asyncio.create_task(self._broadcast_stock_price_update(...))
    asyncio.create_task(self._update_stock_candles_and_forecast(...))
else:
    asyncio.create_task(self._store_stock_data_all_timeframes(...))

# AFTER:
# Always store data first
asyncio.create_task(self._store_stock_data_all_timeframes(symbol, price_data))

# Then broadcast if connections exist
if symbol in self.active_connections and self.active_connections[symbol]:
    asyncio.create_task(self._broadcast_stock_price_update(...))
    asyncio.create_task(self._update_stock_candles_and_forecast(...))
```

---

### **Change 4: macro_realtime_service.py**

**Location 1** (Line ~100 in `_macro_data_stream`):
```python
# BEFORE:
if symbol in self.active_connections and self.active_connections[symbol]:
    asyncio.create_task(self._store_macro_data_all_timeframes(...))
    asyncio.create_task(self._broadcast_macro_update(...))

# AFTER:
# Always store data first
asyncio.create_task(self._store_macro_data_all_timeframes(symbol, self.price_cache[symbol]))

# Then broadcast if connections exist
if symbol in self.active_connections and self.active_connections[symbol]:
    asyncio.create_task(self._broadcast_macro_update(...))
```

---

### **Change 5: main.py**

**Add after line ~60** (after `init_multistep_predictor`):
```python
# Periodic gap filling function
async def periodic_gap_filling():
    """Run gap filling every 7 days to keep data fresh"""
    while True:
        try:
            await asyncio.sleep(604800)  # 7 days in seconds
            print("🔄 Running scheduled gap filling (every 7 days)...")
            from gap_filling_service import GapFillingService
            gap_filler = GapFillingService(model)
            await gap_filler.fill_missing_data(db)
            print("✅ Scheduled gap filling completed successfully")
        except Exception as e:
            print(f"⚠️ Scheduled gap filling failed: {e}")
            # Continue running, will retry in 7 days
```

**Add in lifespan function** (after line ~130):
```python
# Add periodic gap filling to background tasks
background_tasks.append(
    asyncio.create_task(periodic_gap_filling())
)
print("✅ Periodic gap filling scheduled (every 7 days)")
```

---

### **Change 6: utils/cache_manager.py**

**Location 1** (Line ~10):
```python
# BEFORE:
class CacheTTL:
    PRICE_CRYPTO = 60
    PRICE_STOCK = 60
    PRICE_MACRO = 600
    PREDICTION_HOT = 30
    PREDICTION_NORMAL = 60
    PREDICTION_COLD = 120

# AFTER:
class CacheTTL:
    PRICE_CRYPTO = 300      # 5 minutes (was 60)
    PRICE_STOCK = 300       # 5 minutes (was 60)
    PRICE_MACRO = 1800      # 30 minutes (was 600)
    PREDICTION_HOT = 120    # 2 minutes (was 30)
    PREDICTION_NORMAL = 300 # 5 minutes (was 60)
    PREDICTION_COLD = 600   # 10 minutes (was 120)
```

---

## Testing Plan

### **Test 1: Verify Data Retention**
```sql
-- Check record count per symbol
SELECT symbol, COUNT(*) as record_count 
FROM actual_prices 
GROUP BY symbol 
ORDER BY record_count DESC;

-- Should show ~2000 records per symbol
```

### **Test 2: Verify Continuous Storage**
```sql
-- Check recent data (last hour)
SELECT symbol, MAX(timestamp) as last_update
FROM actual_prices
GROUP BY symbol
ORDER BY last_update DESC;

-- All symbols should have data within last hour
```

### **Test 3: Verify Periodic Gap Filling**
```bash
# Check logs after 7 days
grep "Scheduled gap filling" logs.txt

# Should show:
# "🔄 Running scheduled gap filling (every 7 days)..."
# "✅ Scheduled gap filling completed successfully"
```

---

## Deployment Checklist

- [ ] Backup current database
- [ ] Apply Change 1: gap_filling_service.py (2000 records)
- [ ] Apply Change 2: realtime_websocket_service.py (always store)
- [ ] Apply Change 3: stock_realtime_service.py (always store)
- [ ] Apply Change 4: macro_realtime_service.py (always store)
- [ ] Apply Change 5: main.py (periodic gap filling)
- [ ] Apply Change 6: cache_manager.py (longer TTL)
- [ ] Test locally for 1 hour
- [ ] Deploy to production
- [ ] Monitor for 24 hours
- [ ] Verify data retention after 7 days

---

## Expected Results

### **Before Fix**:
```
Day 1: 200 records ✅
Day 2: 200 records ✅
Day 3: 200 records ✅
Day 4: 200 records (old deleted) ❌
Day 5: Validation fails ❌
```

### **After Fix**:
```
Day 1: 2000 records ✅
Day 2: 2000 records ✅
Day 3: 2000 records ✅
Day 7: Gap filling runs ✅
Day 30: 2000 records ✅
Day 90: 2000 records ✅
Day 365: System still running ✅
```

---

## Maintenance Requirements

**NONE** - System is fully self-sustaining:
- ✅ Data stored continuously
- ✅ Old data cleaned automatically (keeps 2000 records)
- ✅ Gap filling runs every 7 days
- ✅ Cache prevents database overload
- ✅ Real-time services auto-reconnect on failure

**Deploy once, runs forever!**

---

## Rollback Plan (If Issues Occur)

1. Revert to backup database
2. Revert code changes (git reset)
3. Restart application
4. System returns to previous state

---

## Summary

**Total Changes**: 6 files, ~20 lines of code
**Deployment Time**: 10 minutes
**Testing Time**: 1 hour
**Maintenance**: Zero (fully automated)

**Result**: Deploy once, system runs indefinitely without intervention.
