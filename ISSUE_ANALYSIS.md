# Data Validation Issue - Root Cause Analysis

## Problem
After 2-3 days, the system returns error: **"Could not find valid data"**

---

## Root Causes Identified

### **Issue 1: Record Limit Deletion (CRITICAL)**

**Location**: `gap_filling_service.py` - `_maintain_record_limit()` method

```python
async def _maintain_record_limit(self, symbol_tf: str, timeframe: str):
    """Maintain exactly 1000 records per symbol-timeframe"""
    await conn.execute("""
        DELETE FROM actual_prices 
        WHERE symbol = $1 AND id NOT IN (
            SELECT id FROM actual_prices 
            WHERE symbol = $1 
            ORDER BY timestamp DESC 
            LIMIT 200  # ← KEEPS ONLY LAST 200 RECORDS
        )
    """, symbol_tf)
```

**Problem**: 
- Gap filling keeps only **200 records** per symbol-timeframe
- After 2-3 days, older data is deleted
- When validation queries historical data, it finds insufficient records
- Validation fails: "Insufficient data for predictions"

**Timeline**:
```
Day 1: 200 records stored ✅
Day 2: 200 records stored ✅
Day 3: 200 records stored ✅
Day 4: DELETE old records, keep only 200 ❌
       Historical queries fail (not enough data)
```

---

### **Issue 2: Validation Requires Minimum 20 Records**

**Location**: `modules/data_validator.py` and `gap_filling_service.py`

```python
# In _generate_ml_predictions()
if not data or len(data) < 20:
    return predictions  # Returns empty!

# In _get_monthly_data()
if not data or len(data) < 20:
    print(f"❌ Insufficient data for {symbol} {timeframe}")
    continue  # Skips this symbol
```

**Problem**:
- System needs minimum 20 historical data points for feature calculation
- After deletion, only 200 records remain
- But if queries are sparse, might get < 20 records
- Validation fails silently

---

### **Issue 3: Cache Expiration Without Database Fallback**

**Location**: `utils/cache_manager.py` - Cache TTL settings

```python
class CacheTTL:
    PRICE_CRYPTO = 60      # 1 minute
    PRICE_STOCK = 60       # 1 minute
    PRICE_MACRO = 600      # 10 minutes
    PREDICTION_HOT = 30    # 30 seconds
    PREDICTION_NORMAL = 60 # 1 minute
```

**Problem**:
- Cache expires after 30-120 seconds
- If database has no data, cache miss = no data
- After 2-3 days, if real-time streams fail, no fallback data exists
- Validation fails: "No valid data found"

---

### **Issue 4: Real-Time Services Not Storing Data Continuously**

**Location**: `realtime_websocket_service.py` - `_store_realtime_data()`

```python
async def _store_realtime_data(self, db_key, price_data, timeframe):
    # Only stores if update is due (priority-based)
    if self.cache_manager.should_update_prediction(symbol, timeframe):
        # Store prediction
    # But price data might not be stored if no active connections!
```

**Problem**:
- Price data only stored if there are active WebSocket connections
- If no clients connected, data not stored
- After 2-3 days with no connections, database becomes empty
- Validation fails: "No historical data available"

---

### **Issue 5: Gap Filling Only Runs at Startup**

**Location**: `main.py` - Lifespan management

```python
async def setup_database():
    # Gap filling runs ONCE at startup
    await gap_filler.fill_missing_data(db)
    # After this, no more data collection!
```

**Problem**:
- Gap filling runs only once during startup
- After 2-3 days, no new data is collected
- Real-time services might fail or have gaps
- Database becomes stale
- Validation fails: "Insufficient data"

---

## Why It Happens After 2-3 Days

```
Timeline:
Day 0 (Startup):
  ✅ Gap filling collects 200 records per symbol-timeframe
  ✅ Real-time services start streaming live data
  ✅ Cache populated with fresh data

Day 1-2:
  ✅ Real-time services update database continuously
  ✅ Cache hits are frequent
  ✅ Validation passes

Day 3:
  ⚠️ Real-time services might have connection issues
  ⚠️ Cache expires (30-120 seconds)
  ⚠️ Database queries return stale data

Day 4:
  ❌ Gap filling deletes old records (keeps only 200)
  ❌ Real-time services haven't updated in hours
  ❌ Cache is empty
  ❌ Database has insufficient data
  ❌ Validation fails: "Could not find valid data"
```

---

## Solutions

### **Solution 1: Increase Record Limit**
```python
# Change from 200 to 1000+ records
self.max_records = 1000  # Instead of 200

# In _maintain_record_limit():
LIMIT 1000  # Instead of LIMIT 200
```

**Benefit**: More historical data available for validation

---

### **Solution 2: Store All Real-Time Data (Not Just When Connections Exist)**
```python
# In realtime_websocket_service.py
async def _store_realtime_data(self, db_key, price_data, timeframe):
    # ALWAYS store price data
    await db.store_actual_price(db_key, price_data, timeframe)
    
    # Only store predictions if update is due
    if self.cache_manager.should_update_prediction(symbol, timeframe):
        prediction = await self.model.predict(symbol, timeframe)
        await db.store_forecast(db_key, prediction, timeframe)
```

**Benefit**: Database always has fresh data, even without active connections

---

### **Solution 3: Run Gap Filling Periodically (Not Just at Startup)**
```python
# In main.py
async def periodic_gap_filling():
    while True:
        await asyncio.sleep(86400)  # Every 24 hours
        await gap_filler.fill_missing_data(db)

# In lifespan:
background_tasks.append(asyncio.create_task(periodic_gap_filling()))
```

**Benefit**: Database stays fresh with historical data

---

### **Solution 4: Increase Cache TTL for Fallback**
```python
class CacheTTL:
    PRICE_CRYPTO = 300      # 5 minutes (instead of 60)
    PRICE_STOCK = 300       # 5 minutes (instead of 60)
    PRICE_MACRO = 1800      # 30 minutes (instead of 600)
    PREDICTION_HOT = 120    # 2 minutes (instead of 30)
    PREDICTION_NORMAL = 300 # 5 minutes (instead of 60)
```

**Benefit**: Cache lasts longer, reduces database queries

---

### **Solution 5: Add Data Freshness Check**
```python
async def validate_data_freshness(symbol, timeframe):
    """Check if data is recent enough"""
    last_update = await db.get_last_stored_time(symbol, timeframe)
    
    if not last_update:
        return False  # No data
    
    time_diff = datetime.now() - last_update
    
    # Data must be less than 1 hour old
    if time_diff > timedelta(hours=1):
        return False  # Data is stale
    
    return True
```

**Benefit**: Detects stale data before validation fails

---

## Recommended Fix (Priority Order)

1. **Increase record limit from 200 to 1000+**
   - Immediate impact: More data available
   - Risk: Slightly higher database storage

2. **Store all real-time data (not just when connections exist)**
   - Immediate impact: Database always has fresh data
   - Risk: Slightly higher database writes

3. **Run gap filling every 24 hours**
   - Delayed impact: Ensures data freshness
   - Risk: API rate limiting

4. **Increase cache TTL**
   - Immediate impact: Fewer database queries
   - Risk: Slightly stale data

5. **Add data freshness validation**
   - Delayed impact: Better error messages
   - Risk: More validation overhead

---

## Quick Fix (Minimum Changes)

Change in `gap_filling_service.py`:

```python
# Line: self.max_records = 200
self.max_records = 1000  # Increase from 200 to 1000

# In _maintain_record_limit():
# Change: LIMIT 200
# To: LIMIT 1000
```

This single change will prevent data deletion and solve the 2-3 day issue.
