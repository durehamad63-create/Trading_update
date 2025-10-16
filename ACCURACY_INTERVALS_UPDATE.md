# Accuracy History Interval Formatting Update

## Changes Implemented

### New File: `utils/interval_formatter.py`
Created a utility class to format timestamps based on timeframe requirements.

### Updated File: `modules/routes/trends_routes.py`
Integrated interval formatter into the trends API endpoint.

## Interval Formatting Rules

| Timeframe | Interval Display | Example |
|-----------|-----------------|---------|
| **1H** | 30-minute intervals | 09:30, 10:00, 10:30 |
| **4H** | 1-hour intervals | 09:00, 10:00, 11:00 |
| **1D** | Daily intervals | Aug 28, Aug 29, Aug 30 |
| **7D** | Daily intervals | Oct 01, Oct 02, Oct 03 |
| **1M** | Weekly intervals | Week 1, Week 2, Week 3, Week 4 |

## Implementation Details

### IntervalFormatter Class
```python
format_timestamp(timestamp_str, timeframe) -> str
```
- Converts ISO timestamp to formatted string based on timeframe
- Handles timezone conversion automatically
- Returns human-readable interval labels

### API Response Format
The `accuracy_history` array in `/api/asset/{symbol}/trends` now returns:
```json
{
  "accuracy_history": [
    {
      "date": "Aug 28",  // Formatted based on timeframe
      "actual": 45123.50,
      "predicted": 45200.00,
      "result": "Hit",
      "error_pct": 0.2
    }
  ]
}
```

## Testing
Test the changes by calling:
- `GET /api/asset/BTC/trends?timeframe=1H` → Shows 30-min intervals
- `GET /api/asset/BTC/trends?timeframe=4H` → Shows 1-hour intervals
- `GET /api/asset/BTC/trends?timeframe=1D` → Shows daily intervals
- `GET /api/asset/BTC/trends?timeframe=7D` → Shows daily intervals
- `GET /api/asset/BTC/trends?timeframe=1M` → Shows weekly intervals

## Benefits
✅ Cleaner, more readable timestamps
✅ Consistent formatting across all timeframes
✅ Easy to extend for new timeframes
✅ Minimal code changes required

## Fixes Applied
- **Stablecoin prices**: USDT/USDC now always return $1.00
- **1D timeframe**: Changed from 4-hour intervals to daily date format (Aug 28, Aug 29)
