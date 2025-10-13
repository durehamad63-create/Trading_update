# Trading AI Platform - Development Guidelines

## Code Quality Standards

### File Structure and Organization
- **Module docstrings**: Every file starts with a triple-quoted docstring describing its purpose (5/5 files)
- **Shebang for executables**: Python scripts use `#!/usr/bin/env python3` when designed to run standalone (2/5 files)
- **Logical grouping**: Related functionality organized into dedicated modules (routes/, utils/, config/)
- **Separation of concerns**: Clear boundaries between API routes, business logic, data access, and utilities

### Import Organization
- **Standard library first**: Built-in modules imported before third-party packages
- **Third-party packages second**: External dependencies grouped together
- **Local imports last**: Project-specific modules imported after external dependencies
- **Explicit imports**: Prefer specific imports over wildcard imports
- **Conditional imports**: Import heavy dependencies only when needed (e.g., inside functions)

Example pattern:
```python
# Standard library
import asyncio
import logging
from datetime import datetime

# Third-party
from fastapi import FastAPI
import numpy as np

# Local
from modules.ml_predictor import MobileMLModel
from utils.cache_manager import CacheManager
```

### Naming Conventions
- **Classes**: PascalCase (e.g., `MobileMLModel`, `AsyncTaskManager`, `RealTimeWebSocketService`)
- **Functions/Methods**: snake_case (e.g., `predict`, `get_asset_data`, `run_background_task`)
- **Private methods**: Prefix with single underscore (e.g., `_get_real_price`, `_populate_initial_cache`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `CRYPTO_SYMBOLS`, `XGBOOST_AVAILABLE`)
- **Variables**: snake_case (e.g., `current_price`, `prediction_cache`, `task_manager`)

### Type Hints and Documentation
- **Type hints on function signatures**: Parameters and return types specified (4/5 files)
- **Docstrings for public methods**: Clear descriptions of functionality
- **Inline comments**: Explain complex logic, not obvious code
- **TODO/FIXME markers**: Not used - code is production-ready

Example:
```python
async def run_task(self, task_id: str, coro: Callable, *args, **kwargs) -> Any:
    """Run a task with concurrency control"""
```

## Async/Await Patterns

### Consistent Async Usage
- **Async functions throughout**: All I/O operations use async/await (5/5 files)
- **asyncio.create_task**: Background tasks created with `asyncio.create_task()` for non-blocking execution
- **asyncio.gather**: Multiple concurrent operations batched with `asyncio.gather(*tasks, return_exceptions=True)`
- **asyncio.sleep**: Non-blocking delays with `await asyncio.sleep(seconds)`
- **Semaphores for concurrency control**: Limit concurrent operations with `asyncio.Semaphore(n)`

Example pattern:
```python
async def _broadcast_to_timeframe(self, symbol, timeframe, data):
    semaphore = asyncio.Semaphore(10)  # Limit concurrent sends
    
    async def send_batch(conn_id, websocket):
        async with semaphore:
            await websocket.send_text(message)
    
    await asyncio.gather(
        *[send_batch(conn_id, ws) for conn_id, ws in matching_connections],
        return_exceptions=True
    )
```

### Timeout Handling
- **asyncio.wait_for**: Wrap operations with timeouts to prevent hanging
- **Graceful degradation**: Fallback to defaults when timeouts occur

Example:
```python
try:
    real_price = await asyncio.wait_for(self._get_real_price(symbol), timeout=2.0)
except asyncio.TimeoutError:
    # Fallback logic
```

## Error Handling

### Exception Management
- **Try-except blocks**: Wrap risky operations in try-except (5/5 files extensively)
- **Specific exceptions**: Catch specific exception types when possible
- **Centralized error handler**: Use `ErrorHandler` utility for consistent logging
- **Graceful degradation**: Provide fallback behavior instead of crashing
- **Error propagation**: Re-raise exceptions when caller needs to handle them

Example pattern:
```python
try:
    result = await risky_operation()
except SpecificException as e:
    ErrorHandler.log_error('operation_type', symbol, str(e))
    # Fallback logic
except Exception as e:
    # Generic fallback
```

### Logging Strategy
- **Minimal logging in production**: Use WARNING level by default
- **Structured log messages**: Include context (symbol, operation type)
- **Print statements for user feedback**: Use `print()` for startup/status messages
- **Flush output**: Use `flush=True` for critical messages

Example:
```python
print(f"✅ Connected: {symbol} stream active", flush=True)
logging.warning(f"⚠️ Database connection failed: {e}")
```

## Caching Patterns

### Multi-Tier Caching
- **Redis primary**: Use Redis for distributed caching across instances
- **Memory fallback**: In-memory cache when Redis unavailable
- **TTL-based expiration**: All cache entries have time-to-live
- **Hot symbol priority**: Shorter TTL for frequently accessed symbols (BTC, ETH, NVDA, AAPL)

Example:
```python
# Check cache first
cache_key = self.cache_keys.prediction(symbol)
cached_data = self.cache_manager.get_cache(cache_key)
if cached_data:
    return cached_data

# Generate fresh data
result = await generate_data()

# Cache with TTL
ttl = 1 if symbol in ['BTC', 'ETH'] else 3
self.cache_manager.set_cache(cache_key, result, ttl)
```

### Cache Key Patterns
- **Centralized key generation**: Use `CacheKeys` class for consistent naming
- **Hierarchical keys**: Structure keys by type (e.g., `prediction:BTC:1D`, `price:ETH:crypto`)
- **Include timeframe**: Cache keys include timeframe for multi-timeframe support

## Database Operations

### Connection Management
- **Connection pooling**: Use asyncpg pool with 5-20 connections
- **Pool acquisition**: Use `async with db.pool.acquire() as conn:` pattern
- **Null checks**: Always verify `db` and `db.pool` exist before queries
- **Fallback to global**: Try global database instance if local unavailable

Example:
```python
if not db or not db.pool:
    try:
        from database import db as global_db
        if global_db and global_db.pool:
            db = global_db
    except:
        return  # Graceful degradation
```

### Query Patterns
- **Parameterized queries**: Always use `$1, $2` placeholders to prevent SQL injection
- **Async execution**: Use `await conn.fetch()` or `await conn.fetchval()`
- **Transaction management**: Wrap related operations in transactions when needed
- **Error handling**: Catch database exceptions and provide fallbacks

## WebSocket Patterns

### Connection Management
- **Connection pooling**: Store active connections in dictionaries by symbol
- **Connection state checks**: Verify connection state before sending
- **Automatic cleanup**: Remove failed connections from pool
- **Reconnection logic**: Retry with exponential backoff on failures

Example:
```python
self.active_connections[symbol][connection_id] = {
    'websocket': websocket,
    'timeframe': timeframe,
    'connected_at': datetime.now()
}
```

### Message Broadcasting
- **Batch sending**: Send same message to multiple connections efficiently
- **Semaphore limiting**: Limit concurrent sends to prevent overload
- **JSON serialization**: Use `json.dumps(data, default=str)` for datetime handling
- **Error isolation**: Failed sends don't affect other connections

### Stream Management
- **Individual streams per symbol**: Each symbol has dedicated WebSocket stream
- **Ping/pong for keepalive**: Configure `ping_interval` and `ping_timeout`
- **Compression disabled**: Set `compression=None` for Railway compatibility
- **Fallback to REST API**: Use REST endpoints when WebSocket fails

## ML Prediction Patterns

### Model Loading
- **Lazy loading**: Load model at startup, not on first request
- **Fallback model**: Provide simple fallback when main model unavailable
- **Model caching**: Cache loaded model in memory
- **Google Drive download**: Auto-download model from Drive if missing

Example:
```python
try:
    self.mobile_model = joblib.load(model_path)
except Exception as e:
    # Fallback model
    class FallbackModel:
        async def predict(self, symbol):
            return {'symbol': symbol, 'predicted_price': 100}
    model = FallbackModel()
```

### Feature Engineering
- **Deterministic seeds**: Use time-based seeds for reproducible predictions
- **Market-based features**: Calculate features from real market data
- **Volatility adjustment**: Adjust confidence based on market volatility
- **Multi-timeframe support**: Generate predictions for different timeframes

### Prediction Caching
- **Short TTL for real-time**: 1-3 second cache for live predictions
- **Memory + Redis**: Dual-layer caching for speed
- **Symbol-specific TTL**: Hot symbols have shorter cache duration
- **Timestamp tracking**: Track last prediction time per symbol

## API Design Patterns

### Route Organization
- **Modular routes**: Separate route files by functionality (market, forecast, trends, websocket, utility)
- **Setup functions**: Each route module exports `setup_*_routes(app, model, database)` function
- **Dependency injection**: Pass model and database instances to routes
- **Rate limiting**: Apply rate limiting to all public endpoints

Example:
```python
def setup_utility_routes(app: FastAPI, model, database):
    @app.get("/api/health")
    async def health_check():
        return {"status": "healthy"}
```

### Response Formatting
- **Consistent structure**: All responses use similar JSON structure
- **ISO timestamps**: Use `.isoformat()` for datetime serialization
- **Error responses**: Include error messages in response body
- **Type conversion**: Convert numpy/pandas types to native Python types

Example:
```python
return {
    'symbol': str(symbol),
    'current_price': float(current_price),
    'timestamp': datetime.now().isoformat()
}
```

### Request Validation
- **Query parameters**: Use FastAPI query parameters with defaults
- **Path parameters**: Use typed path parameters (e.g., `symbol: str`)
- **Request objects**: Pass `Request` object for rate limiting
- **Optional parameters**: Provide sensible defaults

## Configuration Management

### Environment Variables
- **python-dotenv**: Load environment variables with `load_dotenv()`
- **os.getenv with defaults**: Use `os.getenv('KEY', 'default')` pattern
- **Type conversion**: Convert string env vars to appropriate types
- **Centralized settings**: Use config/settings.py for configuration

Example:
```python
from dotenv import load_dotenv
load_dotenv()

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
```

### Symbol Configuration
- **Centralized symbol definitions**: Use config/symbols.py for all symbols
- **Symbol metadata**: Include display names, API mappings, classifications
- **Fixed prices**: Define fixed prices for stablecoins
- **API mappings**: Map internal symbols to external API symbols (Binance, Yahoo)

## Performance Optimization

### Concurrency Control
- **Semaphores**: Limit concurrent operations with `asyncio.Semaphore`
- **Connection pooling**: Reuse database and HTTP connections
- **Batch operations**: Group related operations together
- **Background tasks**: Use `asyncio.create_task()` for non-blocking work

### Memory Management
- **Limited cache size**: Cap in-memory cache entries
- **LRU eviction**: Remove least recently used entries when full
- **Cleanup tasks**: Periodic cleanup of completed tasks and old cache entries
- **Streaming responses**: Use `StreamingResponse` for large data exports

### Rate Limiting
- **Per-endpoint limits**: Different limits for different endpoint types
- **Time-based windows**: Track requests per time window
- **Graceful degradation**: Return cached data when rate limited
- **Minimal intervals**: Very short intervals (1ms) for real-time operations

## Testing and Debugging

### Debug Patterns
- **Conditional logging**: Use log levels to control verbosity
- **Print debugging**: Use print statements with emoji for visibility (🚀, ✅, ❌, ⚠️)
- **Flush output**: Use `flush=True` for immediate output
- **Error context**: Include symbol, operation type in error messages

Example:
```python
print(f"🚀 Starting Binance streams for {len(self.binance_symbols)} symbols...")
print(f"✅ Connected: {symbol} stream active", flush=True)
print(f"❌ {symbol} stream error: {error_msg}")
```

### Health Checks
- **Comprehensive health endpoint**: Check all services (database, Redis, ML model, APIs)
- **Pool statistics**: Include connection pool stats in health response
- **Degraded status**: Return "degraded" when non-critical services fail
- **Timestamp tracking**: Include timestamps in all health responses

## Security Best Practices

### Input Validation
- **FastAPI validation**: Use Pydantic models for request validation
- **Parameterized queries**: Never concatenate user input into SQL
- **Symbol validation**: Validate symbols against known list
- **Timeout limits**: Set timeouts on all external API calls

### Middleware Configuration
- **CORS middleware**: Configure allowed origins, methods, headers
- **TrustedHost middleware**: Restrict allowed hosts
- **Rate limiting**: Apply rate limits to prevent abuse
- **Error sanitization**: Don't expose internal errors to users

Example:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"]
)
```

## Common Code Idioms

### Dictionary Comprehensions
```python
self.binance_symbols = {k: v['binance'].lower() for k, v in CRYPTO_SYMBOLS.items() if v.get('binance')}
```

### Context Managers
```python
async with db.pool.acquire() as conn:
    result = await conn.fetch(query)
```

### List Comprehensions with Filtering
```python
matching_connections = [
    (connection_id, conn_data['websocket']) 
    for connection_id, conn_data in self.active_connections[symbol].items() 
    if conn_data['timeframe'] == timeframe
]
```

### Ternary Operators
```python
direction = 'UP' if xgb_prediction > 0.01 else 'DOWN' if xgb_prediction < -0.01 else 'HOLD'
```

### Default Dictionary Access
```python
cache_key = self.cache_keys.price(symbol, 'crypto')
cached_data = self.cache_manager.get_cache(cache_key)
if cached_data:
    return cached_data
```

## Frequently Used Annotations

### FastAPI Route Decorators
```python
@app.get("/api/health")
@app.post("/api/favorites/{symbol}")
@app.delete("/api/favorites/{symbol}")
```

### Async Context Managers
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
```

### Type Hints
```python
async def run_task(self, task_id: str, coro: Callable, *args, **kwargs) -> Any:
```

## Internal API Usage Patterns

### Multi-Asset Support
```python
from multi_asset_support import multi_asset

# Get asset data
asset_data = await multi_asset.get_asset_data(symbol)
current_price = asset_data['current_price']

# Get asset name
name = multi_asset.get_asset_name(symbol)

# Format predicted range
predicted_range = multi_asset.format_predicted_range(symbol, predicted_price)
```

### Cache Manager
```python
from utils.cache_manager import CacheManager, CacheKeys

# Get from cache
cache_key = CacheKeys.prediction(symbol)
cached_data = CacheManager.get_cache(cache_key)

# Set cache with TTL
CacheManager.set_cache(cache_key, data, ttl=60)
```

### Error Handler
```python
from utils.error_handler import ErrorHandler

# Log errors
ErrorHandler.log_stream_error('binance_message', symbol, str(e))
ErrorHandler.log_database_error('store_realtime', symbol, str(e))
ErrorHandler.log_prediction_error(symbol, f"Price fetch failed: {e}")
```

### API Client
```python
from utils.api_client import APIClient

# Get prices
price = await APIClient.get_binance_price(binance_symbol)
price = await APIClient.get_yahoo_price(yahoo_symbol)

# Get changes
change = await APIClient.get_binance_change(binance_symbol)
change = await APIClient.get_yahoo_change(yahoo_symbol)
```

## Startup and Lifecycle Management

### Lifespan Pattern
- **asynccontextmanager**: Use FastAPI lifespan for startup/shutdown
- **Service initialization**: Initialize all services at startup
- **Background tasks**: Start background tasks after initialization
- **Graceful shutdown**: Cancel tasks and close connections on shutdown

Example:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_database()
    background_tasks = [
        asyncio.create_task(service.start_streams())
    ]
    app.state.background_tasks = background_tasks
    
    yield
    
    # Shutdown
    for task in background_tasks:
        if not task.done():
            task.cancel()
```

### Initialization Order
1. Database connection
2. ML model loading
3. Cache initialization
4. Real-time services
5. Gap filling (blocking)
6. Background streams (non-blocking)

## Code Formatting Standards

### Line Length
- **Prefer readability**: Break long lines at logical points
- **Method chaining**: Break after dots for readability
- **Long strings**: Use f-strings with line breaks

### Whitespace
- **Two blank lines**: Between top-level functions and classes
- **One blank line**: Between methods in a class
- **No trailing whitespace**: Clean line endings
- **Spaces around operators**: `x = 1 + 2`, not `x=1+2`

### String Formatting
- **f-strings preferred**: Use f-strings for string interpolation (5/5 files)
- **Multi-line strings**: Use triple quotes for docstrings
- **JSON serialization**: Use `json.dumps()` with `default=str` for datetime handling

Example:
```python
message = f"✅ Connected: {symbol} stream active (Binance: {binance_symbol})"
```
