# Trading AI Platform - Project Structure

## Directory Organization

```
Trading/
├── main.py                           # FastAPI application entry point
├── database.py                       # Database operations and connection management
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Docker containerization
├── .env                             # Environment configuration
│
├── config/                          # Configuration management
│   ├── settings.py                  # Application settings and environment variables
│   ├── symbols.py                   # Asset symbol definitions and classifications
│   └── symbol_manager.py            # Symbol validation and management
│
├── modules/                         # Core business logic
│   ├── routes/                      # API endpoint definitions
│   │   ├── __init__.py             # Route registration
│   │   ├── market_routes.py        # Market summary endpoints
│   │   ├── forecast_routes.py      # Asset forecast endpoints
│   │   ├── trends_routes.py        # Historical trends endpoints
│   │   ├── websocket_routes.py     # WebSocket connection handlers
│   │   ├── utility_routes.py       # Health checks and utilities
│   │   └── admin_routes.py         # Administrative endpoints
│   │
│   ├── ml_predictor.py             # ML prediction engine (XGBoost)
│   ├── accuracy_validator.py       # Prediction accuracy tracking
│   ├── data_validator.py           # Data validation and sanitization
│   └── rate_limiter.py             # API rate limiting
│
├── utils/                           # Utility functions
│   ├── api_client.py               # External API client (Binance, Yahoo Finance)
│   ├── api_coordinator.py          # Multi-source API coordination
│   ├── cache_manager.py            # Redis and memory cache management
│   ├── database_manager.py         # Database utility functions
│   ├── error_handler.py            # Centralized error handling
│   ├── timestamp_utils.py          # Time and timezone utilities
│   └── startup_api_client.py       # Startup-time API initialization
│
├── models/                          # ML model storage
│   └── specialized_trading_model.pkl  # Pre-trained XGBoost model
│
├── realtime_websocket_service.py    # Cryptocurrency real-time streaming
├── stock_realtime_service.py        # Stock real-time streaming
├── macro_realtime_service.py        # Macro indicator streaming
├── multi_asset_support.py           # Multi-asset data fetching
├── async_task_manager.py            # Background task management
├── gap_filling_service.py           # Historical data gap filling
│
└── test_*.py / test_*.html          # Testing utilities and interfaces
```

## Core Components

### 1. Application Entry (main.py)
- FastAPI application initialization
- Middleware configuration (CORS, TrustedHost)
- Lifespan management for startup/shutdown
- Service initialization and coordination
- Background task management

**Key Responsibilities**:
- Database connection setup
- ML model loading with fallback
- Real-time service initialization
- Gap filling orchestration
- Route registration

### 2. Database Layer (database.py)
- PostgreSQL connection pooling
- Auto-detection of database type (local/Railway)
- Async database operations
- Connection health monitoring
- Pool statistics tracking

**Key Features**:
- Configurable pool size (5-20 connections)
- Automatic reconnection
- Transaction management
- Query execution with error handling

### 3. ML Prediction Engine (modules/ml_predictor.py)
- XGBoost model loading and inference
- Feature engineering for predictions
- Confidence scoring based on volatility
- Multi-asset prediction support
- Deterministic random seeds for consistency

**Prediction Flow**:
1. Load current price data
2. Engineer features (moving averages, volatility)
3. Run XGBoost inference
4. Calculate confidence scores
5. Return structured prediction

### 4. Real-Time Services
Three specialized services for different asset classes:

**realtime_websocket_service.py** (Crypto):
- Binance WebSocket connections
- Live price streaming for 10 cryptocurrencies
- Automatic reconnection on failures
- In-memory cache for latest prices

**stock_realtime_service.py** (Stocks):
- Yahoo Finance data integration
- Polling-based updates for 10 stocks
- Market hours awareness
- Price normalization

**macro_realtime_service.py** (Macro):
- FRED API integration
- Economic indicator tracking
- Scheduled updates (daily/weekly)
- Historical data management

### 5. API Routes (modules/routes/)
Modular route organization by functionality:

- **market_routes.py**: Market summaries with class filtering
- **forecast_routes.py**: Asset-specific predictions with chart data
- **trends_routes.py**: Historical accuracy analysis
- **websocket_routes.py**: Real-time WebSocket endpoints
- **utility_routes.py**: Health checks, search, system status
- **admin_routes.py**: Administrative operations

### 6. Caching System (utils/cache_manager.py)
Multi-tier caching strategy:

**Redis (Primary)**:
- Distributed cache for production
- Multiple databases for different data types
- TTL-based expiration
- Connection pooling

**Memory (Fallback)**:
- In-process cache when Redis unavailable
- Hot symbol prioritization
- LRU eviction policy
- Thread-safe operations

### 7. Data Management

**gap_filling_service.py**:
- Identifies missing historical data
- Backfills gaps using ML predictions
- Batch processing for efficiency
- Database transaction management

**multi_asset_support.py**:
- Unified interface for multi-source data
- Asset class detection
- Data normalization
- Error handling per source

**async_task_manager.py**:
- Background task scheduling
- Task lifecycle management
- Error recovery
- Resource cleanup

## Architectural Patterns

### Layered Architecture
```
Presentation Layer (FastAPI Routes)
    ↓
Business Logic Layer (Modules)
    ↓
Data Access Layer (Database, Cache)
    ↓
External Services (Binance, Yahoo, FRED)
```

### Service-Oriented Design
- Independent services for crypto, stocks, macro
- Loose coupling through dependency injection
- Shared database and model instances
- Centralized error handling

### Async/Await Pattern
- Non-blocking I/O throughout
- Concurrent request handling
- Background task execution
- Efficient resource utilization

### Dependency Injection
- Model and database passed to routes
- Services initialized at startup
- Testable component design
- Flexible configuration

### Caching Strategy
- Cache-aside pattern
- TTL-based invalidation
- Hot data prioritization
- Graceful degradation

## Data Flow

### Prediction Request Flow
```
Client Request → API Route → Rate Limiter → Cache Check
    ↓ (cache miss)
ML Predictor → Feature Engineering → XGBoost Model
    ↓
Prediction Result → Cache Store → Response
```

### Real-Time Update Flow
```
External API (Binance/Yahoo) → WebSocket/Polling
    ↓
Real-Time Service → Price Normalization
    ↓
Cache Update → Database Insert → WebSocket Broadcast
```

### Gap Filling Flow
```
Database Query → Identify Gaps → Batch Processing
    ↓
ML Predictions → Data Validation → Bulk Insert
    ↓
Cache Invalidation → Completion Report
```

## Configuration Management

### Environment Variables (.env)
- Database connection strings
- Redis configuration
- API keys (FRED)
- Cache TTL settings
- Rate limiting parameters
- WebSocket intervals

### Symbol Configuration (config/symbols.py)
- Asset classifications (crypto, stocks, macro)
- Symbol validation rules
- Display names and metadata
- Trading pair mappings

### Settings Module (config/settings.py)
- Centralized configuration loading
- Environment variable parsing
- Default value management
- Type validation
