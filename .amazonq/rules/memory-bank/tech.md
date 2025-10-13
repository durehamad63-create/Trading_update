# Trading AI Platform - Technology Stack

## Programming Language
- **Python 3.11+**: Primary language for all components

## Core Framework
- **FastAPI 0.104.1**: Modern async web framework for building APIs
  - High performance with async/await support
  - Automatic OpenAPI documentation
  - Built-in request validation with Pydantic
  - WebSocket support

## Web Server
- **Uvicorn 0.24.0**: ASGI server for FastAPI
  - Async request handling
  - WebSocket support
  - Production-ready performance

## Database
- **PostgreSQL 12+**: Primary data store
  - **asyncpg 0.29.0**: Async PostgreSQL driver
  - **psycopg2-binary 2.9.9**: Sync PostgreSQL driver (fallback)
  - Connection pooling (5-20 connections)
  - Transaction support

## Caching
- **Redis 6+**: Distributed cache
  - **redis 5.0.1**: Python Redis client
  - Multi-database support (prediction, ML, chart caches)
  - TTL-based expiration
  - Memory fallback when unavailable

## Machine Learning
- **XGBoost 2.0.2**: Gradient boosting framework for predictions
- **scikit-learn 1.6.1**: ML utilities and preprocessing
- **numpy 1.26.4**: Numerical computing
- **pandas 2.1.4**: Data manipulation and analysis
- **joblib 1.3.2**: Model serialization

## Data Sources & APIs
- **yfinance 0.2.33**: Yahoo Finance data (stocks)
- **fredapi 0.5.1**: Federal Reserve Economic Data (macro indicators)
- **Binance API**: Cryptocurrency real-time data (via websockets)
- **requests 2.31.0**: HTTP client for API calls
- **aiohttp 3.12.15**: Async HTTP client

## WebSocket & Real-Time
- **websockets 12.0**: WebSocket client/server implementation
- **aiosignal 1.4.0**: Async signal handling
- **aiohappyeyeballs 2.6.1**: Fast connection establishment

## Data Validation & Serialization
- **pydantic 2.11.8**: Data validation using Python type hints
- **pydantic_core 2.33.2**: Core validation logic

## Utilities
- **python-dotenv 1.0.0**: Environment variable management
- **python-dateutil 2.9.0.post0**: Date/time utilities
- **pytz 2025.2**: Timezone handling

## Visualization & Charting
- **matplotlib 3.10.6**: Chart generation
- **plotly 6.3.0**: Interactive charts

## Development & Testing
- **limits 5.5.0**: Rate limiting implementation
- **slowapi 0.1.9**: Rate limiting for FastAPI

## Additional Libraries
- **beautifulsoup4 4.13.5**: HTML parsing (for web scraping)
- **lxml 6.0.1**: XML/HTML processing
- **multitasking 0.0.12**: Parallel task execution
- **async-timeout 5.0.1**: Timeout utilities for async operations

## Containerization
- **Docker**: Container platform
  - Dockerfile for image building
  - Multi-stage builds for optimization
  - Environment variable injection

## Development Environment

### Prerequisites
```
Python 3.11+
PostgreSQL 12+
Redis 6+
Git
```

### Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

### Dependency Installation
```bash
pip install -r requirements.txt
```

## Build & Deployment

### Local Development
```bash
# Start Redis
redis-server

# Start PostgreSQL
# (varies by OS)

# Run application
python main.py
```

### Docker Deployment
```bash
# Build image
docker build -t trading-ai .

# Run container
docker run -p 8000:8000 --env-file .env trading-ai
```

### Production Deployment (Railway)
- Automatic deployment from Git repository
- Environment variables configured in Railway dashboard
- PostgreSQL and Redis provided as Railway services
- Internal networking for service communication

## Configuration Files

### requirements.txt
Complete list of Python dependencies with pinned versions for reproducibility.

### .env
Environment-specific configuration:
- Database URLs and credentials
- Redis connection details
- API keys (FRED)
- Cache TTL settings
- Application settings (debug, log level)

### Dockerfile
Multi-stage Docker build:
1. Base Python image
2. Dependency installation
3. Application code copy
4. Entrypoint configuration

## API Endpoints

### REST API
- `GET /api/market/summary` - Market overview
- `GET /api/asset/{symbol}/forecast` - Asset predictions
- `GET /api/asset/{symbol}/trends` - Historical accuracy
- `GET /api/assets/search` - Asset search
- `GET /api/health` - Health check

### WebSocket Endpoints
- `ws://localhost:8000/ws/asset/{symbol}/forecast` - Real-time forecasts
- `ws://localhost:8000/ws/asset/{symbol}/trends` - Live trends
- `ws://localhost:8000/ws/market/summary` - Market updates
- `ws://localhost:8000/ws/chart/{symbol}` - Chart data

## Performance Characteristics

### Response Times
- Cached predictions: <50ms
- Uncached predictions: 100-300ms
- WebSocket updates: Real-time (<100ms latency)
- Database queries: 10-50ms (with pooling)

### Scalability
- Async architecture supports 1000+ concurrent connections
- Connection pooling prevents database overload
- Redis caching reduces database load by 80%+
- Horizontal scaling via multiple instances

### Resource Usage
- Memory: ~200-500MB per instance
- CPU: Low (mostly I/O bound)
- Database connections: 5-20 per instance
- Redis connections: 1-3 per instance

## Security Features
- Rate limiting (configurable per endpoint)
- CORS middleware for cross-origin requests
- Trusted host middleware
- Environment variable protection
- No hardcoded credentials
- SQL injection prevention (parameterized queries)

## Monitoring & Logging
- Structured logging with levels (INFO, WARNING, ERROR)
- Health check endpoint for uptime monitoring
- Database pool statistics
- Cache hit/miss tracking
- Error tracking with stack traces

## Development Commands

### Run Server
```bash
python main.py
```

### Test API
```bash
# Health check
curl http://localhost:8000/api/health

# Market summary
curl "http://localhost:8000/api/market/summary?class=crypto&limit=5"

# Asset forecast
curl "http://localhost:8000/api/asset/BTC/forecast?timeframe=1D"
```

### Database Operations
```sql
-- Connect to database
psql -U postgres -d trading_db

-- Check tables
\dt

-- Query predictions
SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10;
```

### Redis Operations
```bash
# Connect to Redis
redis-cli

# Check keys
KEYS *

# Get cached value
GET prediction:BTC:1D
```

## Version Control
- **Git**: Source control
- **.gitignore**: Excludes .env, __pycache__, models, virtual environments
