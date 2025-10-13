"""
Database models and operations for trading forecasts
"""
import asyncpg
import asyncio
from datetime import datetime
import json
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TradingDatabase:
    def __init__(self, database_url=None):
        self.database_url = database_url
        self.pool = None
        self.connection_status = 'disconnected'
        self.fallback_attempted = False
    
    async def connect(self):
        """Initialize database connection with automatic fallback"""
        # Try provided URL first
        if self.database_url:
            if await self._try_connect(self.database_url):
                return
        
        # Try Railway database
        railway_url = os.getenv('DATABASE_URL')
        if railway_url and await self._try_connect(railway_url):
            return
            
        # Try local database variations
        local_urls = [
            'postgresql://postgres:password@localhost:5432/trading_db',
            'postgresql://postgres@localhost:5432/trading_db',
            'postgresql://trading_user:password@localhost:5432/trading_db'
        ]
        
        for url in local_urls:
            if await self._try_connect(url):
                return
        
        logging.warning("No database connection available - running without persistence")
        self.connection_status = 'failed'
    
    async def _try_connect(self, url):
        """Try connecting to a specific database URL"""
        try:
            self.pool = await asyncpg.create_pool(
                url,
                min_size=2,
                max_size=10,
                command_timeout=5,
                server_settings={
                    'application_name': 'trading_app'
                }
            )
            await self.create_tables()
            self.database_url = url
            self.connection_status = 'connected'
            logging.info(f"Database connected: {url.split('@')[-1] if '@' in url else 'local'}")
            return True
        except Exception as e:
            if self.pool:
                await self.pool.close()
                self.pool = None
            return False
    
    def get_pool_stats(self):
        """Get connection pool statistics"""
        if not self.pool:
            return {
                'status': self.connection_status,
                'database_type': 'none',
                'url': 'no connection'
            }
        
        # Determine database type from URL
        db_type = 'local'
        if self.database_url:
            if 'railway' in self.database_url or 'tcp.railway.app' in self.database_url:
                db_type = 'railway'
            elif 'localhost' in self.database_url or '127.0.0.1' in self.database_url:
                db_type = 'local'
            else:
                db_type = 'remote'
        
        return {
            'size': self.pool.get_size(),
            'min_size': self.pool.get_min_size(),
            'max_size': self.pool.get_max_size(),
            'idle_size': self.pool.get_idle_size(),
            'status': self.connection_status,
            'database_type': db_type,
            'url': self.database_url.split('@')[-1] if '@' in self.database_url else 'local'
        }
    
    async def create_tables(self):
        """Create required tables"""
        async with self.pool.acquire() as conn:
            # Forecasts table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS forecasts (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(30) NOT NULL,
                    forecast_direction VARCHAR(10) NOT NULL,
                    confidence INTEGER NOT NULL,
                    predicted_price DECIMAL(15,2),
                    predicted_range VARCHAR(100),
                    trend_score INTEGER,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Actual prices table with OHLC data
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS actual_prices (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(30) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    open_price DECIMAL(15,2),
                    high DECIMAL(15,2),
                    low DECIMAL(15,2),
                    close_price DECIMAL(15,2),
                    price DECIMAL(15,2) NOT NULL,
                    change_24h DECIMAL(8,4),
                    volume DECIMAL(20,2),
                    timestamp TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Add missing columns if they don't exist (for existing databases)
            try:
                await conn.execute("ALTER TABLE actual_prices ADD COLUMN IF NOT EXISTS timeframe VARCHAR(10) DEFAULT '1D'")
                await conn.execute("ALTER TABLE actual_prices ADD COLUMN IF NOT EXISTS open_price DECIMAL(15,2)")
                await conn.execute("ALTER TABLE actual_prices ADD COLUMN IF NOT EXISTS high DECIMAL(15,2)")
                await conn.execute("ALTER TABLE actual_prices ADD COLUMN IF NOT EXISTS low DECIMAL(15,2)")
                await conn.execute("ALTER TABLE actual_prices ADD COLUMN IF NOT EXISTS close_price DECIMAL(15,2)")
                # Extend symbol column length for existing tables
                await conn.execute("ALTER TABLE actual_prices ALTER COLUMN symbol TYPE VARCHAR(30)")
                await conn.execute("ALTER TABLE forecasts ALTER COLUMN symbol TYPE VARCHAR(30)")
                await conn.execute("ALTER TABLE user_favorites ALTER COLUMN symbol TYPE VARCHAR(30)")
            except Exception as e:
                pass
            
            # Accuracy tracking table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS forecast_accuracy (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    forecast_id INTEGER REFERENCES forecasts(id) UNIQUE,
                    actual_direction VARCHAR(10),
                    result VARCHAR(10), -- 'Hit' or 'Miss'
                    accuracy_score DECIMAL(5,2),
                    evaluated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Favorites table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_favorites (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(50) DEFAULT 'default_user',
                    symbol VARCHAR(30) NOT NULL,
                    added_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(user_id, symbol)
                )
            """)
            
            # Create indexes and unique constraints
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_forecasts_symbol_time ON forecasts(symbol, created_at)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_actual_symbol_time ON actual_prices(symbol, timestamp)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_favorites_user ON user_favorites(user_id)")
            
            # Add unique constraint to prevent duplicate price entries
            try:
                await conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS unique_symbol_timestamp ON actual_prices (symbol, timestamp)")
            except:
                pass  # Index already exists
    
    async def store_forecast(self, db_key, forecast_data, timeframe='1D'):
        """Store forecast prediction - VALIDATES REAL DATA ONLY"""
        if not self.pool:
            return None
        
        # Validate forecast data before storing
        from utils.data_validator import data_validator
        if not data_validator.validate_forecast_data(forecast_data):
            logging.warning(f"Invalid forecast data rejected for {db_key}")
            return None
            
        async with self.pool.acquire() as conn:
            return await conn.fetchval("""
                INSERT INTO forecasts (symbol, forecast_direction, confidence, predicted_price, predicted_range, trend_score)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
            """, db_key, forecast_data['forecast_direction'], forecast_data['confidence'],
                forecast_data.get('predicted_price'), forecast_data.get('predicted_range'),
                forecast_data.get('trend_score'))
    
    async def store_actual_price(self, db_key, price_data, timeframe='1D'):
        """Store actual market price with OHLC data - VALIDATES REAL DATA ONLY"""
        if not self.pool:
            return
        
        # Validate data before storing
        from utils.data_validator import data_validator
        source = price_data.get('data_source', 'api')
        if not data_validator.validate_price_data(price_data, source):
            logging.warning(f"Invalid price data rejected for {db_key}")
            return
        
        if data_validator.is_synthetic_data(price_data):
            logging.error(f"Synthetic data detected and rejected for {db_key}")
            return
            
        async with self.pool.acquire() as conn:
            try:
                await conn.execute("""
                    INSERT INTO actual_prices (symbol, timeframe, open_price, high, low, close_price, price, change_24h, volume, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (symbol, timestamp) DO UPDATE SET
                        price = EXCLUDED.price,
                        change_24h = EXCLUDED.change_24h,
                        volume = EXCLUDED.volume,
                        high = GREATEST(actual_prices.high, EXCLUDED.high),
                        low = LEAST(actual_prices.low, EXCLUDED.low),
                        close_price = EXCLUDED.close_price
                """, db_key, timeframe, 
                    price_data.get('open_price'), price_data.get('high'), 
                    price_data.get('low'), price_data.get('close_price'),
                    price_data['current_price'], price_data.get('change_24h'), 
                    price_data.get('volume'), price_data.get('timestamp', datetime.now()))
            except Exception as e:
                # Silently handle duplicates for high-frequency data
                if "duplicate key" not in str(e).lower():
                    pass
    
    async def get_last_stored_time(self, symbol, timeframe='1D'):
        """Get last stored timestamp for a symbol with centralized format"""
        if not self.pool:
            return None
        
        # Use centralized symbol manager for consistent database keys
        from config.symbol_manager import symbol_manager
        db_symbol = symbol_manager.get_db_key(symbol, timeframe)
            
        async with self.pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT MAX(timestamp) FROM actual_prices WHERE symbol = $1
            """, db_symbol)
            return result
    
    async def store_historical_batch(self, symbol, historical_data, timeframe='1D'):
        """Store batch of historical data with OHLC using centralized symbol format - VALIDATES REAL DATA ONLY"""
        if not self.pool or not historical_data:
            return
        
        # Validate historical data before storing
        from utils.data_validator import data_validator
        source = historical_data[0].get('data_source', 'api') if historical_data else 'api'
        if not data_validator.validate_historical_data(historical_data, source):
            logging.warning(f"Invalid historical data rejected for {symbol}")
            return
        
        # Use centralized symbol manager for consistent database keys
        from config.symbol_manager import symbol_manager
        db_symbol = symbol_manager.get_db_key(symbol, timeframe)
            
        async with self.pool.acquire() as conn:
            # Insert batch with conflict handling
            for data in historical_data:
                try:
                    await conn.execute("""
                        INSERT INTO actual_prices (symbol, timeframe, open_price, high, low, close_price, price, change_24h, volume, timestamp)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        ON CONFLICT DO NOTHING
                    """, db_symbol, timeframe,
                        data.get('open', data.get('open_price')),
                        data.get('high'),
                        data.get('low'), 
                        data.get('close', data.get('close_price', data.get('price'))),
                        data.get('close', data.get('close_price', data.get('price'))),
                        data.get('change_24h', 0),
                        data.get('volume', 0),
                        data['timestamp'])
                except Exception as e:
                    pass
                    pass  # Skip duplicates
    
    async def get_historical_forecasts(self, symbol, days=30, timeframe='1D'):
        """Get historical forecasts for accuracy analysis with centralized symbol format"""
        if not self.pool:
            return []
        
        # Use centralized symbol manager for consistent database keys
        from config.symbol_manager import symbol_manager
        db_symbol = symbol_manager.get_db_key(symbol, timeframe)
            
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT f.*, fa.actual_direction, fa.result, fa.accuracy_score
                FROM forecasts f
                LEFT JOIN forecast_accuracy fa ON f.id = fa.forecast_id
                WHERE f.symbol = $1 AND f.created_at >= NOW() - INTERVAL '%s days'
                ORDER BY f.created_at DESC
            """ % days, db_symbol)
            
            # Return empty list if no data - NO SYNTHETIC DATA
            return [dict(row) for row in rows]
    
    async def calculate_accuracy(self, symbol, days=30, timeframe='1D'):
        """Calculate forecast accuracy percentage with centralized symbol format"""
        if not self.pool:
            return 0
        
        # Use centralized symbol manager for consistent database keys
        from config.symbol_manager import symbol_manager
        db_symbol = symbol_manager.get_db_key(symbol, timeframe)
            
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN fa.result = 'Hit' THEN 1 END) as hits
                FROM forecasts f
                JOIN forecast_accuracy fa ON f.id = fa.forecast_id
                WHERE f.symbol = $1 AND f.created_at >= NOW() - INTERVAL '%s days'
            """ % days, db_symbol)
            
            if result['total'] > 0:
                return round((result['hits'] / result['total']) * 100, 2)
            return 0
    
    async def get_chart_data(self, symbol, timeframe='7D'):
        """Get historical data for charts with enhanced Redis caching"""
        if not self.pool:
            return {'forecast': [], 'actual': [], 'timestamps': []}
        
        # Use centralized symbol manager for consistent database keys
        from config.symbol_manager import symbol_manager
        db_symbol = symbol_manager.get_db_key(symbol, timeframe)
        
        # Enhanced Redis cache with multiple TTLs
        cache_key = f"chart_data:{symbol}:{timeframe}"
        try:
            import redis
            import json
            import os
            from dotenv import load_dotenv
            load_dotenv()
            
            redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', '6379')),
                db=0,  # Use DB 0 for Railway compatibility (unified)
                password=os.getenv('REDIS_PASSWORD', None) if os.getenv('REDIS_PASSWORD') else None,
                decode_responses=True,
                socket_connect_timeout=10,
                socket_timeout=10
            )
            
            cached_data = redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            pass
            redis_client = None
            
        days = {'1D': 7, '7D': 7, '1M': 30, '1Y': 365, '4H': 7, '4h': 7, '5m': 7, '15m': 7, '30m': 7, '1h': 7, '1W': 30}.get(timeframe, 7)
        
        async with self.pool.acquire() as conn:
            # Get forecasts using centralized symbol format
            forecast_rows = await conn.fetch("""
                SELECT predicted_price, created_at
                FROM forecasts
                WHERE symbol = $1 AND created_at >= NOW() - INTERVAL '%s days'
                ORDER BY created_at
            """ % days, db_symbol)
            
            # Get actual prices using centralized symbol format
            actual_rows = await conn.fetch("""
                SELECT price, timestamp
                FROM actual_prices
                WHERE symbol = $1 AND timestamp >= NOW() - INTERVAL '%s days'
                ORDER BY timestamp
            """ % days, db_symbol)
            
            chart_data = {
                'forecast': [float(row['predicted_price']) for row in forecast_rows if row['predicted_price']],
                'actual': [float(row['price']) for row in actual_rows],
                'timestamps': [row['timestamp'].isoformat() for row in actual_rows]
            }
            
            # Enhanced caching with dynamic TTL
            if redis_client:
                try:
                    # Dynamic TTL based on timeframe and symbol popularity
                    hot_symbols = ['BTC', 'ETH', 'NVDA', 'AAPL', 'MSFT']
                    ttl_map = {'1m': 30, '5m': 60, '15m': 120, '1h': 300, '4H': 600, '1D': 900, '1W': 1800}
                    base_ttl = ttl_map.get(timeframe, 300)
                    ttl = base_ttl // 2 if symbol in hot_symbols else base_ttl
                    
                    redis_client.setex(cache_key, ttl, json.dumps(chart_data))
                except Exception as e:
                    pass
            
            return chart_data
    
    async def export_csv_data(self, symbol, timeframe='1M'):
        """Export historical data for CSV - returns only real data"""
        if not self.pool:
            return []
            
        days = {'1W': 7, '1M': 30, '1Y': 365, '5Y': 1825}.get(timeframe, 30)
        
        # Use centralized symbol manager for consistent database keys
        from config.symbol_manager import symbol_manager
        db_symbol = symbol_manager.get_db_key(symbol, timeframe)
        
        async with self.pool.acquire() as conn:
            # Get forecasts with accuracy data using centralized symbol format
            rows = await conn.fetch("""
                SELECT 
                    f.created_at::date as date,
                    f.forecast_direction as forecast,
                    fa.actual_direction as actual,
                    fa.result
                FROM forecasts f
                LEFT JOIN forecast_accuracy fa ON f.symbol = fa.symbol 
                    AND DATE(f.created_at) = DATE(fa.evaluated_at)
                WHERE f.symbol = $1 AND f.created_at >= NOW() - INTERVAL '%s days'
                ORDER BY f.created_at DESC
                LIMIT 100
            """ % days, db_symbol)
            
            # Return only real data - NO SYNTHETIC DATA
            # Filter out rows with missing actual data
            result_data = []
            for row in rows:
                row_dict = dict(row)
                # Only include rows with complete real data
                if row_dict['actual'] is not None and row_dict['result'] is not None:
                    result_data.append(row_dict)
            
            return result_data
    
    async def add_favorite(self, symbol, user_id='default_user'):
        """Add symbol to favorites"""
        if not self.pool:
            return False
        async with self.pool.acquire() as conn:
            try:
                await conn.execute(
                    "INSERT INTO user_favorites (user_id, symbol) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                    user_id, symbol
                )
                return True
            except:
                return False
    
    async def remove_favorite(self, symbol, user_id='default_user'):
        """Remove symbol from favorites"""
        if not self.pool:
            return False
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM user_favorites WHERE user_id = $1 AND symbol = $2",
                user_id, symbol
            )
            return True
    
    async def get_favorites(self, user_id='default_user'):
        """Get user's favorite symbols"""
        if not self.pool:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT symbol FROM user_favorites WHERE user_id = $1 ORDER BY added_at DESC",
                user_id
            )
            return [row['symbol'] for row in rows]

# Global database instance with auto-detection
db = TradingDatabase()