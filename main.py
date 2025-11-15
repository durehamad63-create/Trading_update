"""
Modular Mobile Trading Server
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import logging
import sys
import os
from contextlib import asynccontextmanager

# Enable API request logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.getLogger('uvicorn.access').setLevel(logging.INFO)
logging.getLogger('asyncio').setLevel(logging.ERROR)

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
sys.path.append(os.path.dirname(__file__))

from modules.ml_predictor import MobileMLModel
from modules.routes import setup_all_routes

# Initialize database with auto-detection
print("🔄 Setting up database connection...")
try:
    from database import TradingDatabase
    db = TradingDatabase()
    print("✅ Database instance created")
except Exception as e:
    print(f"❌ Database setup failed: {e}")
    db = None

async def init_database():
    global db
    if db:
        try:
            await db.connect()
            stats = db.get_pool_stats()
            print(f"✅ Database connected: {stats['database_type']} ({stats['url']})")
        except Exception as e:
            print(f"⚠️ Database connection failed: {e}")
            print("⚠️ Running without database persistence")

# Load real ML model at startup - REQUIRED
print("🔄 Loading ML Model...")
try:
    model = MobileMLModel()
    print("✅ ML Model loaded successfully")
except Exception as e:
    print(f"❌ CRITICAL: ML Model failed to load: {e}")
    print("❌ Application cannot start without ML model")
    sys.exit(1)

# Initialize multi-step predictor
from multistep_predictor import init_multistep_predictor
init_multistep_predictor(model)
print("✅ Multi-step predictor initialized")

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """INSTANT startup - streams start immediately"""
    import asyncio
    import asyncio
    
    # Initialize services INSTANTLY without waiting
    from realtime_websocket_service import RealTimeWebSocketService
    from stock_realtime_service import StockRealtimeService
    from macro_realtime_service import MacroRealtimeService
    import realtime_websocket_service as rws_module
    import stock_realtime_service as stock_module
    import macro_realtime_service as macro_module
    
    # Setup services with database connection
    realtime_service = RealTimeWebSocketService(model, db)
    stock_service = StockRealtimeService(model, db)
    macro_service = MacroRealtimeService(model, db)
    
    # Make services available IMMEDIATELY
    rws_module.realtime_service = realtime_service
    stock_module.stock_realtime_service = stock_service
    macro_module.macro_realtime_service = macro_service
    
    # Give services a moment to start populating caches
    await asyncio.sleep(2)
    
    # Let services populate their own caches
    print("🚀 Services initialized, gap filling first...")
    
    # DISABLE STREAMS UNTIL GAP FILLING COMPLETES
    background_tasks = []
    
    # Database connection and gap filling (blocking)
    async def setup_database():
        try:
            await init_database()
            
            # Only proceed if database is connected
            if db and db.pool and db.connection_status == 'connected':
                # Clean database on startup if env var set
                clean_on_startup = os.getenv('CLEAN_DB_ON_STARTUP', 'false').lower() == 'true'
                if clean_on_startup:
                    try:
                        print("🧹 Cleaning database on startup...")
                        async with db.pool.acquire() as conn:
                            await conn.execute("TRUNCATE TABLE actual_prices, forecasts, forecast_accuracy RESTART IDENTITY CASCADE")
                        print("✅ Database cleaned")
                    except Exception as e:
                        print(f"⚠️ Database cleanup failed: {e}")
                        print("ℹ️ Continuing without cleanup...")
                
                print("🔄 Starting gap filling...")
                from gap_filling_service import GapFillingService
                gap_filler = GapFillingService(model)
                await gap_filler.fill_missing_data(db)
                print("✅ Gap filling completed")
                
                            # NOW START REAL-TIME SERVICES AFTER GAP FILLING
                print("🚀 Starting real-time services...")
                background_tasks.extend([
                    asyncio.create_task(realtime_service.start_binance_streams()),
                    asyncio.create_task(stock_service.start_stock_streams()),
                    asyncio.create_task(macro_service.start_macro_streams()),
                    asyncio.create_task(periodic_gap_filling())
                ])
                print("✅ Real-time services started")
                print("✅ Periodic gap filling scheduled (every 7 days)")
            else:
                print("⚠️ Skipping gap filling - no database connection")
                print("ℹ️ Application will run with in-memory data only")
                # Still add periodic gap filling in case database comes online
                background_tasks.append(asyncio.create_task(periodic_gap_filling()))
        except Exception as e:
            print(f"⚠️ Database setup failed: {e}")
            print("⚠️ Application will run with in-memory data only")
            # Still add periodic gap filling in case database comes online
            background_tasks.append(asyncio.create_task(periodic_gap_filling()))
    
    # Initialize database and gap filling (blocking)
    await setup_database()
    
    app.state.background_tasks = background_tasks
    print(f"📊 Total background tasks: {len(background_tasks)}")
    
    yield
    
    # Cleanup
    print("🛑 Shutting down background tasks...")
    for task in background_tasks:
        if not task.done():
            task.cancel()
    print("✅ All background tasks stopped")


app = FastAPI(title="Mobile Trading AI", lifespan=lifespan)

# Add CORS middleware FIRST (order matters!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600
)

# Add security middleware AFTER CORS
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]
)

# Setup API routes with model and database
setup_all_routes(app, model, db)

if __name__ == "__main__":
    import uvicorn
    import asyncio
    import sys
    
    # Fix Windows asyncio issues
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    print("🚀 Starting Trading AI Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")