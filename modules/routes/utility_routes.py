"""
Utility Routes - Search, Favorites, Export, Health Check
"""
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from datetime import datetime
import io

def setup_utility_routes(app: FastAPI, model, database):
    db = database
    
    from rate_limiter import rate_limiter
    from multi_asset_support import multi_asset
    from utils.cache_manager import CacheManager
    cache_manager = CacheManager
    
    @app.get("/api/assets/search")
    async def search_assets(query: str):
        """Search available assets"""
        assets = [
            {'symbol': 'BTC', 'name': 'Bitcoin', 'class': 'crypto'},
            {'symbol': 'ETH', 'name': 'Ethereum', 'class': 'crypto'},
            {'symbol': 'BNB', 'name': 'Binance Coin', 'class': 'crypto'},
            {'symbol': 'USDT', 'name': 'Tether', 'class': 'crypto'},
            {'symbol': 'XRP', 'name': 'Ripple', 'class': 'crypto'},
            {'symbol': 'SOL', 'name': 'Solana', 'class': 'crypto'},
            {'symbol': 'USDC', 'name': 'USD Coin', 'class': 'crypto'},
            {'symbol': 'DOGE', 'name': 'Dogecoin', 'class': 'crypto'},
            {'symbol': 'ADA', 'name': 'Cardano', 'class': 'crypto'},
            {'symbol': 'TRX', 'name': 'Tron', 'class': 'crypto'},
            {'symbol': 'NVDA', 'name': 'NVIDIA', 'class': 'stocks'},
            {'symbol': 'MSFT', 'name': 'Microsoft', 'class': 'stocks'},
            {'symbol': 'AAPL', 'name': 'Apple', 'class': 'stocks'},
            {'symbol': 'GOOGL', 'name': 'Google', 'class': 'stocks'},
            {'symbol': 'AMZN', 'name': 'Amazon', 'class': 'stocks'},
            {'symbol': 'META', 'name': 'Meta', 'class': 'stocks'},
            {'symbol': 'AVGO', 'name': 'Broadcom', 'class': 'stocks'},
            {'symbol': 'TSLA', 'name': 'Tesla', 'class': 'stocks'},
            {'symbol': 'BRK-B', 'name': 'Berkshire Hathaway', 'class': 'stocks'},
            {'symbol': 'JPM', 'name': 'JPMorgan Chase', 'class': 'stocks'},
            {'symbol': 'GDP', 'name': 'Gross Domestic Product', 'class': 'macro'},
            {'symbol': 'CPI', 'name': 'Consumer Price Index', 'class': 'macro'},
            {'symbol': 'UNEMPLOYMENT', 'name': 'Unemployment Rate', 'class': 'macro'},
            {'symbol': 'FED_RATE', 'name': 'Federal Interest Rate', 'class': 'macro'},
            {'symbol': 'CONSUMER_CONFIDENCE', 'name': 'Consumer Confidence Index', 'class': 'macro'}
        ]
        
        query_lower = query.lower()
        results = [
            asset for asset in assets
            if query_lower in asset['symbol'].lower() or query_lower in asset['name'].lower()
        ]
        
        if not results:
            results = [
                asset for asset in assets
                if any(query_lower in word.lower() for word in asset['name'].split()) or
                asset['symbol'].lower().startswith(query_lower)
            ]
        
        return {'results': results}
    
    @app.post("/api/favorites/{symbol}")
    async def add_favorite(symbol: str, request: Request = None):
        """Add symbol to favorites"""
        if request:
            await rate_limiter.check_rate_limit(request)
        
        if not db or not db.pool:
            return {"success": False, "symbol": symbol, "error": "Database not available"}
        
        success = await db.add_favorite(symbol)
        return {"success": success, "symbol": symbol}
    
    @app.delete("/api/favorites/{symbol}")
    async def remove_favorite(symbol: str, request: Request = None):
        """Remove symbol from favorites"""
        if request:
            await rate_limiter.check_rate_limit(request)
        
        if not db or not db.pool:
            return {"success": False, "symbol": symbol, "error": "Database not available"}
        
        success = await db.remove_favorite(symbol)
        return {"success": success, "symbol": symbol}
    
    @app.get("/api/favorites")
    async def get_favorites(request: Request = None):
        """Get user's favorite symbols"""
        if request:
            await rate_limiter.check_rate_limit(request)
        
        if not db or not db.pool:
            return {"favorites": [], "error": "Database not available"}
        
        favorites = await db.get_favorites()
        return {"favorites": favorites}
    
    @app.get("/api/asset/{symbol}/export")
    async def export_data(symbol: str, timeframe: str = "1M", request: Request = None):
        """Export historical data as CSV"""
        if request:
            await rate_limiter.check_rate_limit(request, 'export')
        
        if not db or not db.pool:
            csv_data = "Date,Forecast,Actual,Result\n"
            csv_data += "Database not available\n"
            return StreamingResponse(
                io.StringIO(csv_data),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={symbol}_export.csv"}
            )
        
        export_data = await db.export_csv_data(symbol, timeframe)
        csv_data = "Date,Forecast,Actual,Result\n"
        
        for record in export_data:
            date = record['date'].strftime('%Y-%m-%d') if record['date'] else 'N/A'
            forecast = record['forecast'] or 'N/A'
            actual = record['actual'] or 'N/A'
            result = record['result'] or 'N/A'
            csv_data += f"{date},{forecast},{actual},{result}\n"
        
        if len(export_data) == 0:
            csv_data += "No historical data available\n"
        
        return StreamingResponse(
            io.StringIO(csv_data),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={symbol}_export.csv"}
        )
    
    @app.get("/api/asset/{symbol}/trends/export")
    async def export_trends_data(symbol: str, timeframe: str = "1M", request: Request = None):
        """Export trends historical data as CSV"""
        if request:
            await rate_limiter.check_rate_limit(request, 'export')
        
        if not db or not db.pool:
            csv_data = "Date,Forecast,Actual,Result,Accuracy\n"
            csv_data += "Database not available\n"
            return StreamingResponse(
                io.StringIO(csv_data),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={symbol}_trends.csv"}
            )
        
        export_data = await db.export_csv_data(symbol, timeframe)
        csv_data = "Date,Forecast,Actual,Result,Accuracy\n"
        
        for record in export_data:
            date = record['date'].strftime('%Y-%m-%d') if record['date'] else 'N/A'
            forecast = record['forecast'] or 'N/A'
            actual = record['actual'] or 'N/A'
            result = record['result'] or 'N/A'
            accuracy = "Hit" if result == "Hit" else "Miss" if result == "Miss" else "N/A"
            csv_data += f"{date},{forecast},{actual},{result},{accuracy}\n"
        
        if len(export_data) == 0:
            csv_data += "No historical trends data available\n"
        
        return StreamingResponse(
            io.StringIO(csv_data),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={symbol}_trends.csv"}
        )
    
    @app.get("/api/health")
    async def health_check():
        """System health check"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {}
        }
        
        # Check database
        if db and db.pool:
            try:
                async with db.pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                pool_stats = db.get_pool_stats()
                health_status["services"]["database"] = f"connected ({pool_stats['database_type']})"
                health_status["db_pool_stats"] = pool_stats
                health_status["database_url"] = pool_stats['url']
            except Exception as e:
                health_status["services"]["database"] = f"error: {str(e)}"
                health_status["status"] = "degraded"
        else:
            health_status["services"]["database"] = "disconnected"
            health_status["database_url"] = "none"
        
        # Check Redis
        redis_status = []
        if cache_manager and hasattr(cache_manager, 'redis_client'):
            try:
                cache_manager.redis_client.ping()
                redis_status.append("centralized_cache")
            except:
                pass
        
        if hasattr(model, 'cache_manager') and model.cache_manager:
            try:
                model.cache_manager.redis_client.ping()
                redis_status.append("ml_cache")
            except:
                pass
        
        health_status["services"]["redis"] = f"connected ({', '.join(redis_status)})" if redis_status else "memory_cache_fallback"
        
        # Check ML model
        if model:
            try:
                test_prediction = await model.predict('BTC')
                health_status["services"]["ml_model"] = "operational"
                health_status["model_type"] = "XGBoost" if hasattr(model, 'xgb_model') and model.xgb_model else "Enhanced Technical"
                health_status["cache_type"] = "Redis" if hasattr(model, 'redis_client') and model.redis_client else "Memory"
            except Exception as e:
                health_status["services"]["ml_model"] = f"error: {str(e)}"
                health_status["status"] = "degraded"
        else:
            health_status["services"]["ml_model"] = "not available"
        
        # Check external APIs
        try:
            await multi_asset.get_asset_data('BTC')
            health_status["services"]["external_apis"] = "operational"
        except Exception as e:
            health_status["services"]["external_apis"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        return health_status
    
    @app.get("/api/system/stats")
    async def system_stats():
        """Get system performance statistics"""
        return {
            'timestamp': datetime.now().isoformat()
        }
