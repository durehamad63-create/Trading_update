"""
Admin Routes - Database and Service Management
"""
from fastapi import FastAPI
from datetime import datetime
import asyncio

def setup_admin_routes(app: FastAPI, model, database):
    db = database
    
    @app.get("/api/admin")
    async def admin_panel():
        """Admin panel endpoints list"""
        return {
            "endpoints": {
                "database": {
                    "clean": "POST /api/admin/database/clean - Clean all database tables",
                    "status": "GET /api/admin/database/status - Get database status and counts",
                    "clean_symbol": "DELETE /api/admin/database/symbol/{symbol} - Clean specific symbol data"
                },
                "gap_filling": {
                    "restart": "POST /api/admin/gap-filling/restart - Restart gap filling service",
                    "status": "GET /api/admin/gap-filling/status - Get gap filling status"
                },
                "services": {
                    "restart": "POST /api/admin/services/restart - Restart all real-time services"
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/api/admin/database/status")
    async def database_status():
        """Get database status and record counts"""
        if not db or not db.pool:
            return {"success": False, "error": "Database not available"}
        
        async with db.pool.acquire() as conn:
            actual_count = await conn.fetchval("SELECT COUNT(*) FROM actual_prices")
            forecast_count = await conn.fetchval("SELECT COUNT(*) FROM forecasts")
            accuracy_count = await conn.fetchval("SELECT COUNT(*) FROM forecast_accuracy")
            
            symbol_counts = await conn.fetch(
                "SELECT symbol, COUNT(*) as count FROM actual_prices GROUP BY symbol ORDER BY count DESC"
            )
            
            return {
                "success": True,
                "tables": {
                    "actual_prices": actual_count,
                    "forecasts": forecast_count,
                    "forecast_accuracy": accuracy_count
                },
                "symbol_distribution": [{"symbol": r["symbol"], "count": r["count"]} for r in symbol_counts],
                "timestamp": datetime.now().isoformat()
            }
    
    @app.post("/api/admin/database/clean")
    async def clean_database():
        """Clean all database tables and restart gap filling"""
        if not db or not db.pool:
            return {"success": False, "error": "Database not available"}
        
        async with db.pool.acquire() as conn:
            await conn.execute("TRUNCATE TABLE actual_prices, forecasts, forecast_accuracy RESTART IDENTITY CASCADE")
            
            try:
                await conn.execute("ALTER SEQUENCE actual_prices_id_seq RESTART WITH 1")
                await conn.execute("ALTER SEQUENCE forecasts_id_seq RESTART WITH 1")
                await conn.execute("ALTER SEQUENCE forecast_accuracy_id_seq RESTART WITH 1")
            except:
                pass
        
        from gap_filling_service import gap_filler
        asyncio.create_task(gap_filler.fill_missing_data(db))
        
        return {
            "success": True,
            "message": "Database cleaned and gap filling service restarted",
            "timestamp": datetime.now().isoformat()
        }
    
    @app.delete("/api/admin/database/symbol/{symbol}")
    async def clean_symbol_data(symbol: str):
        """Clean data for specific symbol"""
        if not db or not db.pool:
            return {"success": False, "error": "Database not available"}
        
        from config.symbol_manager import symbol_manager
        
        async with db.pool.acquire() as conn:
            timeframes = ['1H', '4H', '1D', '1W', '1M']
            deleted_count = 0
            
            for tf in timeframes:
                db_symbol = symbol_manager.get_db_key(symbol, tf)
                
                actual_deleted = await conn.fetchval(
                    "DELETE FROM actual_prices WHERE symbol = $1 RETURNING COUNT(*)", db_symbol
                )
                forecast_deleted = await conn.fetchval(
                    "DELETE FROM forecasts WHERE symbol = $1 RETURNING COUNT(*)", db_symbol
                )
                accuracy_deleted = await conn.fetchval(
                    "DELETE FROM forecast_accuracy WHERE symbol = $1 RETURNING COUNT(*)", db_symbol
                )
                
                deleted_count += (actual_deleted or 0) + (forecast_deleted or 0) + (accuracy_deleted or 0)
        
        return {
            "success": True,
            "symbol": symbol,
            "deleted_records": deleted_count,
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/api/admin/gap-filling/status")
    async def gap_filling_status():
        """Get gap filling service status"""
        if not db or not db.pool:
            return {"success": False, "error": "Database not available"}
        
        async with db.pool.acquire() as conn:
            recent_data = await conn.fetchval(
                "SELECT COUNT(*) FROM actual_prices WHERE timestamp > NOW() - INTERVAL '10 minutes'"
            )
            
            total_symbols = await conn.fetchval(
                "SELECT COUNT(DISTINCT symbol) FROM actual_prices"
            )
            
            return {
                "success": True,
                "recent_insertions": recent_data,
                "total_symbols_with_data": total_symbols,
                "expected_symbols": 125,
                "coverage_percentage": round((total_symbols / 125) * 100, 1) if total_symbols else 0,
                "timestamp": datetime.now().isoformat()
            }
    
    @app.post("/api/admin/gap-filling/restart")
    async def restart_gap_filling():
        """Restart gap filling service"""
        from gap_filling_service import gap_filler
        asyncio.create_task(gap_filler.fill_missing_data(db))
        
        return {
            "success": True,
            "message": "Gap filling service restarted",
            "timestamp": datetime.now().isoformat()
        }
    
    @app.post("/api/admin/services/restart")
    async def restart_services():
        """Restart all real-time services"""
        from modules.api_routes import realtime_service, stock_realtime_service, macro_realtime_service
        
        restart_tasks = []
        
        if realtime_service and hasattr(realtime_service, 'restart'):
            restart_tasks.append(realtime_service.restart())
        
        if stock_realtime_service and hasattr(stock_realtime_service, 'restart'):
            restart_tasks.append(stock_realtime_service.restart())
        
        if macro_realtime_service and hasattr(macro_realtime_service, 'restart'):
            restart_tasks.append(macro_realtime_service.restart())
        
        if restart_tasks:
            await asyncio.gather(*restart_tasks, return_exceptions=True)
        
        return {
            "success": True,
            "message": "All services restart initiated",
            "timestamp": datetime.now().isoformat()
        }
