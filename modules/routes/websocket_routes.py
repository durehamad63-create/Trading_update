"""WebSocket Routes - Real-time Data Streams"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import json
import uuid
import logging
from multi_asset_support import multi_asset
from config.symbols import CRYPTO_SYMBOLS, STOCK_SYMBOLS
from utils.websocket_security import WebSocketSecurity

logger = logging.getLogger(__name__)

def setup_websocket_routes(app: FastAPI, model, database):
    import realtime_websocket_service as rws_module
    import stock_realtime_service as stock_module
    import macro_realtime_service as macro_module
    
    @app.websocket("/ws/asset/{symbol}/forecast")
    async def asset_forecast_websocket(websocket: WebSocket, symbol: str):
        try:
            # Sanitize inputs
            symbol = WebSocketSecurity.sanitize_symbol(symbol)
            timeframe = WebSocketSecurity.validate_timeframe(
                websocket.query_params.get('timeframe', '1D')
            )
        except ValueError as e:
            logger.error(f"Invalid WebSocket parameters: {e}")
            await websocket.close(code=1008, reason="Invalid parameters")
            return
        
        await websocket.accept()
        logger.info(f"WebSocket connected: {symbol} (timeframe: {timeframe})")
        
        crypto_symbols = ['BTC', 'ETH', 'BNB', 'USDT', 'XRP', 'SOL', 'USDC', 'DOGE', 'ADA', 'TRX']
        stock_symbols = ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'AVGO', 'TSLA', 'BRK-B', 'JPM']
        macro_symbols = ['GDP', 'CPI', 'UNEMPLOYMENT', 'FED_RATE', 'CONSUMER_CONFIDENCE']
        
        is_crypto = symbol in crypto_symbols
        is_stock = symbol in stock_symbols
        is_macro = symbol in macro_symbols
        
        if not (is_crypto or is_stock or is_macro):
            logger.warning(f"Unsupported symbol: {symbol}")
            await websocket.close(code=1008, reason="Unsupported symbol")
            return
        
        # Get the appropriate service
        if is_crypto:
            service = rws_module.realtime_service
        elif is_stock:
            service = stock_module.stock_realtime_service
        else:
            service = macro_module.macro_realtime_service
        
        if not service:
            logger.error(f"Service not available for {symbol}")
            await websocket.close(code=1011, reason="Service initializing")
            return
        
        connection_id = str(uuid.uuid4())
        
        try:
            # Add connection to service
            await service.add_connection(websocket, symbol, connection_id, timeframe)
            logger.info(f"Connection added for {symbol}")
            
            # Keep connection alive
            while True:
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
        
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {symbol}")
        except Exception as e:
            logger.error(f"WebSocket error for {symbol}: {e}", exc_info=True)
        finally:
            # Always cleanup connection
            try:
                service.remove_connection(symbol, connection_id)
            except Exception as cleanup_error:
                logger.error(f"Cleanup error for {symbol}: {cleanup_error}")
    
    @app.websocket("/ws/market/summary")
    async def market_summary_websocket(websocket: WebSocket):
        await websocket.accept()
        logger.info("Market summary WebSocket connected")
        
        try:
            while True:
                await asyncio.sleep(2)
                
                assets = []
                crypto_symbols = ['BTC', 'ETH', 'BNB']
                
                realtime_service = rws_module.realtime_service
                
                if realtime_service:
                    for symbol in crypto_symbols:
                        if symbol in realtime_service.price_cache:
                            price_data = realtime_service.price_cache[symbol]
                            assets.append({
                                'symbol': WebSocketSecurity.sanitize_string(symbol),
                                'current_price': WebSocketSecurity.safe_float(price_data.get('current_price', 0)),
                                'change_24h': WebSocketSecurity.safe_float(price_data.get('change_24h', 0)),
                                'volume': WebSocketSecurity.safe_float(price_data.get('volume', 0))
                            })
                
                await websocket.send_text(json.dumps({
                    "type": "market_summary_update",
                    "assets": assets,
                    "timestamp": WebSocketSecurity.get_utc_now().isoformat()
                }))
        
        except WebSocketDisconnect:
            logger.info("Market summary WebSocket disconnected")
        except Exception as e:
            logger.error(f"Market summary WebSocket error: {e}", exc_info=True)
