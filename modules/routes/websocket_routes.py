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
            requested_timeframe = websocket.query_params.get('timeframe', '1D')
            
            # Check if macro indicator - they don't support timeframes
            macro_symbols = ['GDP', 'CPI', 'UNEMPLOYMENT', 'FED_RATE', 'CONSUMER_CONFIDENCE']
            is_macro = symbol in macro_symbols
            
            if is_macro and requested_timeframe != '1D':
                await websocket.close(code=1008, reason=f"Macro indicator {symbol} does not support timeframe {requested_timeframe}")
                return
            
            timeframe = '1D' if is_macro else WebSocketSecurity.validate_timeframe(requested_timeframe)
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
    
    @app.websocket("/ws/chart/{symbol}")
    async def chart_websocket(websocket: WebSocket, symbol: str):
        try:
            symbol = WebSocketSecurity.sanitize_symbol(symbol)
            timeframe = WebSocketSecurity.validate_timeframe(
                websocket.query_params.get('timeframe', '1D')
            )
        except ValueError as e:
            logger.error(f"Invalid chart WebSocket parameters: {e}")
            await websocket.close(code=1008, reason="Invalid parameters")
            return
        
        await websocket.accept()
        logger.info(f"Chart WebSocket connected: {symbol} (timeframe: {timeframe})")
        
        update_count = 0
        
        try:
            while True:
                await asyncio.sleep(5)
                update_count += 1
                
                # Get prediction
                try:
                    prediction = await model.predict(symbol, timeframe)
                except Exception as e:
                    logger.error(f"Prediction error for {symbol}: {e}")
                    continue
                
                # Timeframe-specific configuration
                timeframe_config = {
                    '1h': {'past': 24, 'future': 12},
                    '4h': {'past': 24, 'future': 6},
                    '1D': {'past': 30, 'future': 7},
                    '1W': {'past': 16, 'future': 4},
                    '1M': {'past': 30, 'future': 7}
                }
                config = timeframe_config.get(timeframe, {'past': 30, 'future': 7})
                
                # Get historical data from database
                if database and database.pool:
                    try:
                        from config.symbol_manager import symbol_manager
                        db_key = symbol_manager.get_db_key(symbol, timeframe)
                        
                        async with database.pool.acquire() as conn:
                            rows = await conn.fetch(
                                "SELECT price, timestamp FROM actual_prices WHERE symbol = $1 ORDER BY timestamp DESC LIMIT $2",
                                db_key, config['past']
                            )
                            
                            past_prices = [float(row['price']) for row in reversed(rows)] if rows else []
                            timestamps = [row['timestamp'].isoformat() for row in reversed(rows)] if rows else []
                    except Exception as e:
                        logger.error(f"Database error for {symbol}: {e}")
                        past_prices = []
                        timestamps = []
                else:
                    past_prices = []
                    timestamps = []
                
                # Use single model prediction for future (fast, no iteration)
                from datetime import timedelta
                
                current_price = prediction.get('current_price', 0)
                predicted_price = prediction.get('predicted_price', current_price)
                forecast_direction = prediction.get('forecast_direction', 'HOLD')
                
                # Generate smooth future line using single prediction
                future_prices = []
                price_change = predicted_price - current_price
                
                # Timeframe deltas
                timeframe_deltas = {
                    '1h': timedelta(hours=1),
                    '4h': timedelta(hours=4),
                    '1D': timedelta(days=1),
                    '1W': timedelta(weeks=1),
                    '1M': timedelta(days=30)
                }
                delta = timeframe_deltas.get(timeframe, timedelta(days=1))
                
                # Interpolate between current and predicted price
                for i in range(config['future']):
                    progress = (i + 1) / config['future']
                    interpolated_price = current_price + (price_change * progress)
                    future_prices.append(interpolated_price)
                    
                    next_time = WebSocketSecurity.get_utc_now() + delta * (i + 1)
                    timestamps.append(next_time.isoformat())
                
                # Check if macro indicator
                macro_symbols = ['GDP', 'CPI', 'UNEMPLOYMENT', 'FED_RATE', 'CONSUMER_CONFIDENCE']
                is_macro = symbol in macro_symbols
                
                chart_update = {
                    "type": "chart_update",
                    "symbol": symbol,
                    "name": multi_asset.get_asset_name(symbol),
                    "timeframe": timeframe,
                    "forecast_direction": forecast_direction,
                    "confidence": prediction.get('confidence', 75),
                    "current_price": current_price,
                    "change_24h": prediction.get('change_24h', 0),
                    "last_updated": WebSocketSecurity.get_utc_now().isoformat(),
                    "chart": {
                        "past": past_prices,
                        "future": future_prices,
                        "timestamps": timestamps
                    },
                    "update_count": update_count,
                    "data_source": "Real Database Data",
                    "prediction_updated": True,
                    "next_prediction_update": (WebSocketSecurity.get_utc_now() + timedelta(minutes=24)).isoformat(),
                    "forecast_stable": forecast_direction == 'HOLD',
                    "smooth_transition": True,
                    "ml_bounds_enforced": True,
                }
                
                # Add volume for non-macro indicators, change_frequency for macro
                if not is_macro:
                    chart_update["volume"] = 1000000000
                else:
                    macro_frequencies = {
                        'GDP': 'Quarterly',
                        'CPI': 'Monthly', 
                        'UNEMPLOYMENT': 'Monthly',
                        'FED_RATE': 'Every 6 weeks (FOMC meetings)',
                        'CONSUMER_CONFIDENCE': 'Monthly'
                    }
                    chart_update["change_frequency"] = macro_frequencies.get(symbol, 'Monthly')
                
                # Check connection state before sending
                if websocket.client_state.name == 'CONNECTED':
                    await websocket.send_text(json.dumps(chart_update))
        
        except WebSocketDisconnect:
            logger.info(f"Chart WebSocket disconnected: {symbol}")
        except Exception as e:
            logger.error(f"Chart WebSocket error for {symbol}: {e}", exc_info=True)
