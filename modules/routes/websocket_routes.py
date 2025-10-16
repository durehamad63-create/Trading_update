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
        logger.info(f"📊 Chart WebSocket connected: {symbol} (timeframe: {timeframe})")
        
        # Send initial connection success message
        try:
            await websocket.send_text(json.dumps({
                "type": "connected",
                "symbol": symbol,
                "timeframe": timeframe,
                "message": "WebSocket connected successfully"
            }))
            logger.info(f"✅ Sent connection message to client for {symbol}")
        except Exception as e:
            logger.error(f"❌ Failed to send connection message: {e}")
            return
        
        update_count = 0
        last_update_time = asyncio.get_event_loop().time()
        
        # Small delay to ensure client receives connected message
        await asyncio.sleep(0.5)
        
        try:
            while True:
                # Check connection state
                if websocket.client_state.name != 'CONNECTED':
                    break
                
                # Check for incoming messages (timeframe change or ping)
                try:
                    message = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                    data = json.loads(message)
                    
                    if data.get('type') == 'change_timeframe':
                        new_timeframe = WebSocketSecurity.validate_timeframe(data.get('timeframe', '1D'))
                        if new_timeframe != timeframe:
                            timeframe = new_timeframe
                            logger.info(f"📊 Timeframe changed to {timeframe} for {symbol}")
                            update_count = 0  # Reset counter
                            # Send acknowledgment
                            await websocket.send_text(json.dumps({
                                "type": "timeframe_changed",
                                "timeframe": timeframe,
                                "symbol": symbol
                            }))
                    elif data.get('type') == 'ping':
                        # Respond to ping
                        await websocket.send_text(json.dumps({"type": "pong"}))
                except asyncio.TimeoutError:
                    pass  # No message, continue with update
                except WebSocketDisconnect:
                    logger.info(f"🔌 Client disconnected: {symbol}")
                    break  # Exit loop on disconnect
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ Invalid JSON from client: {symbol}")
                    pass  # Ignore invalid JSON
                except Exception as e:
                    logger.error(f"❌ Message handling error for {symbol}: {e}")
                    break  # Exit on any other error
                
                update_count += 1
                
                # Map display timeframes to model timeframes
                model_timeframe = {'7D': '1W', '1Y': '1W', '5Y': '1M'}.get(timeframe, timeframe)
                
                # Get prediction with fallback
                try:
                    prediction = await model.predict(symbol, model_timeframe)
                except Exception as e:
                    logger.error(f"Prediction error for {symbol}: {e}")
                    # Send error message to client instead of closing
                    if websocket.client_state.name == 'CONNECTED':
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Prediction temporarily unavailable",
                            "retry_in": 5
                        }))
                    await asyncio.sleep(5)
                    continue
                
                # Get historical data from database
                past_prices = []
                timestamps = []
                
                if database and database.pool:
                    try:
                        from config.symbol_manager import symbol_manager
                        db_key = symbol_manager.get_db_key(symbol, model_timeframe)
                        
                        async with database.pool.acquire() as conn:
                            rows = await conn.fetch(
                                "SELECT price, timestamp FROM actual_prices WHERE symbol = $1 ORDER BY timestamp DESC LIMIT 30",
                                db_key
                            )
                            
                            past_prices = [float(row['price']) for row in reversed(rows)] if rows else []
                            timestamps = [row['timestamp'].isoformat() for row in reversed(rows)] if rows else []
                    except Exception as e:
                        logger.error(f"Database error for {symbol}: {e}")
                
                # Multi-step prediction with caching
                from datetime import timedelta
                current_price = prediction.get('current_price', 0)
                predicted_price = prediction.get('predicted_price', current_price)
                
                # Check if macro indicator
                macro_symbols = ['GDP', 'CPI', 'UNEMPLOYMENT', 'FED_RATE', 'CONSUMER_CONFIDENCE']
                is_macro = symbol in macro_symbols
                
                future_prices = []
                future_timestamps = []
                
                if is_macro:
                    # Macro: single prediction
                    future_prices = [predicted_price]
                    future_timestamps = []
                else:
                    # Use multi-step predictor based on MODEL timeframe
                    timeframe_steps = {
                        '1h': 12,  # 12 hourly predictions
                        '4h': 6,   # 6 4-hour predictions
                        '1D': 7,   # 7 daily predictions
                        '1W': 4,   # 4 weekly predictions
                        '1M': 3    # 3 monthly predictions
                    }
                    
                    num_steps = timeframe_steps.get(model_timeframe, 1)
                    
                    if num_steps > 1:
                        try:
                            from multistep_predictor import multistep_predictor
                            if multistep_predictor:
                                multistep_data = await multistep_predictor.get_multistep_forecast(symbol, model_timeframe, num_steps)
                                if multistep_data:
                                    future_prices = multistep_data['prices']
                                    future_timestamps = multistep_data['timestamps']
                                else:
                                    future_prices = [predicted_price]
                                    future_timestamps = []
                            else:
                                future_prices = [predicted_price]
                                future_timestamps = []
                        except Exception as e:
                            logger.warning(f"Multistep prediction failed for {symbol}: {e}, using single prediction")
                            future_prices = [predicted_price]
                            future_timestamps = []
                    else:
                        future_prices = [predicted_price]
                        future_timestamps = []
                
                # Combine timestamps
                if future_timestamps:
                    timestamps.extend(future_timestamps)
                
                forecast_direction = prediction.get('forecast_direction', 'HOLD')
                
                chart_update = {
                    "type": "chart_update",
                    "symbol": symbol,
                    "name": multi_asset.get_asset_name(symbol),
                    "timeframe": timeframe,
                    "prediction_steps": len(future_prices),
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
                    "data_source": "Multi-step ML prediction" if len(future_prices) > 1 else "Real-time ML prediction",
                    "prediction_updated": True,
                    "next_prediction_update": (WebSocketSecurity.get_utc_now() + timedelta(minutes=5)).isoformat(),
                    "forecast_stable": forecast_direction == 'HOLD',
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
                
                # Send chart update
                if websocket.client_state.name == 'CONNECTED':
                    try:
                        await websocket.send_text(json.dumps(chart_update))
                        last_update_time = asyncio.get_event_loop().time()
                    except Exception as e:
                        logger.error(f"❌ Failed to send chart update for {symbol}: {e}")
                        break
                
                # Check if connection is stale
                current_time = asyncio.get_event_loop().time()
                if current_time - last_update_time > 60:
                    logger.warning(f"⏰ Connection stale for {symbol}, closing")
                    break
        
        except WebSocketDisconnect:
            logger.info(f"❌ Chart WebSocket disconnected: {symbol}")
        except Exception as e:
            logger.error(f"⚠️ Chart WebSocket error for {symbol}: {e}", exc_info=True)
            # Try to send error to client before closing
            try:
                if websocket.client_state.name == 'CONNECTED':
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Server error occurred",
                        "details": str(e)
                    }))
            except:
                pass

        finally:
            logger.info(f"🔒 Chart WebSocket closed: {symbol} (timeframe: {timeframe})")
