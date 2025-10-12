"""WebSocket Routes - Real-time Data Streams"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import json
from datetime import datetime
from multi_asset_support import multi_asset

def setup_websocket_routes(app: FastAPI, model, database):
    import realtime_websocket_service as rws_module
    import stock_realtime_service as stock_module
    import macro_realtime_service as macro_module
    
    realtime_service = rws_module.realtime_service
    stock_realtime_service = stock_module.stock_realtime_service
    macro_realtime_service = macro_module.macro_realtime_service
    
    @app.websocket("/ws/asset/{symbol}/forecast")
    async def asset_forecast_websocket(websocket: WebSocket, symbol: str):
        await websocket.accept()
        
        crypto_symbols = ['BTC', 'ETH', 'BNB', 'USDT', 'XRP', 'SOL', 'USDC', 'DOGE', 'ADA', 'TRX']
        stock_symbols = ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'AVGO', 'TSLA', 'BRK-B', 'JPM']
        
        is_crypto = symbol in crypto_symbols
        is_stock = symbol in stock_symbols
        
        if not (is_crypto or is_stock):
            await websocket.close(code=1000, reason="Unsupported symbol")
            return
        
        service = realtime_service if is_crypto else stock_realtime_service
        
        if not service:
            await websocket.close(code=1000, reason="Service unavailable")
            return
        
        try:
            while True:
                await asyncio.sleep(2)
                
                price_data = service.price_cache.get(symbol)
                
                if not price_data:
                    api_data = await multi_asset.get_asset_data(symbol)
                    price_data = {
                        'current_price': api_data['current_price'],
                        'change_24h': api_data['change_24h'],
                        'volume': api_data.get('volume', 0)
                    }
                
                if price_data:
                    await websocket.send_text(json.dumps({
                        "type": "realtime_update",
                        "symbol": symbol,
                        "current_price": price_data['current_price'],
                        "change_24h": price_data['change_24h'],
                        "volume": price_data['volume'],
                        "timestamp": datetime.now().isoformat()
                    }))
        
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"WebSocket error for {symbol}: {e}")
    
    @app.websocket("/ws/market/summary")
    async def market_summary_websocket(websocket: WebSocket):
        await websocket.accept()
        
        try:
            while True:
                await asyncio.sleep(2)
                
                assets = []
                crypto_symbols = ['BTC', 'ETH', 'BNB']
                
                for symbol in crypto_symbols:
                    if realtime_service and symbol in realtime_service.price_cache:
                        price_data = realtime_service.price_cache[symbol]
                        assets.append({
                            'symbol': symbol,
                            'current_price': price_data['current_price'],
                            'change_24h': price_data['change_24h'],
                            'volume': price_data['volume']
                        })
                
                await websocket.send_text(json.dumps({
                    "type": "market_summary_update",
                    "assets": assets,
                    "timestamp": datetime.now().isoformat()
                }))
        
        except WebSocketDisconnect:
            pass
