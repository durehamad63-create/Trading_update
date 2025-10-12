"""Market Summary Routes - Real Data Only"""
from fastapi import FastAPI, Request
from multi_asset_support import multi_asset
from modules.rate_limiter import rate_limiter
from utils.cache_manager import CacheManager, CacheKeys

# Global services
realtime_service = None
stock_realtime_service = None
macro_realtime_service = None

def setup_market_routes(app: FastAPI, model, database):
    cache_manager = CacheManager
    cache_keys = CacheKeys
    
    @app.get("/api/market/summary")
    async def market_summary(request: Request, limit: int = 10):
        await rate_limiter.check_rate_limit(request)
        
        class_param = request.query_params.get('class', 'crypto')
        assets = []
        
        if class_param == "crypto":
            symbols = ['BTC', 'ETH', 'BNB', 'USDT', 'XRP', 'SOL', 'USDC', 'DOGE', 'ADA', 'TRX'][:limit]
            for symbol in symbols:
                cache_key = cache_keys.price(symbol, 'crypto')
                price_data = cache_manager.get_cache(cache_key)
                
                if not price_data and realtime_service:
                    price_data = realtime_service.price_cache.get(symbol)
                
                if not price_data:
                    api_data = await multi_asset.get_asset_data(symbol)
                    price_data = {
                        'current_price': api_data['current_price'],
                        'change_24h': api_data['change_24h'],
                        'volume': api_data.get('volume', 0)
                    }
                
                if price_data:
                    prediction = await model.predict(symbol)
                    assets.append({
                        'symbol': symbol,
                        'name': multi_asset.get_asset_name(symbol),
                        'current_price': price_data['current_price'],
                        'change_24h': price_data['change_24h'],
                        'volume': price_data['volume'],
                        'forecast_direction': prediction.get('forecast_direction', 'HOLD'),
                        'confidence': prediction.get('confidence', 75),
                        'asset_class': 'crypto'
                    })
        
        elif class_param == "stocks":
            symbols = ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'AVGO', 'TSLA', 'BRK-B', 'JPM'][:limit]
            for symbol in symbols:
                cache_key = cache_keys.price(symbol, 'stock')
                price_data = cache_manager.get_cache(cache_key)
                
                if not price_data and stock_realtime_service:
                    price_data = stock_realtime_service.price_cache.get(symbol)
                
                if not price_data:
                    api_data = await multi_asset.get_asset_data(symbol)
                    price_data = {
                        'current_price': api_data['current_price'],
                        'change_24h': api_data['change_24h'],
                        'volume': api_data.get('volume', 0)
                    }
                
                if price_data:
                    prediction = await model.predict(symbol)
                    assets.append({
                        'symbol': symbol,
                        'name': multi_asset.get_asset_name(symbol),
                        'current_price': price_data['current_price'],
                        'change_24h': price_data['change_24h'],
                        'volume': price_data['volume'],
                        'forecast_direction': prediction.get('forecast_direction', 'HOLD'),
                        'confidence': prediction.get('confidence', 75),
                        'asset_class': 'stocks'
                    })
        
        elif class_param == "macro":
            symbols = ['GDP', 'CPI', 'UNEMPLOYMENT', 'FED_RATE', 'CONSUMER_CONFIDENCE']
            for symbol in symbols:
                cache_key = cache_keys.price(symbol, 'macro')
                price_data = cache_manager.get_cache(cache_key)
                
                if not price_data and macro_realtime_service:
                    price_data = macro_realtime_service.price_cache.get(symbol)
                
                if not price_data:
                    price_data = multi_asset._get_macro_data(symbol)
                
                if price_data:
                    prediction = await model.predict(symbol)
                    assets.append({
                        'symbol': symbol,
                        'name': multi_asset.get_asset_name(symbol),
                        'current_price': price_data['current_price'],
                        'change_24h': price_data['change_24h'],
                        'volume': price_data.get('volume', 0),
                        'forecast_direction': prediction.get('forecast_direction', 'HOLD'),
                        'confidence': prediction.get('confidence', 75),
                        'asset_class': 'macro'
                    })
        
        return {"assets": assets}
    
    @app.get("/api/assets/search")
    async def search_assets(query: str):
        assets = [
            {'symbol': 'BTC', 'name': 'Bitcoin', 'class': 'crypto'},
            {'symbol': 'ETH', 'name': 'Ethereum', 'class': 'crypto'},
            {'symbol': 'NVDA', 'name': 'NVIDIA', 'class': 'stocks'},
            {'symbol': 'AAPL', 'name': 'Apple', 'class': 'stocks'},
        ]
        query_lower = query.lower()
        results = [a for a in assets if query_lower in a['symbol'].lower() or query_lower in a['name'].lower()]
        return {'results': results}
