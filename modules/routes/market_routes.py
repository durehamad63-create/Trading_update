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
        print(f"\n🔍 Market Summary Request: class={class_param}, limit={limit}")
        assets = []
        
        if class_param == "crypto":
            symbols = ['BTC', 'ETH', 'BNB', 'USDT', 'XRP', 'SOL', 'USDC', 'DOGE', 'ADA', 'TRX'][:limit]
            for symbol in symbols:
                try:
                    # Stablecoins: always use fixed price
                    if symbol in ['USDT', 'USDC']:
                        price_data = {
                            'current_price': 1.0,
                            'change_24h': 0.0,
                            'volume': 1000000000
                        }
                    else:
                        # Use cached price from realtime service
                        price_data = None
                        if realtime_service and symbol in realtime_service.price_cache:
                            price_data = realtime_service.price_cache[symbol]
                        
                        if not price_data:
                            cache_key = cache_keys.price(symbol, 'crypto')
                            price_data = cache_manager.get_cache(cache_key)
                        
                        if not price_data:
                            continue
                    
                    # Get prediction from cache (1D timeframe for market summary)
                    pred_cache_key = cache_keys.prediction(symbol, '1D')
                    prediction = cache_manager.get_cache(pred_cache_key)
                    print(f"  📊 {symbol}: Cache={'Hit' if prediction else 'Miss'}")
                    
                    if not prediction:
                        try:
                            print(f"  🔮 {symbol}: Generating prediction...")
                            prediction = await model.predict(symbol, '1D')
                            print(f"  ✅ {symbol}: {prediction.get('forecast_direction')} @ ${prediction.get('predicted_price')} (conf: {prediction.get('confidence')}%)")
                        except Exception as e:
                            print(f"  ❌ {symbol}: Prediction error - {e}")
                            continue
                    
                    if not prediction:
                        print(f"  ⚠️ {symbol}: No prediction")
                        continue
                    
                    # Use real model range values
                    current_price = price_data['current_price']
                    range_low = prediction.get('range_low', current_price)
                    range_high = prediction.get('range_high', current_price)
                    
                    # Format range based on price magnitude
                    if current_price >= 1000:
                        predicted_range = f"${range_low/1000:.1f}k–${range_high/1000:.1f}k"
                    elif current_price >= 1:
                        predicted_range = f"${range_low:.2f}–${range_high:.2f}"
                    else:
                        predicted_range = f"${range_low:.4f}–${range_high:.4f}"
                    
                    assets.append({
                        'symbol': symbol,
                        'name': multi_asset.get_asset_name(symbol),
                        'current_price': price_data['current_price'],
                        'change_24h': price_data.get('change_24h', 0),
                        'volume': price_data.get('volume', 0),
                        'forecast_direction': prediction['forecast_direction'],
                        'confidence': prediction['confidence'],
                        'predicted_price': prediction['predicted_price'],
                        'predicted_range': predicted_range,
                        'asset_class': 'crypto',
                        'timeframe': '1D'
                    })
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
                    continue
        
        elif class_param == "stocks":
            symbols = ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'AVGO', 'TSLA', 'BRK-B', 'JPM'][:limit]
            for symbol in symbols:
                try:
                    price_data = None
                    if stock_realtime_service and symbol in stock_realtime_service.price_cache:
                        price_data = stock_realtime_service.price_cache[symbol]
                    
                    if not price_data:
                        cache_key = cache_keys.price(symbol, 'stock')
                        price_data = cache_manager.get_cache(cache_key)
                    
                    if not price_data:
                        continue
                    
                    pred_cache_key = cache_keys.prediction(symbol, '1D')
                    prediction = cache_manager.get_cache(pred_cache_key)
                    
                    if not prediction:
                        try:
                            prediction = await model.predict(symbol, '1D')
                        except Exception:
                            continue
                    
                    if not prediction:
                        continue
                    
                    # Use real model range values
                    current_price = price_data['current_price']
                    range_low = prediction.get('range_low')
                    range_high = prediction.get('range_high')
                    
                    # Format range for stocks
                    if range_low and range_high:
                        predicted_range = f"${range_low:.2f}–${range_high:.2f}"
                    else:
                        predicted_range = None
                    
                    assets.append({
                        'symbol': symbol,
                        'name': multi_asset.get_asset_name(symbol),
                        'current_price': price_data['current_price'],
                        'change_24h': price_data.get('change_24h', 0),
                        'volume': price_data.get('volume', 0),
                        'forecast_direction': prediction['forecast_direction'],
                        'confidence': prediction['confidence'],
                        'predicted_price': prediction['predicted_price'],
                        'predicted_range': predicted_range,
                        'asset_class': 'stocks',
                        'timeframe': '1D'
                    })
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
                    continue
        
        elif class_param == "macro":
            symbols = ['GDP', 'CPI', 'UNEMPLOYMENT', 'FED_RATE', 'CONSUMER_CONFIDENCE']
            for symbol in symbols:
                try:
                    price_data = None
                    if macro_realtime_service and symbol in macro_realtime_service.price_cache:
                        price_data = macro_realtime_service.price_cache[symbol]
                    
                    if not price_data:
                        cache_key = cache_keys.price(symbol, 'macro')
                        price_data = cache_manager.get_cache(cache_key)
                    
                    if not price_data:
                        price_data = multi_asset._get_macro_data(symbol)
                    
                    if not price_data:
                        continue
                    
                    pred_cache_key = cache_keys.prediction(symbol, '1D')
                    prediction = cache_manager.get_cache(pred_cache_key)
                    
                    if not prediction:
                        try:
                            prediction = await model.predict(symbol, '1D')
                        except Exception:
                            continue
                    
                    if not prediction:
                        continue
                    
                    # Define update frequencies for macro indicators
                    macro_frequencies = {
                        'GDP': 'Quarterly',
                        'CPI': 'Monthly', 
                        'UNEMPLOYMENT': 'Monthly',
                        'FED_RATE': 'Every 6 weeks (FOMC meetings)',
                        'CONSUMER_CONFIDENCE': 'Monthly'
                    }
                    
                    # Use real model range values
                    range_low = prediction.get('range_low')
                    range_high = prediction.get('range_high')
                    
                    # Format based on indicator type
                    if range_low and range_high:
                        if symbol == 'GDP':
                            predicted_range = f"${range_low/1000:.1f}T–${range_high/1000:.1f}T"
                        elif symbol in ['UNEMPLOYMENT', 'FED_RATE']:
                            predicted_range = f"{range_low:.2f}%–{range_high:.2f}%"
                        else:
                            predicted_range = f"{range_low:.1f}–{range_high:.1f}"
                    else:
                        predicted_range = None
                    
                    assets.append({
                        'symbol': symbol,
                        'name': multi_asset.get_asset_name(symbol),
                        'current_price': price_data['current_price'],
                        'change_24h': price_data.get('change_24h', 0),
                        'forecast_direction': prediction['forecast_direction'],
                        'confidence': prediction['confidence'],
                        'predicted_price': prediction['predicted_price'],
                        'predicted_range': predicted_range,
                        'asset_class': 'macro',
                        'timeframe': '1D',
                        'change_frequency': macro_frequencies.get(symbol, 'Monthly')
                    })
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
                    continue
        
        elif class_param == "all":
            # Combine all asset classes
            all_symbols = [
                ('BTC', 'crypto'), ('ETH', 'crypto'), ('BNB', 'crypto'),
                ('NVDA', 'stocks'), ('MSFT', 'stocks'), ('AAPL', 'stocks'),
                ('GDP', 'macro'), ('CPI', 'macro')
            ][:limit]
            
            for symbol, asset_class in all_symbols:
                try:
                    # Stablecoins: always use fixed price
                    if symbol in ['USDT', 'USDC']:
                        price_data = {
                            'current_price': 1.0,
                            'change_24h': 0.0,
                            'volume': 1000000000
                        }
                    else:
                        price_data = None
                        
                        if asset_class == 'crypto' and realtime_service:
                            price_data = realtime_service.price_cache.get(symbol)
                        elif asset_class == 'stocks' and stock_realtime_service:
                            price_data = stock_realtime_service.price_cache.get(symbol)
                        elif asset_class == 'macro' and macro_realtime_service:
                            price_data = macro_realtime_service.price_cache.get(symbol)
                        
                        if not price_data:
                            cache_key = cache_keys.price(symbol, asset_class)
                            price_data = cache_manager.get_cache(cache_key)
                        
                        if not price_data:
                            continue
                    
                    pred_cache_key = cache_keys.prediction(symbol, '1D')
                    prediction = cache_manager.get_cache(pred_cache_key)
                    
                    if not prediction:
                        try:
                            prediction = await model.predict(symbol, '1D')
                        except Exception:
                            continue
                    
                    if not prediction:
                        continue
                    
                    # Define update frequencies for macro indicators
                    macro_frequencies = {
                        'GDP': 'Quarterly',
                        'CPI': 'Monthly', 
                        'UNEMPLOYMENT': 'Monthly',
                        'FED_RATE': 'Every 6 weeks (FOMC meetings)',
                        'CONSUMER_CONFIDENCE': 'Monthly'
                    }
                    
                    # Use real model range values
                    current_price = price_data['current_price']
                    range_low = prediction.get('range_low')
                    range_high = prediction.get('range_high')
                    
                    # Format range based on asset class
                    if range_low and range_high:
                        if asset_class == 'crypto':
                            if current_price >= 1000:
                                predicted_range = f"${range_low/1000:.1f}k–${range_high/1000:.1f}k"
                            elif current_price >= 1:
                                predicted_range = f"${range_low:.2f}–${range_high:.2f}"
                            else:
                                predicted_range = f"${range_low:.4f}–${range_high:.4f}"
                        elif asset_class == 'stocks':
                            predicted_range = f"${range_low:.2f}–${range_high:.2f}"
                        else:  # macro
                            if symbol == 'GDP':
                                predicted_range = f"${range_low/1000:.1f}T–${range_high/1000:.1f}T"
                            elif symbol in ['UNEMPLOYMENT', 'FED_RATE']:
                                predicted_range = f"{range_low:.2f}%–{range_high:.2f}%"
                            else:
                                predicted_range = f"{range_low:.1f}–{range_high:.1f}"
                    else:
                        predicted_range = None
                    
                    # Remove volume for macro indicators
                    asset_data = {
                        'symbol': symbol,
                        'name': multi_asset.get_asset_name(symbol),
                        'current_price': price_data['current_price'],
                        'change_24h': price_data.get('change_24h', 0),
                        'forecast_direction': prediction['forecast_direction'],
                        'confidence': prediction['confidence'],
                        'predicted_price': prediction['predicted_price'],
                        'predicted_range': predicted_range,
                        'asset_class': asset_class,
                        'timeframe': '1D'
                    }
                    
                    # Add volume only for crypto and stocks, not macro
                    if asset_class != 'macro':
                        asset_data['volume'] = price_data.get('volume', 0)
                    else:
                        # Add change_frequency for macro indicators
                        asset_data['change_frequency'] = macro_frequencies.get(symbol, 'Monthly')
                    
                    assets.append(asset_data)
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
                    continue
        
        return {"assets": assets, "total": len(assets)}
    
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
