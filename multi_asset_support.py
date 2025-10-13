"""
Multi-asset support for stocks and macro indicators
"""
import aiohttp
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from dotenv import load_dotenv
from config.symbols import CRYPTO_SYMBOLS, STOCK_SYMBOLS, MACRO_SYMBOLS, SYMBOL_NAMES
from utils.api_client import APIClient
from utils.error_handler import ErrorHandler

load_dotenv()

class MultiAssetSupport:
    def __init__(self):
        self.crypto_symbols = list(CRYPTO_SYMBOLS.keys())
        self.stock_symbols = list(STOCK_SYMBOLS.keys())
        self.macro_symbols = list(MACRO_SYMBOLS.keys())
    
    async def get_asset_data(self, symbol):
        """Get current price and change for any asset type - ASYNC"""
        if symbol in self.crypto_symbols:
            return await self._get_crypto_data(symbol)
        elif symbol in self.stock_symbols:
            return await self._get_stock_data(symbol)
        elif symbol in self.macro_symbols:
            return self._get_macro_data(symbol)
        else:
            raise Exception(f"Unsupported symbol: {symbol}")
    
    async def _get_crypto_data(self, symbol):
        """Get crypto data using centralized API client - ASYNC"""
        if symbol not in CRYPTO_SYMBOLS:
            raise Exception(f"Crypto symbol not supported: {symbol}")
        
        config = CRYPTO_SYMBOLS[symbol]
        
        # Handle stablecoins
        if config.get('fixed_price'):
            return {
                'current_price': config['fixed_price'],
                'change_24h': 0.0,
                'data_source': 'Fixed Price'
            }
        
        # Try Binance first
        if config.get('binance'):
            price = await APIClient.get_binance_price(config['binance'])
            change = await APIClient.get_binance_change(config['binance'])
            if price:
                return {
                    'current_price': price,
                    'change_24h': change,
                    'data_source': 'Binance API'
                }
        
        # Fallback to Yahoo
        if config.get('yahoo'):
            price = await APIClient.get_yahoo_price(config['yahoo'])
            change = await APIClient.get_yahoo_change(config['yahoo'])
            if price:
                return {
                    'current_price': price,
                    'change_24h': change,
                    'data_source': 'Yahoo Finance API'
                }
        
        raise Exception(f"No data source available for {symbol}")
    
    async def _get_stock_data(self, symbol):
        """Get stock data using centralized API client - ASYNC"""
        if symbol not in STOCK_SYMBOLS:
            raise Exception(f"Stock symbol not supported: {symbol}")
        
        config = STOCK_SYMBOLS[symbol]
        price = await APIClient.get_yahoo_price(config['yahoo'])
        change = await APIClient.get_yahoo_change(config['yahoo'])
        
        if price:
            return {
                'current_price': price,
                'change_24h': change,
                'data_source': 'Yahoo Finance API'
            }
        
        raise Exception(f"Stock data unavailable for {symbol}")
    

    
    def _get_macro_data(self, symbol):
        """Get macro indicator data using centralized configuration"""
        if symbol not in MACRO_SYMBOLS:
            raise Exception(f"Macro symbol not supported: {symbol}")
        
        config = MACRO_SYMBOLS[symbol]
        return {
            'current_price': config['value'],
            'change_24h': config['change'],
            'data_source': 'Economic Data'
        }
    
    async def get_historical_data(self, symbol, periods=100):
        """Get historical data for any asset type - ASYNC"""
        if symbol in self.crypto_symbols:
            return await self._get_crypto_historical(symbol, periods)
        elif symbol in self.stock_symbols:
            return await self._get_stock_historical(symbol, periods)
        elif symbol in self.macro_symbols:
            return self._get_macro_historical(symbol, periods)
        else:
            raise Exception(f"Unsupported symbol: {symbol}")
    
    async def _get_crypto_historical(self, symbol, periods):
        """Get crypto historical data from Binance - ASYNC"""
        if symbol not in CRYPTO_SYMBOLS:
            raise Exception(f"Crypto symbol not supported: {symbol}")
        
        config = CRYPTO_SYMBOLS[symbol]
        binance_symbol = config.get('binance')
        if not binance_symbol:
            raise Exception(f"No Binance mapping for {symbol}")
        
        import aiohttp
        binance_url = os.getenv('BINANCE_API_URL', 'https://api.binance.com/api/v3')
        url = f"{binance_url}/klines?symbol={binance_symbol}&interval=1h&limit={periods}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status == 200:
                    data = await response.json()
                    prices = [float(kline[4]) for kline in data]  # Close price
                    volumes = [float(kline[5]) for kline in data]  # Volume
                    return np.array(prices), np.array(volumes)
                else:
                    raise Exception(f"Binance API failed: {response.status}")
    
    async def _get_stock_historical(self, symbol, periods):
        """Get stock historical data using direct Yahoo Finance API - ASYNC"""
        
        try:
            import aiohttp
            # Use 1d interval for better data availability
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=3mo"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15), headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = data['chart']['result'][0]
                        
                        # Extract price and volume data
                        timestamps = result['timestamp']
                        indicators = result['indicators']['quote'][0]
                        
                        closes = [x for x in indicators['close'] if x is not None]
                        volumes = [x for x in indicators['volume'] if x is not None]
                        
                        if len(closes) == 0:
                            raise Exception(f"No valid price data for {symbol}")
                        
                        # Return last 'periods' data points
                        prices = np.array(closes[-periods:])
                        vols = np.array(volumes[-periods:] if len(volumes) >= len(prices) else [1000000] * len(prices))
                        
                        return prices, vols
                    else:
                        raise Exception(f"Yahoo API failed: {response.status}")
                
        except Exception as e:
            raise Exception(f"Stock historical data failed for {symbol}: {str(e)}")
    
    def _get_macro_historical(self, symbol, periods):
        """Get real macro economic data from FRED API"""
        try:
            from fredapi import Fred
            fred_api_key = os.getenv('FRED_API_KEY')
            
            if not fred_api_key:
                raise Exception("FRED_API_KEY not configured")
            
            fred = Fred(api_key=fred_api_key)
            
            # FRED series IDs for each indicator
            fred_series = {
                'GDP': 'GDP',  # Gross Domestic Product
                'CPI': 'CPIAUCSL',  # Consumer Price Index
                'UNEMPLOYMENT': 'UNRATE',  # Unemployment Rate
                'FED_RATE': 'FEDFUNDS',  # Federal Funds Rate
                'CONSUMER_CONFIDENCE': 'UMCSENT'  # Consumer Sentiment
            }
            
            series_id = fred_series.get(symbol)
            if not series_id:
                raise Exception(f"No FRED series for {symbol}")
            
            # Get real data from FRED
            data = fred.get_series(series_id, observation_start=datetime.now() - timedelta(days=periods*30))
            
            if data is None or len(data) == 0:
                raise Exception(f"No FRED data available for {symbol}")
            
            # Convert to numpy arrays
            prices = np.array(data.values[-periods:])
            volumes = np.ones(len(prices)) * 1000000  # Placeholder volume
            
            return prices, volumes
            
        except Exception as e:
            logging.error(f"FRED API failed for {symbol}: {e}")
            raise Exception(f"Cannot get real macro data for {symbol}: FRED API failed - {e}")
    
    def format_predicted_range(self, symbol, predicted_price):
        """Format predicted range based on asset type"""
        if symbol == 'GDP':
            return f'${predicted_price*0.98:.0f}B–${predicted_price*1.02:.0f}B'
        elif symbol in ['CPI', 'CONSUMER_CONFIDENCE']:
            return f'{predicted_price*0.98:.1f}–{predicted_price*1.02:.1f}'
        elif symbol in ['UNEMPLOYMENT', 'FED_RATE']:
            return f'{predicted_price*0.98:.2f}%–{predicted_price*1.02:.2f}%'
        else:
            return f'${predicted_price*0.98:.2f}–${predicted_price*1.02:.2f}'
    
    def get_asset_name(self, symbol):
        """Get full name for asset using centralized configuration"""
        return SYMBOL_NAMES.get(symbol, symbol)

# Global instance
multi_asset = MultiAssetSupport()