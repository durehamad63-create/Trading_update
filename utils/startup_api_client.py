import aiohttp
import asyncio
import logging

class StartupAPIClient:
    """Simplified API client for startup data collection without rate limiting"""
    
    def __init__(self):
        self._session = None
    
    async def _get_session(self):
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=60)
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def get_binance_historical(self, symbol, periods=200, interval='1h'):
        """Get historical data from Binance with specified interval"""
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={periods}"
            session = await self._get_session()
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    raise Exception(f"Binance API failed: {response.status}")
        except Exception as e:
            raise Exception(f"Binance historical data failed: {e}")
    
    async def get_yahoo_historical(self, symbol):
        """Get historical data from Yahoo without rate limiting"""
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=3mo"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            session = await self._get_session()
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    raise Exception(f"Yahoo API failed: {response.status}")
        except Exception as e:
            raise Exception(f"Yahoo historical data failed: {e}")

    async def get_yahoo_historical_custom(self, url):
        """Get historical data from Yahoo with custom URL"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            session = await self._get_session()
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    raise Exception(f"Yahoo API failed: {response.status}")
        except Exception as e:
            raise Exception(f"Yahoo historical data failed: {e}")

startup_api = StartupAPIClient()