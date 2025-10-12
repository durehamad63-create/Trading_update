import aiohttp
import asyncio
import logging

class APIClient:
    _session = None
    
    @classmethod
    async def get_session(cls):
        if cls._session is None or cls._session.closed:
            cls._session = aiohttp.ClientSession()
        return cls._session
    
    @classmethod
    async def close_session(cls):
        if cls._session and not cls._session.closed:
            await cls._session.close()
            cls._session = None
    
    @staticmethod
    async def get_binance_price(symbol):
        """Get price from Binance API - ASYNC"""
        try:
            session = await APIClient.get_session()
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data['price'])
        except Exception as e:
            logging.warning(f"Binance price API error for {symbol}: {e}")
        return None
    
    @staticmethod
    async def get_binance_change(symbol):
        """Get 24h change from Binance API - ASYNC"""
        try:
            session = await APIClient.get_session()
            url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data['priceChangePercent'])
        except Exception as e:
            logging.warning(f"Binance change API error for {symbol}: {e}")
        return 0.0
    
    @staticmethod
    async def get_yahoo_price(symbol):
        """Get price from Yahoo Finance API - ASYNC"""
        try:
            session = await APIClient.get_session()
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30), headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'chart' in data and data['chart']['result']:
                        return float(data['chart']['result'][0]['meta']['regularMarketPrice'])
        except Exception as e:
            logging.warning(f"Yahoo price API error for {symbol}: {e}")
        return None
    
    @staticmethod
    async def get_yahoo_change(symbol):
        """Get 24h change from Yahoo Finance API - ASYNC"""
        try:
            session = await APIClient.get_session()
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30), headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'chart' in data and data['chart']['result']:
                        meta = data['chart']['result'][0]['meta']
                        current = meta['regularMarketPrice']
                        prev = meta['previousClose']
                        return ((current - prev) / prev) * 100
        except Exception as e:
            logging.warning(f"Yahoo change API error for {symbol}: {e}")
        return 0.0
    
    # Sync fallback for compatibility
    @staticmethod
    def get_binance_price_sync(symbol):
        try:
            import requests
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return float(response.json()['price'])
        except Exception:
            pass
        return None