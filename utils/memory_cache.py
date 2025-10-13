"""Thread-safe LRU memory cache"""
from collections import OrderedDict
from datetime import datetime
import asyncio

class LRUCache:
    def __init__(self, maxsize=1000):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.lock = asyncio.Lock()
    
    async def get(self, key):
        async with self.lock:
            if key in self.cache:
                item = self.cache.pop(key)
                if (datetime.now() - item['timestamp']).total_seconds() < item['ttl']:
                    self.cache[key] = item
                    return item['value']
                return None
            return None
    
    async def set(self, key, value, ttl=300):
        async with self.lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)
            
            self.cache[key] = {
                'value': value,
                'timestamp': datetime.now(),
                'ttl': ttl
            }
    
    async def clear(self):
        async with self.lock:
            self.cache.clear()
