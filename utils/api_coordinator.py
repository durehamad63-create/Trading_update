import asyncio
import time
from collections import defaultdict

class APICoordinator:
    def __init__(self):
        self.locks = defaultdict(asyncio.Lock)
        self.last_call = defaultdict(float)
        self.min_interval = 2  # Minimum 2 seconds between calls to same endpoint
    
    async def coordinate_call(self, api_name, symbol, call_func, *args, **kwargs):
        """Coordinate API calls to prevent conflicts"""
        key = f"{api_name}_{symbol}"
        
        async with self.locks[key]:
            # Ensure minimum interval between calls
            now = time.time()
            time_since_last = now - self.last_call[key]
            
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                await asyncio.sleep(wait_time)
            
            try:
                result = await call_func(*args, **kwargs)
                self.last_call[key] = time.time()
                return result
            except Exception as e:
                self.last_call[key] = time.time()
                raise e

api_coordinator = APICoordinator()