#!/usr/bin/env python3
"""
API Performance Test Script
Tests all endpoints and measures response times
"""
import asyncio
import aiohttp
import time
from datetime import datetime

API_BASE = "http://localhost:8000"

async def test_endpoint(session, name, url, method="GET"):
    """Test a single endpoint and measure response time"""
    start = time.time()
    try:
        if method == "GET":
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                elapsed = time.time() - start
                status = response.status
                if status == 200:
                    data = await response.json()
                    print(f"✅ {name}: {elapsed:.2f}s (status={status}, size={len(str(data))} bytes)")
                    return True, elapsed
                else:
                    print(f"❌ {name}: {elapsed:.2f}s (status={status})")
                    return False, elapsed
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        print(f"⏰ {name}: TIMEOUT after {elapsed:.2f}s")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ {name}: ERROR after {elapsed:.2f}s - {e}")
        return False, elapsed

async def main():
    print("=" * 60)
    print("API Performance Test")
    print(f"Testing: {API_BASE}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    
    async with aiohttp.ClientSession() as session:
        tests = [
            ("Health Check", f"{API_BASE}/api/health"),
            ("Market Summary (All)", f"{API_BASE}/api/market/summary?limit=20"),
            ("Market Summary (Crypto)", f"{API_BASE}/api/market/summary?class=crypto&limit=10"),
            ("Market Summary (Stocks)", f"{API_BASE}/api/market/summary?class=stocks&limit=10"),
            ("Market Summary (Macro)", f"{API_BASE}/api/market/summary?class=macro&limit=5"),
            ("BTC Forecast", f"{API_BASE}/api/asset/BTC/forecast?timeframe=1D"),
            ("ETH Forecast", f"{API_BASE}/api/asset/ETH/forecast?timeframe=1D"),
            ("NVDA Forecast", f"{API_BASE}/api/asset/NVDA/forecast?timeframe=1D"),
            ("BTC Trends", f"{API_BASE}/api/asset/BTC/trends?timeframe=7D"),
            ("Asset Search", f"{API_BASE}/api/assets/search?query=BTC"),
        ]
        
        results = []
        total_start = time.time()
        
        for name, url in tests:
            success, elapsed = await test_endpoint(session, name, url)
            results.append((name, success, elapsed))
            await asyncio.sleep(0.5)  # Small delay between tests
        
        total_elapsed = time.time() - total_start
        
        print()
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        
        successful = sum(1 for _, success, _ in results if success)
        failed = len(results) - successful
        avg_time = sum(elapsed for _, _, elapsed in results) / len(results)
        
        print(f"Total Tests: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Average Response Time: {avg_time:.2f}s")
        print(f"Total Test Duration: {total_elapsed:.2f}s")
        print()
        
        # Show slowest endpoints
        print("Slowest Endpoints:")
        sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
        for name, success, elapsed in sorted_results[:5]:
            status = "✅" if success else "❌"
            print(f"  {status} {name}: {elapsed:.2f}s")
        
        print()
        print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
