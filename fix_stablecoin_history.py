#!/usr/bin/env python3
"""
Fix stablecoin historical prices in database
Updates all USDT and USDC prices to $1.00
"""
import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

async def fix_stablecoin_prices():
    """Update all stablecoin prices to $1.00"""
    
    # Connect to database
    database_url = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/trading_db')
    
    try:
        pool = await asyncpg.create_pool(database_url, min_size=1, max_size=5)
        
        async with pool.acquire() as conn:
            # Get all stablecoin symbols in database
            stablecoin_patterns = ['USDT%', 'USDC%']
            
            for pattern in stablecoin_patterns:
                # Update actual_prices table
                result = await conn.execute("""
                    UPDATE actual_prices 
                    SET price = 1.0,
                        open_price = 1.0,
                        high = 1.0,
                        low = 1.0,
                        close_price = 1.0,
                        change_24h = 0.0
                    WHERE symbol LIKE $1
                """, pattern)
                
                print(f"✅ Updated actual_prices for {pattern}: {result}")
                
                # Update forecasts table
                result = await conn.execute("""
                    UPDATE forecasts 
                    SET predicted_price = 1.0
                    WHERE symbol LIKE $1
                """, pattern)
                
                print(f"✅ Updated forecasts for {pattern}: {result}")
        
        await pool.close()
        print("✅ Stablecoin prices fixed successfully!")
        
    except Exception as e:
        print(f"❌ Error fixing stablecoin prices: {e}")

if __name__ == "__main__":
    asyncio.run(fix_stablecoin_prices())
