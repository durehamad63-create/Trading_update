"""
Quick test to verify 4H timeframe returns 6 predictions
"""
import asyncio
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

async def test_4h_predictions():
    print("🧪 Testing 4H Timeframe Multi-Step Predictions\n")
    
    # Import after path setup
    from modules.ml_predictor import MobileMLModel
    from multistep_predictor import init_multistep_predictor
    
    # Initialize model
    print("📦 Loading ML model...")
    model = MobileMLModel()
    print("✅ Model loaded\n")
    
    # Initialize multistep predictor
    print("🔧 Initializing multistep predictor...")
    predictor = init_multistep_predictor(model)
    print("✅ Multistep predictor initialized\n")
    
    # Test 4H predictions for BTC
    symbol = "BTC"
    timeframe = "4H"
    num_steps = 6
    
    print(f"🔍 Testing {symbol} with timeframe={timeframe}, num_steps={num_steps}")
    print("-" * 60)
    
    result = await predictor.get_multistep_forecast(symbol, timeframe, num_steps)
    
    if result:
        prices = result.get('prices', [])
        timestamps = result.get('timestamps', [])
        
        print(f"✅ SUCCESS: Got {len(prices)} predictions")
        print(f"   Expected: {num_steps} predictions")
        print(f"   Match: {'✅ YES' if len(prices) == num_steps else '❌ NO'}\n")
        
        print("📊 Predictions:")
        for i, (price, ts) in enumerate(zip(prices, timestamps), 1):
            print(f"   Step {i}: ${price:,.2f} at {ts}")
        
        if len(prices) == num_steps:
            print("\n🎉 TEST PASSED: 4H timeframe returns 6 predictions!")
            return True
        else:
            print(f"\n❌ TEST FAILED: Expected {num_steps} predictions, got {len(prices)}")
            return False
    else:
        print("❌ TEST FAILED: No result returned")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_4h_predictions())
    sys.exit(0 if success else 1)
