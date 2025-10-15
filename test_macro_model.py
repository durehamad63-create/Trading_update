#!/usr/bin/env python3
"""
Test Macro Model - Generate Predictions with Range and Confidence
"""
import pandas as pd
import numpy as np
import joblib
import os
from fredapi import Fred
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class MacroModelTester:
    def __init__(self):
        self.symbols = ['GDP', 'CPI', 'UNEMPLOYMENT', 'FED_RATE', 'CONSUMER_CONFIDENCE']
        self.fred_series = {
            'GDP': 'GDP',
            'CPI': 'CPIAUCSL',
            'UNEMPLOYMENT': 'UNRATE',
            'FED_RATE': 'FEDFUNDS',
            'CONSUMER_CONFIDENCE': 'UMCSENT'
        }
        
        fred_api_key = os.getenv('FRED_API_KEY')
        if not fred_api_key:
            raise ValueError("FRED_API_KEY not found")
        
        self.fred = Fred(api_key=fred_api_key)
        
        # Load trained models
        model_path = 'models/macro/macro_range_models.pkl'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.models = joblib.load(model_path)
        print(f"✅ Loaded models for {len(self.models)} indicators\n")
    
    def fetch_recent_data(self, symbol, periods=20):
        """Fetch recent data for prediction"""
        series_id = self.fred_series.get(symbol)
        if not series_id:
            return None
        
        try:
            data = self.fred.get_series(series_id)
            if data is None or len(data) == 0:
                return None
            
            df = pd.DataFrame({
                'timestamp': data.index,
                'close': data.values
            })
            df = df.dropna()
            
            # Get last N periods
            df = df.tail(periods).reset_index(drop=True)
            return df
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def calculate_features(self, df):
        """Calculate features for prediction"""
        df['change_1'] = df['close'].pct_change(1)
        df['change_4'] = df['close'].pct_change(4)
        df['ma_4'] = df['close'].rolling(4).mean()
        df['ma_12'] = df['close'].rolling(12).mean()
        df['trend'] = (df['close'] - df['ma_12']) / df['ma_12']
        df['volatility'] = df['change_1'].rolling(12).std()
        df['quarter'] = df['timestamp'].dt.quarter
        df['lag_1'] = df['close'].shift(1)
        df['lag_4'] = df['close'].shift(4)
        df['change_lag_1'] = df['change_1'].shift(1)
        
        return df
    
    def predict(self, symbol):
        """Generate prediction with range and confidence"""
        if symbol not in self.models:
            return None
        
        # Fetch recent data
        df = self.fetch_recent_data(symbol, periods=20)
        if df is None or len(df) < 15:
            return None
        
        # Calculate features
        df = self.calculate_features(df)
        df = df.dropna()
        
        if len(df) == 0:
            return None
        
        # Get latest row for prediction
        latest = df.iloc[-1]
        current_price = latest['close']
        
        # Get model
        model_data = self.models[symbol]['1D']
        features = model_data['features']
        
        # Prepare input
        X = latest[features].values.reshape(1, -1)
        
        # Scale features
        scaler = model_data['scaler']
        X_scaled = scaler.transform(X)
        
        # Predict
        price_change = model_data['price_model'].predict(X_scaled)[0]
        lower_change = model_data['lower_model'].predict(X_scaled)[0]
        upper_change = model_data['upper_model'].predict(X_scaled)[0]
        
        # Calculate predicted values
        predicted_price = current_price * (1 + price_change)
        lower_bound = current_price * (1 + lower_change)
        upper_bound = current_price * (1 + upper_change)
        
        # Direction
        direction = "UP" if price_change > 0 else "DOWN" if price_change < 0 else "FLAT"
        
        # Confidence based on range width and model R²
        range_width = abs(upper_change - lower_change)
        model_r2 = model_data['metrics']['price_r2']
        
        # Base confidence on model performance
        if model_r2 > 0.2:
            base_confidence = 75
        elif model_r2 > 0.1:
            base_confidence = 70
        elif model_r2 > 0:
            base_confidence = 65
        else:
            base_confidence = 60
        
        # Adjust by range width (wider range = lower confidence)
        confidence = int(base_confidence - (range_width * 100))
        confidence = max(50, min(90, confidence))
        
        return {
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'predicted_price': round(predicted_price, 2),
            'lower_bound': round(lower_bound, 2),
            'upper_bound': round(upper_bound, 2),
            'predicted_range': f"{round(lower_bound, 2)}–{round(upper_bound, 2)}",
            'direction': direction,
            'confidence': confidence,
            'price_change_pct': round(price_change * 100, 2),
            'model_r2': round(model_r2, 4),
            'timestamp': latest['timestamp'].strftime('%Y-%m-%d')
        }
    
    def test_all(self):
        """Test all macro indicators"""
        print("🏛️  MACRO INDICATORS PREDICTIONS")
        print("=" * 80)
        
        for symbol in self.symbols:
            print(f"\n📊 {symbol}")
            print("-" * 80)
            
            result = self.predict(symbol)
            
            if result is None:
                print("  ❌ Prediction failed")
                continue
            
            print(f"  Current Value:     {result['current_price']}")
            print(f"  Predicted Value:   {result['predicted_price']} ({result['price_change_pct']:+.2f}%)")
            print(f"  Predicted Range:   {result['predicted_range']}")
            print(f"  Direction:         {result['direction']}")
            print(f"  Confidence:        {result['confidence']}%")
            print(f"  Model R²:          {result['model_r2']}")
            print(f"  Last Update:       {result['timestamp']}")

def main():
    try:
        tester = MacroModelTester()
        tester.test_all()
        
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("💡 Please set FRED_API_KEY in your .env file")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Run train_macro_model.py first to train the models")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
