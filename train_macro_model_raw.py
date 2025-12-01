#!/usr/bin/env python3
"""Raw macro model trainer - Compatible with ML predictor"""

import pandas as pd
import numpy as np
import joblib
import os
from fredapi import Fred
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

class RawMacroModelTrainer:
    def __init__(self):
        self.fred_api_key = '8dc164eeafb6133fa4b3fea4109187aa'
        self.fred = Fred(api_key=self.fred_api_key)
        
        self.symbols = {
            'GDP': 'GDP',
            'CPI': 'CPIAUCSL',
            'UNEMPLOYMENT': 'UNRATE',
            'FED_RATE': 'FEDFUNDS',
            'CONSUMER_CONFIDENCE': 'UMCSENT'
        }
    
    def fetch_macro_data(self, fred_series, start_date='2000-01-01'):
        """Fetch real macro data from FRED"""
        try:
            data = self.fred.get_series(fred_series, start=start_date)
            if data is None or len(data) < 20:
                return None
            
            df = pd.DataFrame({'timestamp': data.index, 'close': data.values})
            df = df.dropna()
            return df
        except Exception as e:
            print(f"    ❌ Error: {e}")
            return None
    
    def calculate_features(self, df):
        """Calculate features matching ML predictor expectations"""
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
        
        return df.dropna()
    
    def create_targets(self, df):
        """Create targets"""
        feature_cols = ['lag_1', 'lag_4', 'ma_4', 'ma_12', 'change_1', 'change_4', 
                       'change_lag_1', 'trend', 'volatility', 'quarter']
        
        df['next_return'] = df['change_1'].shift(-1)
        df['next_upper'] = df['close'].shift(-1) / df['close'] - 1
        df['next_lower'] = df['close'].shift(-1) / df['close'] - 1
        
        df = df.dropna()
        
        X = df[feature_cols]
        y_price = df['next_return']
        y_upper = df['next_upper']
        y_lower = df['next_lower']
        
        return X, y_price, y_upper, y_lower, feature_cols
    
    def train_model(self, X, y_price, y_upper, y_lower, symbol):
        """Train models matching ML predictor structure"""
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_price_train, y_price_test = y_price.iloc[:split_idx], y_price.iloc[split_idx:]
        y_upper_train, y_upper_test = y_upper.iloc[:split_idx], y_upper.iloc[split_idx:]
        y_lower_train, y_lower_test = y_lower.iloc[:split_idx], y_lower.iloc[split_idx:]
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        price_model = RandomForestRegressor(n_estimators=80, max_depth=8, random_state=42)
        upper_model = RandomForestRegressor(n_estimators=80, max_depth=8, random_state=42)
        lower_model = RandomForestRegressor(n_estimators=80, max_depth=8, random_state=42)
        
        price_model.fit(X_train_scaled, y_price_train)
        upper_model.fit(X_train_scaled, y_upper_train)
        lower_model.fit(X_train_scaled, y_lower_train)
        
        price_pred = price_model.predict(X_test_scaled)
        upper_pred = upper_model.predict(X_test_scaled)
        lower_pred = lower_model.predict(X_test_scaled)
        
        # Asset-specific confidence
        confidence_features = []
        confidence_scores = []
        
        for i in range(len(X_train)):
            train_pred = price_model.predict(X_train_scaled[i:i+1])[0]
            train_upper = upper_model.predict(X_train_scaled[i:i+1])[0]
            train_lower = lower_model.predict(X_train_scaled[i:i+1])[0]
            
            pred_error = abs(train_pred - y_price_train.iloc[i])
            upper_error = abs(train_upper - y_upper_train.iloc[i])
            lower_error = abs(train_lower - y_lower_train.iloc[i])
            
            conf_features = [
                pred_error,
                upper_error,
                lower_error,
                X_train.iloc[i]['volatility'],
                abs(X_train.iloc[i]['trend']),
                abs(X_train.iloc[i]['change_1']),
                abs(train_upper - train_lower),
                X_train.iloc[i]['ma_4'] / X_train.iloc[i]['lag_1'] if X_train.iloc[i]['lag_1'] != 0 else 1.0,
                X_train.iloc[i]['ma_12'] / X_train.iloc[i]['lag_1'] if X_train.iloc[i]['lag_1'] != 0 else 1.0
            ]
            confidence_features.append(conf_features)
            
            base_conf = max(60, 90 - (pred_error * 300))
            confidence_scores.append(np.clip(base_conf, 65, 95))
        
        confidence_model = RandomForestRegressor(n_estimators=40, max_depth=6, random_state=45)
        confidence_model.fit(np.array(confidence_features), confidence_scores)
        
        price_r2 = r2_score(y_price_test, price_pred)
        upper_r2 = r2_score(y_upper_test, upper_pred)
        lower_r2 = r2_score(y_lower_test, lower_pred)
        
        print(f"    Price R²: {price_r2:.4f}, Upper R²: {upper_r2:.4f}, Lower R²: {lower_r2:.4f}")
        
        return {
            'price_model': price_model,
            'high_model': upper_model,
            'low_model': lower_model,
            'confidence_model': confidence_model,
            'confidence_features': ['pred_error', 'upper_error', 'lower_error', 'volatility', 'trend', 'change', 'range_width', 'ma4_ratio', 'ma12_ratio'],
            'scaler': scaler,
            'features': X.columns.tolist(),
            'metrics': {'price_r2': price_r2, 'upper_r2': upper_r2, 'lower_r2': lower_r2}
        }
    
    def train_all_models(self):
        """Train macro models for 1D timeframe only"""
        print("🚀 Training Macro Models - Compatible with ML Predictor")
        print("=" * 60)
        
        all_models = {}
        
        for symbol, fred_series in self.symbols.items():
            print(f"\n📈 Training {symbol}...")
            
            df = self.fetch_macro_data(fred_series)
            if df is None or len(df) < 20:
                print(f"    ❌ Insufficient data")
                continue
            
            print(f"    📊 Raw data: {len(df)} records")
            
            df = self.calculate_features(df)
            print(f"    🔧 After features: {len(df)} records")
            
            X, y_price, y_upper, y_lower, features = self.create_targets(df)
            print(f"    🎯 Training data: {len(X)} records")
            
            if len(X) < 15:
                print(f"    ❌ Insufficient processed data")
                continue
            
            model_data = self.train_model(X, y_price, y_upper, y_lower, symbol)
            all_models[symbol] = {'1D': model_data}
            print(f"    ✅ Trained successfully")
        
        os.makedirs('models/macro_raw', exist_ok=True)
        model_path = 'models/macro_raw/macro_raw_models.pkl'
        joblib.dump(all_models, model_path)
        
        print(f"\n✅ Macro models saved to {model_path}")
        return all_models

def main():
    trainer = RawMacroModelTrainer()
    models = trainer.train_all_models()

if __name__ == "__main__":
    main()
