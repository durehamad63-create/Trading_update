"""
Data Validation Module for BTC Trends
Ensures actual prices are realistic and properly validated
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class DataValidator:
    """Validates and sanitizes time series data for trends"""
    
    # Realistic price ranges for validation (updated for 2024)
    PRICE_RANGES = {
        'BTC': {'min': 15000, 'max': 150000},
        'ETH': {'min': 1000, 'max': 10000},
        'BNB': {'min': 200, 'max': 800},
        'SOL': {'min': 10, 'max': 300},
        'XRP': {'min': 0.3, 'max': 3.0},
        'DOGE': {'min': 0.05, 'max': 0.5},
        'ADA': {'min': 0.2, 'max': 3.0},
        'TRX': {'min': 0.05, 'max': 0.3},
        'USDT': {'min': 0.99, 'max': 1.01},
        'USDC': {'min': 0.99, 'max': 1.01},
        # Stocks
        'NVDA': {'min': 100, 'max': 1500},
        'MSFT': {'min': 200, 'max': 600},
        'AAPL': {'min': 100, 'max': 300},
        'GOOGL': {'min': 80, 'max': 200},
        'AMZN': {'min': 80, 'max': 250},
        'META': {'min': 100, 'max': 600},
        'TSLA': {'min': 100, 'max': 500},
    }
    
    # Maximum allowed price change between consecutive points
    MAX_CHANGE_PCT = {
        '1h': 5.0,   # 5% max change per hour
        '4H': 10.0,  # 10% max change per 4 hours
        '1D': 15.0,  # 15% max change per day
        '7D': 30.0,  # 30% max change per week
        '1W': 30.0,
        '1M': 50.0,  # 50% max change per month
    }
    
    @classmethod
    def validate_price(cls, symbol: str, price: float) -> bool:
        """Validate if price is within realistic range"""
        if price <= 0:
            return False
        
        ranges = cls.PRICE_RANGES.get(symbol)
        if not ranges:
            # For unknown symbols, just check if positive
            return price > 0
        
        return ranges['min'] <= price <= ranges['max']
    
    @classmethod
    def validate_price_series(cls, symbol: str, prices: List[float], timeframe: str = '1D') -> List[bool]:
        """Validate entire price series for anomalies"""
        if not prices:
            return []
        
        valid_flags = []
        max_change = cls.MAX_CHANGE_PCT.get(timeframe, 15.0)
        
        for i, price in enumerate(prices):
            # Check absolute range
            if not cls.validate_price(symbol, price):
                valid_flags.append(False)
                continue
            
            # Check relative change from previous point
            if i > 0:
                prev_price = prices[i-1]
                if prev_price > 0:
                    change_pct = abs((price - prev_price) / prev_price * 100)
                    if change_pct > max_change:
                        valid_flags.append(False)
                        continue
            
            valid_flags.append(True)
        
        return valid_flags
    
    @classmethod
    def clean_price_series(cls, symbol: str, prices: List[float], timestamps: List[datetime], 
                          timeframe: str = '1D') -> tuple[List[float], List[datetime]]:
        """Clean price series by removing outliers and interpolating"""
        if not prices or len(prices) != len(timestamps):
            return prices, timestamps
        
        valid_flags = cls.validate_price_series(symbol, prices, timeframe)
        
        cleaned_prices = []
        cleaned_timestamps = []
        last_valid_price = None
        
        for i, (price, timestamp, is_valid) in enumerate(zip(prices, timestamps, valid_flags)):
            if is_valid:
                cleaned_prices.append(price)
                cleaned_timestamps.append(timestamp)
                last_valid_price = price
            else:
                # Interpolate invalid prices
                if last_valid_price is not None:
                    # Find next valid price
                    next_valid_price = None
                    for j in range(i+1, len(prices)):
                        if valid_flags[j]:
                            next_valid_price = prices[j]
                            break
                    
                    if next_valid_price:
                        # Linear interpolation
                        interpolated = (last_valid_price + next_valid_price) / 2
                    else:
                        # Use last valid price
                        interpolated = last_valid_price
                    
                    cleaned_prices.append(interpolated)
                    cleaned_timestamps.append(timestamp)
                    logging.warning(f"Interpolated invalid price for {symbol} at {timestamp}: {price} -> {interpolated}")
        
        return cleaned_prices, cleaned_timestamps
    
    @classmethod
    def validate_accuracy_data(cls, actual: List[float], predicted: List[float], 
                               symbol: str, timeframe: str = '1D') -> Dict:
        """Validate accuracy data and compute error metrics"""
        if not actual or not predicted or len(actual) != len(predicted):
            return {
                'valid': False,
                'error': 'Mismatched or empty data',
                'actual_count': len(actual) if actual else 0,
                'predicted_count': len(predicted) if predicted else 0
            }
        
        # Validate all actual prices
        valid_actual = cls.validate_price_series(symbol, actual, timeframe)
        valid_predicted = cls.validate_price_series(symbol, predicted, timeframe)
        
        invalid_actual_count = sum(1 for v in valid_actual if not v)
        invalid_predicted_count = sum(1 for v in valid_predicted if not v)
        
        # Calculate error metrics for valid pairs
        errors = []
        for i, (a, p, va, vp) in enumerate(zip(actual, predicted, valid_actual, valid_predicted)):
            if va and vp and a > 0:
                error_pct = abs((p - a) / a * 100)
                errors.append(error_pct)
        
        if not errors:
            return {
                'valid': False,
                'error': 'No valid price pairs found',
                'invalid_actual': invalid_actual_count,
                'invalid_predicted': invalid_predicted_count
            }
        
        return {
            'valid': True,
            'mean_error_pct': sum(errors) / len(errors),
            'max_error_pct': max(errors),
            'min_error_pct': min(errors),
            'valid_pairs': len(errors),
            'invalid_actual': invalid_actual_count,
            'invalid_predicted': invalid_predicted_count,
            'total_pairs': len(actual)
        }
    
    @classmethod
    def compute_hit_miss(cls, actual: List[float], predicted: List[float], 
                        threshold_pct: float = 5.0) -> List[str]:
        """Compute Hit/Miss results based on error threshold"""
        if not actual or not predicted or len(actual) != len(predicted):
            return []
        
        results = []
        for a, p in zip(actual, predicted):
            if a > 0:
                error_pct = abs((p - a) / a * 100)
                result = 'Hit' if error_pct <= threshold_pct else 'Miss'
                results.append(result)
            else:
                results.append('Invalid')
        
        return results
    
    @classmethod
    def get_realistic_price_range(cls, symbol: str) -> Optional[Dict]:
        """Get realistic price range for symbol"""
        return cls.PRICE_RANGES.get(symbol)

# Global validator instance
data_validator = DataValidator()
