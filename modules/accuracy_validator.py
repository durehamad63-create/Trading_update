"""
Enhanced Accuracy Validation System
"""
import asyncio
import logging
from datetime import datetime, timedelta
from database import db
from multi_asset_support import multi_asset
from utils.data_validator import data_validator

class AccuracyValidator:
    def __init__(self):
        self.validation_threshold = 0.05  # 5% price difference threshold
    
    async def validate_forecasts(self, symbol, days=7):
        """Validate recent forecasts using real database data only - NO SYNTHETIC DATA"""
        if not db or not db.pool:
            logging.warning(f"Database not available for validation of {symbol}")
            return {'error': 'Database unavailable', 'message': 'Cannot validate without database connection'}
        
        try:
            # Get database accuracy
            accuracy = await db.calculate_accuracy(symbol, days)
            
            # Get forecast count
            async with db.pool.acquire() as conn:
                count = await conn.fetchval("""
                    SELECT COUNT(*) FROM forecasts 
                    WHERE symbol LIKE $1 AND created_at >= NOW() - INTERVAL '%s days'
                """ % days, f"{symbol}%")
            
            # Return real data only - NO SYNTHETIC FALLBACK
            if accuracy == 0 and count == 0:
                return {
                    'error': 'No data available',
                    'message': f'No historical forecast data available for {symbol}. Please wait for data collection.'
                }
            
            return {'accuracy': round(accuracy, 2), 'validated': count}
            
        except Exception as e:
            logging.error(f"Forecast validation failed for {symbol}: {e}")
            return {'error': str(e), 'message': 'Validation failed due to database error'}
    
    def _check_direction_accuracy(self, predicted_direction, actual_change):
        """Check if direction prediction was correct"""
        if predicted_direction == 'UP' and actual_change > 0:
            return True
        elif predicted_direction == 'DOWN' and actual_change < 0:
            return True
        elif predicted_direction == 'HOLD' and abs(actual_change) < 1:
            return True
        return False
    
    async def _store_validation_result(self, forecast_id, forecast_direction, actual_direction, result, accuracy_score):
        """Store validation result in database"""
        if not db.pool:
            return
        
        async with db.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO forecast_accuracy (forecast_id, actual_direction, result, accuracy_score)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (forecast_id) DO UPDATE SET
                    actual_direction = $2,
                    result = $3,
                    accuracy_score = $4,
                    evaluated_at = NOW()
            """, forecast_id, actual_direction, result, accuracy_score)

accuracy_validator = AccuracyValidator()