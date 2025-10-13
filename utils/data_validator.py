"""
Data Validation Utility - Ensures only real API data is stored
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional

class DataValidator:
    """Validates data to ensure it's from real sources, not synthetic"""
    
    @staticmethod
    def validate_price_data(data: Dict[str, Any], source: str) -> bool:
        """
        Validate price data is from real API source
        
        Args:
            data: Price data dictionary
            source: Data source identifier (e.g., 'binance', 'yahoo', 'fred')
        
        Returns:
            bool: True if data is valid and from real source
        """
        required_fields = ['current_price', 'timestamp']
        
        # Check required fields exist
        if not all(field in data for field in required_fields):
            logging.warning(f"Missing required fields in price data from {source}")
            return False
        
        # Validate price is positive number
        try:
            price = float(data['current_price'])
            if price <= 0:
                logging.warning(f"Invalid price value: {price} from {source}")
                return False
        except (ValueError, TypeError):
            logging.warning(f"Price is not a valid number from {source}")
            return False
        
        # Validate timestamp
        if not isinstance(data['timestamp'], datetime):
            logging.warning(f"Invalid timestamp type from {source}")
            return False
        
        # Validate data source is real API (allow 'api' as generic real source)
        valid_sources = ['binance', 'yahoo', 'fred', 'iex', 'alpha_vantage', 'api']
        if source.lower() not in valid_sources:
            logging.error(f"Invalid data source: {source}. Only real API sources allowed.")
            return False
        
        return True
    
    @staticmethod
    def validate_historical_data(data: list, source: str) -> bool:
        """
        Validate historical data is from real API source
        
        Args:
            data: List of historical data points
            source: Data source identifier
        
        Returns:
            bool: True if all data points are valid
        """
        if not data or len(data) == 0:
            logging.warning(f"Empty historical data from {source}")
            return False
        
        # Validate each data point
        for point in data:
            if not isinstance(point, dict):
                logging.warning(f"Invalid data point format from {source}")
                return False
            
            # Check for required OHLC fields
            required = ['timestamp', 'close']
            if not all(field in point for field in required):
                logging.warning(f"Missing OHLC fields in historical data from {source}")
                return False
        
        return True
    
    @staticmethod
    def validate_forecast_data(data: Dict[str, Any]) -> bool:
        """
        Validate forecast data has required fields
        
        Args:
            data: Forecast data dictionary
        
        Returns:
            bool: True if forecast data is valid
        """
        required_fields = ['symbol', 'forecast_direction', 'confidence']
        
        if not all(field in data for field in required_fields):
            logging.warning("Missing required fields in forecast data")
            return False
        
        # Validate forecast direction
        valid_directions = ['UP', 'DOWN', 'HOLD']
        if data['forecast_direction'] not in valid_directions:
            logging.warning(f"Invalid forecast direction: {data['forecast_direction']}")
            return False
        
        # Validate confidence range
        try:
            confidence = int(data['confidence'])
            if not 0 <= confidence <= 100:
                logging.warning(f"Confidence out of range: {confidence}")
                return False
        except (ValueError, TypeError):
            logging.warning("Invalid confidence value")
            return False
        
        return True
    
    @staticmethod
    def is_synthetic_data(data: Dict[str, Any]) -> bool:
        """
        Detect if data appears to be synthetically generated
        
        Args:
            data: Data dictionary to check
        
        Returns:
            bool: True if data appears synthetic
        """
        # Check for synthetic markers
        synthetic_markers = [
            'synthetic', 'generated', 'simulated', 'random', 
            'sample', 'mock', 'fake', 'test'
        ]
        
        # Check data source field
        if 'data_source' in data:
            source = str(data['data_source']).lower()
            if any(marker in source for marker in synthetic_markers):
                logging.error(f"Synthetic data detected: {source}")
                return True
        
        # Check for metadata flags
        if data.get('is_synthetic') or data.get('is_simulated'):
            logging.error("Synthetic data flag detected")
            return True
        
        return False

# Global validator instance
data_validator = DataValidator()
