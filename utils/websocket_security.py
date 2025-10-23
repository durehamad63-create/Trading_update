"""WebSocket Security Utilities"""
import html
import re
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class WebSocketSecurity:
    """Security utilities for WebSocket connections"""
    
    # Allowed symbols pattern (alphanumeric, dash, underscore only)
    SYMBOL_PATTERN = re.compile(r'^[A-Z0-9_-]+$')
    
    @staticmethod
    def sanitize_symbol(symbol: str) -> str:
        """Sanitize symbol to prevent XSS"""
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Invalid symbol")
        
        # Convert to uppercase and strip whitespace
        symbol = symbol.upper().strip()
        
        # Validate pattern
        if not WebSocketSecurity.SYMBOL_PATTERN.match(symbol):
            raise ValueError(f"Invalid symbol format: {symbol}")
        
        # Additional length check
        if len(symbol) > 20:
            raise ValueError("Symbol too long")
        
        return symbol
    
    @staticmethod
    def sanitize_string(value: str) -> str:
        """Sanitize string to prevent XSS"""
        if not isinstance(value, str):
            return str(value)
        return html.escape(value)
    
    @staticmethod
    def get_utc_now() -> datetime:
        """Get timezone-aware UTC datetime"""
        return datetime.now(timezone.utc)
    
    @staticmethod
    def validate_timeframe(timeframe: str) -> str:
        """Validate and normalize timeframe (case-insensitive)"""
        # Normalize to uppercase for consistency
        timeframe_upper = timeframe.upper()
        
        # Map of valid timeframes (all uppercase)
        valid_timeframes = ['1M', '5M', '15M', '30M', '1H', '4H', '1D', '7D', '1W', '1WK', '1MO', '1Y', '5Y']
        
        if timeframe_upper not in valid_timeframes:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        # Return normalized uppercase version
        return timeframe_upper
    
    @staticmethod
    def safe_float(value, default=0.0) -> float:
        """Safely convert to float"""
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Failed to convert {value} to float, using default {default}")
            return default
    
    @staticmethod
    def safe_int(value, default=0) -> int:
        """Safely convert to int"""
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"Failed to convert {value} to int, using default {default}")
            return default
