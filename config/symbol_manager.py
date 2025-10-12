"""
Centralized Symbol Manager
Handles all symbol formatting and database key generation consistently
"""
from .symbols import CRYPTO_SYMBOLS, STOCK_SYMBOLS, MACRO_SYMBOLS, SYMBOL_NAMES

class SymbolManager:
    """Centralized symbol management for consistent database keys and API calls"""
    
    @staticmethod
    def get_db_key(symbol: str, timeframe: str) -> str:
        """Generate consistent database key for symbol-timeframe combination"""
        return f"{symbol}_{timeframe}"
    
    @staticmethod
    def get_binance_symbol(symbol: str) -> str:
        """Get Binance API symbol format"""
        if symbol in CRYPTO_SYMBOLS:
            binance_symbol = CRYPTO_SYMBOLS[symbol].get('binance')
            if binance_symbol:
                return binance_symbol
            # Fallback for stablecoins
            if symbol in ['USDT', 'USDC']:
                return 'BTCUSDT'  # Use BTC as proxy
        return f"{symbol}USDT"
    
    @staticmethod
    def get_yahoo_symbol(symbol: str) -> str:
        """Get Yahoo Finance symbol format"""
        if symbol in CRYPTO_SYMBOLS:
            return CRYPTO_SYMBOLS[symbol].get('yahoo', f"{symbol}-USD")
        elif symbol in STOCK_SYMBOLS:
            return STOCK_SYMBOLS[symbol].get('yahoo', symbol)
        return symbol
    
    @staticmethod
    def get_symbol_name(symbol: str) -> str:
        """Get human-readable symbol name"""
        return SYMBOL_NAMES.get(symbol, symbol)
    
    @staticmethod
    def is_crypto(symbol: str) -> bool:
        """Check if symbol is cryptocurrency"""
        return symbol in CRYPTO_SYMBOLS
    
    @staticmethod
    def is_stock(symbol: str) -> bool:
        """Check if symbol is stock"""
        return symbol in STOCK_SYMBOLS
    
    @staticmethod
    def is_macro(symbol: str) -> bool:
        """Check if symbol is macro indicator"""
        return symbol in MACRO_SYMBOLS
    
    @staticmethod
    def get_all_symbols() -> list:
        """Get all supported symbols"""
        return list(CRYPTO_SYMBOLS.keys()) + list(STOCK_SYMBOLS.keys()) + list(MACRO_SYMBOLS.keys())
    
    @staticmethod
    def get_crypto_symbols() -> list:
        """Get all crypto symbols"""
        return list(CRYPTO_SYMBOLS.keys())
    
    @staticmethod
    def get_stock_symbols() -> list:
        """Get all stock symbols"""
        return list(STOCK_SYMBOLS.keys())
    
    @staticmethod
    def get_macro_symbols() -> list:
        """Get all macro symbols"""
        return list(MACRO_SYMBOLS.keys())

# Global instance
symbol_manager = SymbolManager()