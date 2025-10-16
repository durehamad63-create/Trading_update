"""Interval Formatter for Accuracy History"""
from datetime import datetime

class IntervalFormatter:
    @staticmethod
    def format_timestamp(timestamp_str: str, timeframe: str) -> str:
        """Format timestamp based on timeframe interval requirements"""
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        
        if timeframe == '1H':
            # 30-minute intervals: 09:30, 10:00
            return dt.strftime('%H:%M')
        elif timeframe == '4H':
            # 1-hour intervals: 09:00, 10:00, 11:00
            return dt.strftime('%H:00')
        elif timeframe == '1D':
            # Daily intervals: Aug 28, Aug 29
            return dt.strftime('%b %d')
        elif timeframe in ['7D', '1W']:
            # Daily intervals: Oct 01, Oct 02
            return dt.strftime('%b %d')
        elif timeframe == '1M':
            # Monthly intervals: Sep 2021, Oct 2021
            return dt.strftime('%b %Y')
        else:
            return dt.strftime('%Y-%m-%d %H:%M')
    
    @staticmethod
    def get_interval_description(timeframe: str) -> str:
        """Get human-readable interval description"""
        intervals = {
            '1H': '30-minute intervals',
            '4H': '1-hour intervals',
            '1D': 'Daily intervals',
            '7D': 'Daily intervals',
            '1W': 'Daily intervals',
            '1M': 'Monthly intervals'
        }
        return intervals.get(timeframe, 'Custom intervals')

interval_formatter = IntervalFormatter()
