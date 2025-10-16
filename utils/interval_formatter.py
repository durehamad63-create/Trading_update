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
            # 4-hour intervals: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
            hour = (dt.hour // 4) * 4
            return f"{hour:02d}:00"
        elif timeframe in ['7D', '1W']:
            # Daily intervals: Oct 01, Oct 02
            return dt.strftime('%b %d')
        elif timeframe == '1M':
            # Weekly intervals: Week 1, Week 2
            week_num = (dt.day - 1) // 7 + 1
            return f"Week {week_num}"
        else:
            return dt.strftime('%Y-%m-%d %H:%M')
    
    @staticmethod
    def get_interval_description(timeframe: str) -> str:
        """Get human-readable interval description"""
        intervals = {
            '1H': '30-minute intervals',
            '4H': '1-hour intervals',
            '1D': '4-hour intervals',
            '7D': 'Daily intervals',
            '1W': 'Daily intervals',
            '1M': 'Weekly intervals'
        }
        return intervals.get(timeframe, 'Custom intervals')

interval_formatter = IntervalFormatter()
