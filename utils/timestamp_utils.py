from datetime import timedelta

class TimestampUtils:
    @staticmethod
    def adjust_for_timeframe(timestamp, timeframe):
        """Normalize timestamp to timeframe boundary"""
        timeframe = timeframe.upper()
        
        if timeframe == '1W':
            days_since_monday = timestamp.weekday()
            week_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            return week_start - timedelta(days=days_since_monday)
        elif timeframe == '1D':
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        elif timeframe == '4H':
            hour_boundary = (timestamp.hour // 4) * 4
            return timestamp.replace(hour=hour_boundary, minute=0, second=0, microsecond=0)
        elif timeframe == '1H':
            return timestamp.replace(minute=0, second=0, microsecond=0)
        else:
            return timestamp.replace(second=0, microsecond=0)