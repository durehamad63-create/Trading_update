"""Routes Module"""
from .market_routes import setup_market_routes
from .trends_routes import setup_trends_routes
from .forecast_routes import setup_forecast_routes
from .websocket_routes import setup_websocket_routes
from .utility_routes import setup_utility_routes
from .admin_routes import setup_admin_routes

def setup_all_routes(app, model, database):
    setup_market_routes(app, model, database)
    setup_trends_routes(app, model, database)
    setup_forecast_routes(app, model, database)
    setup_websocket_routes(app, model, database)
    setup_utility_routes(app, model, database)
    setup_admin_routes(app, model, database)
