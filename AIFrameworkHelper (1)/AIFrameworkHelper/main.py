import eventlet
# Patch standard library for async compatibility BEFORE importing anything else
eventlet.monkey_patch()

from app import app  # noqa: F401

# Import and register blueprints
from api.routes import api_bp
from api.ml_routing_api import ml_routing_bp
from api.socketio_init import socketio

# Register blueprints
app.register_blueprint(api_bp)
app.register_blueprint(ml_routing_bp)

# Initialize SocketIO with app
socketio.init_app(app)

# Import WebSocket handlers after socketio is initialized
import api.websocket_handlers

# Register Swagger API documentation
from api.swagger_docs import api_bp as swagger_api_bp
app.register_blueprint(swagger_api_bp)
from api.websocket_handlers import *  # noqa: F401

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
