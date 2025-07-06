"""
Initialize SocketIO separately to avoid circular imports
"""
from flask_socketio import SocketIO

# Create SocketIO instance
socketio = SocketIO(cors_allowed_origins="*", async_mode='eventlet')