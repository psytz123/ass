"""
Main entry point for the AI Orchestration Framework Flask application
"""

from app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

