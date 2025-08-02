""
Flask application for serving Brent oil price analysis results.
"""

import os
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Enable CORS
    CORS(app)
    
    # Configuration
    app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False') == 'True'
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev')
    
    # Register blueprints
    from .routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    @app.route('/')
    def index():
        """Root endpoint with API information."""
        return jsonify({
            'name': 'Brent Oil Price Analysis API',
            'version': '1.0.0',
            'endpoints': {
                'data': '/api/data',
                'change_points': '/api/change-points',
                'events': '/api/events',
                'analysis': '/api/analysis',
            }
        })
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000)
