""
API routes for the Brent Oil Price Analysis application.
"""

from flask import Blueprint, jsonify, request
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

# Create blueprint
api_bp = Blueprint('api', __name__)

# Sample data - in a real application, this would be loaded from a database or file
SAMPLE_DATA = {
    'prices': [],
    'change_points': [],
    'events': []
}

def load_data():
    """Load data from files if they exist."""
    try:
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
        
        # Load price data
        price_path = os.path.join(data_dir, 'processed_prices.csv')
        if os.path.exists(price_path):
            df = pd.read_csv(price_path, parse_dates=['Date'])
            SAMPLE_DATA['prices'] = df.to_dict('records')
        
        # Load change points
        cp_path = os.path.join(data_dir, 'change_points.json')
        if os.path.exists(cp_path):
            with open(cp_path, 'r') as f:
                SAMPLE_DATA['change_points'] = json.load(f)
        
        # Load events
        events_path = os.path.join(data_dir, 'events', 'geopolitical_events.csv')
        if os.path.exists(events_path):
            events_df = pd.read_csv(events_path, parse_dates=['event_date'])
            SAMPLE_DATA['events'] = events_df.to_dict('records')
            
    except Exception as e:
        print(f"Error loading data: {e}")

# Load data when the module is imported
load_data()

@api_bp.route('/data', methods=['GET'])
def get_price_data():
    """
    Get historical price data with optional date range filtering.
    
    Query Parameters:
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
    """
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Convert sample data to DataFrame for filtering
        df = pd.DataFrame(SAMPLE_DATA['prices'])
        
        # Filter by date range if provided
        if start_date:
            df = df[df['Date'] >= start_date]
        if end_date:
            df = df[df['Date'] <= end_date]
            
        return jsonify({
            'status': 'success',
            'data': df.to_dict('records'),
            'count': len(df)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api_bp.route('/change-points', methods=['GET'])
def get_change_points():
    """
    Get detected change points with their statistics.
    """
    try:
        return jsonify({
            'status': 'success',
            'data': SAMPLE_DATA['change_points']
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api_bp.route('/events', methods=['GET'])
def get_events():
    """
    Get geopolitical and economic events with optional filtering.
    
    Query Parameters:
        start_date: Filter events after this date (YYYY-MM-DD)
        end_date: Filter events before this date (YYYY-MM-DD)
        event_type: Filter by event type (e.g., 'political', 'economic')
    """
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        event_type = request.args.get('event_type')
        
        events = SAMPLE_DATA['events'].copy()
        
        # Apply filters
        if start_date:
            events = [e for e in events if e['event_date'] >= start_date]
        if end_date:
            events = [e for e in events if e['event_date'] <= end_date]
        if event_type:
            events = [e for e in events if e['event_type'] == event_type]
            
        return jsonify({
            'status': 'success',
            'data': events,
            'count': len(events)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api_bp.route('/analysis', methods=['GET'])
def get_analysis():
    """
    Get analysis results including change points with nearby events.
    """
    try:
        # In a real application, this would perform more sophisticated analysis
        # For now, we'll just return the combined data
        
        # Convert to DataFrames for easier manipulation
        change_points = pd.DataFrame(SAMPLE_DATA['change_points'])
        events = pd.DataFrame(SAMPLE_DATA['events'])
        
        # Convert date strings to datetime for comparison
        if not events.empty and 'event_date' in events.columns:
            events['event_date'] = pd.to_datetime(events['event_date'])
        
        if not change_points.empty and 'date' in change_points.columns:
            change_points['date'] = pd.to_datetime(change_points['date'])
            
            # Find events near each change point (within 30 days)
            analysis_results = []
            for _, cp in change_points.iterrows():
                cp_date = pd.to_datetime(cp['date'])
                
                # Find nearby events
                if not events.empty:
                    events['days_diff'] = (events['event_date'] - cp_date).dt.days.abs()
                    nearby_events = events[events['days_diff'] <= 30].copy()
                    
                    # Convert events to dict and remove the temporary column
                    nearby_events = nearby_events.drop(columns=['days_diff']).to_dict('records')
                else:
                    nearby_events = []
                
                # Add to results
                result = cp.to_dict()
                result['nearby_events'] = nearby_events
                analysis_results.append(result)
        else:
            analysis_results = []
        
        return jsonify({
            'status': 'success',
            'data': analysis_results
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
