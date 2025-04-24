import torch
import pandas as pd
import numpy as np
import joblib
import requests
import logging
import os
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add LayerNormalization to torch.nn to handle the import in Model.py
# This creates a compatibility layer without modifying Model.py
import torch.nn as nn
if not hasattr(nn, 'LayerNormalization'):
    nn.LayerNormalization = nn.LayerNorm
    logger.info("Added compatibility for nn.LayerNormalization")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model and scalers
model = None
feature_scaler = None
target_scaler = None
stop_id_mapping = None

# Define LSTMModel class here to match the saved model architecture
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Multiply by 2 for bidirectional
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # Take the last hidden state
        return self.fc(out)

def load_model_and_scalers():
    """Load the trained LSTM model and scalers"""
    global model, feature_scaler, target_scaler, stop_id_mapping
    
    try:
        # Load the model
        model_path = "best_model.pth"  # or model.pth based on your preference
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Initialize model with correct architecture to match saved weights
        input_dim = 6  # Number of features
        hidden_dim = 128
        output_dim = 1
        num_layers = 3  # Based on your error message showing lstm layers up to l2
        
        # Use LSTMModel defined in this file (not importing from Model.py)
        model = LSTMModel(
            input_dim=input_dim, 
            hidden_dim=hidden_dim,
            output_dim=output_dim, 
            num_layers=num_layers
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Load scalers
        feature_scaler = joblib.load("feature_scaler.pkl")
        target_scaler = joblib.load("target_scaler.pkl")
        
        # Load stop_id mapping
        try:
            with open("stop_id_mapping.json", "r") as f:
                stop_id_mapping = json.load(f)
            logger.info(f"Loaded stop ID mapping with {len(stop_id_mapping)} entries")
        except Exception as mapping_error:
            logger.error(f"Error loading stop ID mapping: {str(mapping_error)}")
            stop_id_mapping = {}
        
        logger.info("Model and scalers loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model or scalers: {str(e)}")
        return False

def map_stop_id(stop_id):
    """Map string stop_id to numeric value using mapping file"""
    try:
        # If we have a mapping and the stop_id is in the mapping
        if stop_id_mapping and stop_id in stop_id_mapping:
            numeric_id = stop_id_mapping[stop_id]
            logger.info(f"Mapped stop_id '{stop_id}' to numeric value {numeric_id}")
            return numeric_id
        
        # If stop_id is already numeric, return it
        if isinstance(stop_id, (int, float)) or (isinstance(stop_id, str) and stop_id.isdigit()):
            return int(stop_id)
        
        # If we couldn't map it, return a default value
        logger.warning(f"Could not map stop_id '{stop_id}', using default value 0")
        return 0
    except Exception as e:
        logger.error(f"Error mapping stop_id: {str(e)}")
        return 0

def fetch_gtfs_data():
    """Fetch real-time GTFS data from the API"""
    try:
        response = requests.get("http://gtfs.ltconline.ca/TripUpdate/TripUpdates.json", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching GTFS data: {str(e)}")
        return None

def get_scheduled_time(gtfs_data, route_id, stop_id):
    """Extract the scheduled arrival time for the specified route and stop"""
    if not gtfs_data or 'entity' not in gtfs_data:
        logger.warning("GTFS data missing or has no 'entity' field")
        scheduled_time = int(time.time()) + 900
        logger.info(f"Using mock scheduled time for testing: {scheduled_time}")
        return scheduled_time
    
    try:
        # Log what we're looking for
        logger.info(f"Searching for route_id={route_id}, stop_id={stop_id}")
        logger.info(f"Found {len(gtfs_data['entity'])} entities in GTFS data")
        
        # If stop_id is numeric, also try to find its string representation
        numeric_stop_id = stop_id
        string_stop_id = None
        
        if isinstance(stop_id, (int, float)) or (isinstance(stop_id, str) and stop_id.isdigit()):
            # If we have a numeric stop_id, find the corresponding string ID
            numeric_stop_id = int(stop_id) if isinstance(stop_id, str) else stop_id
            # Reverse lookup in stop_id_mapping
            for key, value in stop_id_mapping.items():
                if value == numeric_stop_id:
                    string_stop_id = key
                    logger.info(f"Found string representation of stop_id {numeric_stop_id}: {string_stop_id}")
                    break
        else:
            # If we have a string stop_id, use it directly
            string_stop_id = stop_id
            # Also get the numeric representation if available
            if stop_id in stop_id_mapping:
                numeric_stop_id = stop_id_mapping[stop_id]
                logger.info(f"Found numeric representation of stop_id {string_stop_id}: {numeric_stop_id}")
        
        # Collect routes and stops for debugging
        found_routes = set()
        found_stops = set()
        
        for i, entity in enumerate(gtfs_data['entity']):
            trip_update = entity.get('trip_update', {})
            trip_info = trip_update.get('trip', {})
            
            # Collect route IDs for debugging
            current_route = trip_info.get('route_id')
            if current_route:
                found_routes.add(current_route)
            
            # Check if this is the requested route
            if current_route == route_id:
                logger.info(f"Found matching route: {route_id} in entity {i}")
                stop_time_updates = trip_update.get('stop_time_update', [])
                logger.info(f"This route has {len(stop_time_updates)} stop updates")
                
                # Collect all stops for this route
                for stop_update in stop_time_updates:
                    current_stop = stop_update.get('stop_id')
                    if current_stop:
                        found_stops.add(current_stop)
                        
                    # Check for both string and numeric representation of stop_id
                    if (current_stop == string_stop_id) or (current_stop == str(numeric_stop_id)):
                        logger.info(f"Found matching stop: {current_stop}")
                        
                        # Check for arrival time first, then departure time
                        arrival = stop_update.get('arrival', {})
                        if arrival and arrival.get('time'):
                            logger.info(f"Found arrival time: {arrival.get('time')}")
                            return arrival.get('time')
                            
                        # If no arrival time, try departure time
                        departure = stop_update.get('departure', {})
                        if departure and departure.get('time'):
                            logger.info(f"Found departure time: {departure.get('time')}")
                            return departure.get('time')
        
        # If we get here, we didn't find a match
        logger.warning(f"No scheduled time found for route {route_id} at stop {stop_id}")
        logger.warning(f"Available routes: {found_routes}")
        logger.warning(f"Stops for route {route_id}: {found_stops}")
        
        # For testing purposes, return a scheduled time 15 minutes from now
        scheduled_time = int(time.time()) + 900
        logger.info(f"Using mock scheduled time for testing: {scheduled_time}")
        return scheduled_time
    except Exception as e:
        logger.error(f"Error processing GTFS data: {str(e)}")
        # For testing purposes, return a scheduled time 15 minutes from now
        return int(time.time()) + 900

def preprocess_input(unix_time, stop_id, scheduled_time):
    """Preprocess the input data for the model"""
    try:
        # Convert Unix time to datetime
        dt = datetime.fromtimestamp(unix_time)
        
        # Extract relevant features
        day_of_week = dt.weekday()  # 0-6 (Monday-Sunday)
        day_of_year = dt.timetuple().tm_yday  # 1-366
        hour = dt.hour
        minute = dt.minute
        
        # Convert scheduled_time to hour/minute if it's in Unix format
        if scheduled_time:
            scheduled_dt = datetime.fromtimestamp(scheduled_time)
            scheduled_hour = scheduled_dt.hour
            scheduled_minute = scheduled_dt.minute
        else:
            # If no scheduled time is available, use current time as a fallback
            scheduled_hour = hour
            scheduled_minute = minute
        
        # Map string stop_id to numeric value
        numeric_stop_id = map_stop_id(stop_id)
        
        # Create features DataFrame
        features = pd.DataFrame({
            'stop_id': [numeric_stop_id],
            'day': [day_of_week],
            'day_of_year': [day_of_year],
            'scheduled_time': [scheduled_hour * 60 + scheduled_minute],  # Convert to minutes from midnight
            'hour': [hour],
            'minute': [minute]
        })
        
        # Normalize features
        normalized_features = feature_scaler.transform(features)
        
        # LSTMModel expects sequence input (batch_size, sequence_length, input_dim)
        sequence_length = 1  # For LSTMModel, we can use a single time step
        
        # Reshape for LSTM model - (batch_size, sequence_length, input_dim)
        model_input = torch.FloatTensor(normalized_features).view(1, sequence_length, -1)
        
        return model_input
    except Exception as e:
        logger.error(f"Error preprocessing input: {str(e)}")
        return None

def make_prediction(model_input):
    """Make prediction using the loaded model"""
    try:
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_input = model_input.to(device)
        
        with torch.no_grad():
            # Forward pass
            output = model(model_input)
            
            # Convert output tensor to numpy array
            output_np = output.cpu().numpy()
            
            # Inverse transform to get the actual delay prediction
            predicted_delay = target_scaler.inverse_transform(output_np.reshape(-1, 1))
            
            return predicted_delay[0][0]  # Return the scalar value
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": time.time()})

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Get JSON input
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Extract required fields
        current_time = data.get('current_time')
        route_id = data.get('route_id')
        stop_id = data.get('stop_id')
        
        # Validate input
        if not all([current_time, route_id, stop_id]):
            return jsonify({"error": "Missing required fields: current_time, route_id, stop_id"}), 400
        
        # Fetch GTFS data
        gtfs_data = fetch_gtfs_data()
        if not gtfs_data:
            logger.warning("Could not fetch GTFS data, using mock data for testing")
            gtfs_data = {"entity": []}
        
        # Get scheduled time
        scheduled_time = get_scheduled_time(gtfs_data, route_id, stop_id)
        if not scheduled_time:
            return jsonify({"error": f"No scheduled time found for route {route_id} at stop {stop_id}"}), 404
        
        # Preprocess input
        model_input = preprocess_input(current_time, stop_id, scheduled_time)
        if model_input is None:
            return jsonify({"error": "Failed to preprocess input data"}), 500
        
        # Make prediction
        predicted_delay = make_prediction(model_input)
        if predicted_delay is None:
            return jsonify({"error": "Failed to make prediction"}), 500
        
        # Calculate predicted arrival time
        predicted_arrival_time = scheduled_time + predicted_delay
        
        # Format response
        response = {
            "route_id": route_id,
            "stop_id": stop_id,
            "scheduled_time": scheduled_time,
            "predicted_delay": float(predicted_delay),
            "predicted_arrival_time": float(predicted_arrival_time),
            "current_time": current_time,
            "human_readable": {
                "scheduled": datetime.fromtimestamp(scheduled_time).strftime('%Y-%m-%d %H:%M:%S'),
                "predicted": datetime.fromtimestamp(predicted_arrival_time).strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # Log successful prediction
        logger.info(f"Prediction made for route {route_id}, stop {stop_id}: delay={predicted_delay:.2f}s")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing prediction request: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# Create a function to initialize our app
def initialize():
    """Initialize the model and scalers"""
    if not load_model_and_scalers():
        logger.critical("Failed to load model and scalers. API may not function correctly.")

# Call initialize() at startup
initialize()

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=False)