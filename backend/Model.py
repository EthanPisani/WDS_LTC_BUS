import torch
import pandas as pd
import numpy as np
import joblib
import requests
import logging
import os
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
    global model, feature_scaler, target_scaler
    
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
        
        # Use LSTMModel instead of ImprovedLSTM since that's what your weights are for
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
        
        logger.info("Model and scalers loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model or scalers: {str(e)}")
        return False

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
        return None
    
    try:
        for entity in gtfs_data['entity']:
            trip_update = entity.get('trip_update', {})
            trip_info = trip_update.get('trip', {})
            
            # Check if this is the requested route
            if trip_info.get('route_id') == route_id:
                stop_time_updates = trip_update.get('stop_time_update', [])
                
                for stop_time in stop_time_updates:
                    if stop_time.get('stop_id') == stop_id:
                        # Return the scheduled arrival time if available
                        arrival = stop_time.get('arrival', {})
                        return arrival.get('time')
        
        logger.warning(f"No scheduled time found for route {route_id} at stop {stop_id}")
        return None
    except Exception as e:
        logger.error(f"Error processing GTFS data: {str(e)}")
        return None

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
        
        # Create features DataFrame
        features = pd.DataFrame({
            'stop_id': [stop_id],
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
            return jsonify({"error": "Failed to fetch GTFS data"}), 503
        
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
    port = int(os.environ.get("PORT", 5002))
    app.run(host='0.0.0.0', port=port, debug=False)