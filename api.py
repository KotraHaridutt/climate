# api.py
from fastapi import FastAPI
import torch
import numpy as np
from model import ExtremeWeatherModel # Your existing model class

app = FastAPI(title="Extreme Weather Prediction API")

# Initialize model globally so it only loads once when the server starts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ExtremeWeatherModel(input_channels=5, hidden_dim=16).to(device)

try:
    model.load_state_dict(torch.load("convlstm_extreme_weather.pt", map_location=device))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model weights. {e}")

@app.get("/predict_next_hour")
def predict_next_hour():
    """
    Simulates fetching the last 24 hours of data from 6:00 AM yesterday to 6:00 AM today,
    runs inference, and returns the 21x21 grid of probabilities for 7:00 AM.
    """
    # 1. Simulate pulling 24 hours of live API data (B, T, C, H, W)
    # In production, replace this with actual Open-Meteo API fetching logic
    live_tensor = torch.rand(1, 24, 5, 21, 21).to(device) 
    
    # 2. Run Inference
    with torch.no_grad():
        raw_logits = model(live_tensor)
        probabilities = torch.sigmoid(raw_logits)
    
    # Extract the 21x21 probability matrix and convert to a standard Python list
    prob_grid = probabilities[0, 0, :, :].cpu().numpy().tolist()
    
    # 3. Simulate the "Proof" data (e.g., average regional moisture trend over 24h)
    # We send this back so the frontend can plot a chart showing WHY the model triggered
    simulated_moisture_trend = np.linspace(0.2, 0.9, 24).tolist() 
    
    return {
        "status": "success",
        "target_time": "Next Hour (+1)",
        "probability_grid": prob_grid, # 21x21 array
        "proof_metrics": {
            "moisture_trend_24h": simulated_moisture_trend,
            "dominant_wind_direction": "North-East"
        }
    }

# Run this using: uvicorn api:app --reload