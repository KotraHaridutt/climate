
from fastapi import FastAPI
import torch
import numpy as np
import requests
import math
import xarray as xr
from model import ExtremeWeatherModel 

app = FastAPI(title="Extreme Weather Prediction API")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ExtremeWeatherModel(input_channels=5, hidden_dim=16).to(device)

try:
    model.load_state_dict(torch.load("convlstm_extreme_weather.pt", map_location=device))
    model.eval()
except Exception as e:
    print(f"Warning: Model not loaded. {e}")

# --- HELPER FUNCTIONS ---
def run_autoregressive_loop(initial_window, max_precip_scale=20.0):
    """Runs the 24-step loop to predict tomorrow's trajectory."""
    current_window = initial_window.clone()
    future_precip = []
    
    with torch.no_grad():
        for _ in range(24):
            raw_logits = model(current_window)
            pred_prob = torch.sigmoid(raw_logits)
            
            # Dampen feedback slightly to prevent OOD explosion on live flat data
            dampened_pred = pred_prob * 0.90 
            
            last_vars = current_window[:, -1:, 1:, :, :]
            new_step = torch.cat([dampened_pred.unsqueeze(2), last_vars], dim=2) 
            current_window = torch.cat([current_window[:, 1:, :, :, :], new_step], dim=1)
            
            # Center pixel (10, 10) for the UI chart
            future_precip.append(float(pred_prob[0, 0, 10, 10].cpu().numpy()) * max_precip_scale) 
            
    final_grid = current_window[0, -1, 0, :, :].cpu().numpy().tolist()
    return future_precip, final_grid

# --- ENDPOINT 1: LIVE DATA (Open-Meteo) ---
@app.get("/predict_live")
def predict_live():
    # Fetch live Hyderabad/Secunderabad data
    url = "https://api.open-meteo.com/v1/forecast?latitude=17.44&longitude=78.50&hourly=precipitation,dew_point_2m,surface_pressure,wind_speed_10m,wind_direction_10m&past_days=1&forecast_days=0"
    response = requests.get(url).json()
    hourly = response['hourly']
    
    h_precip = hourly['precipitation'][-24:]
    h_dew = hourly['dew_point_2m'][-24:]
    h_pres = hourly['surface_pressure'][-24:]
    w_speed = hourly['wind_speed_10m'][-24:]
    w_dir = hourly['wind_direction_10m'][-24:]
    
    h_u = [-s * math.sin(math.radians(d)) for s, d in zip(w_speed, w_dir)]
    h_v = [-s * math.cos(math.radians(d)) for s, d in zip(w_speed, w_dir)]
    
    # Gaussian spatial mask to simulate a local system and prevent flat-grid panic
    x, y = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21))
    spatial_mask = np.exp(-((np.sqrt(x*x + y*y))**2 / 0.8)) 

    tensor_data = np.zeros((1, 24, 5, 21, 21), dtype=np.float32)
    for t in range(24):
        tensor_data[0, t, 0, :, :] = min(h_precip[t] / 35.0, 1.0) * spatial_mask
        tensor_data[0, t, 1, :, :] = ((h_dew[t] - 5) / 30.0) * spatial_mask 
        tensor_data[0, t, 2, :, :] = (h_u[t] + 20) / 40.0
        tensor_data[0, t, 3, :, :] = (h_v[t] + 20) / 40.0
        tensor_data[0, t, 4, :, :] = (h_pres[t] - 900) / 150.0

    future_precip, final_grid = run_autoregressive_loop(torch.tensor(tensor_data).to(device), max_precip_scale=5.0)
    
    return {
        "historical_24h": {"precip": h_precip, "dew": h_dew, "u_wind": h_u, "v_wind": h_v, "pressure": h_pres},
        "future_24h_precip": future_precip,
        "probability_grid": final_grid
    }

# --- ENDPOINT 2: HISTORICAL STORM DATA (ERA5 .nc) ---
@app.get("/predict_historical")
def predict_historical():
    ds = xr.open_dataset('testing/telangana_weather_test_data.nc', engine='netcdf4')
    precip_array = ds['tp'].values
    max_idx = np.unravel_index(precip_array.argmax(), precip_array.shape)[0]
    start_idx = max(0, max_idx - 24)
    past_24h = ds.isel(valid_time=slice(start_idx, start_idx + 24))
    
    h_precip = np.nan_to_num(past_24h['tp'].values)
    h_dew = np.nan_to_num(past_24h['d2m'].values)
    h_u = np.nan_to_num(past_24h['u10'].values)
    h_v = np.nan_to_num(past_24h['v10'].values)
    h_pres = np.nan_to_num(past_24h['sp'].values)
    
    tensor_data = np.zeros((1, 24, 5, 21, 21), dtype=np.float32)
    tensor_data[0, :, 0, :, :] = h_precip / (np.max(h_precip) + 1e-5)
    tensor_data[0, :, 1, :, :] = (h_dew - np.min(h_dew)) / (np.ptp(h_dew) + 1e-5)
    tensor_data[0, :, 2, :, :] = (h_u - np.min(h_u)) / (np.ptp(h_u) + 1e-5)
    tensor_data[0, :, 3, :, :] = (h_v - np.min(h_v)) / (np.ptp(h_v) + 1e-5)
    tensor_data[0, :, 4, :, :] = (h_pres - np.min(h_pres)) / (np.ptp(h_pres) + 1e-5)

    future_precip, final_grid = run_autoregressive_loop(torch.tensor(tensor_data).to(device), max_precip_scale=20.0)
    
    return {
        "historical_24h": {
            "precip": (h_precip[:, 10, 10] * 1000).tolist(),
            "dew": (h_dew[:, 10, 10] - 273.15).tolist(), # Kelvin to C
            "u_wind": h_u[:, 10, 10].tolist(),
            "v_wind": h_v[:, 10, 10].tolist(),
            "pressure": (h_pres[:, 10, 10] / 100).tolist() # Pa to hPa
        },
        "future_24h_precip": future_precip,
        "probability_grid": final_grid
    }