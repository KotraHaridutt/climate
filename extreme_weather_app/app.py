import streamlit as st
import torch
import numpy as np
import requests
import math
import xarray as xr
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.graph_objects as go
from shapely.geometry import shape, Point
from shapely.ops import unary_union
import os

from model import ExtremeWeatherModel # Make sure model.py is in the same folder!

st.set_page_config(layout="wide", page_title="Extreme Weather AI")
st.title("🌩️ Extreme Precipitation Early Warning System")
st.markdown("**(Spatio-Temporal ConvLSTM Anomaly Detector)**")

# --- 1. CACHE THE MODEL (Crucial for Cloud Run) ---
@st.cache_resource
def load_ai_model():
    # Cloud Run instances use CPUs, so we force CPU mapping here to prevent crashes
    device = torch.device("cpu")
    model = ExtremeWeatherModel(input_channels=5, hidden_dim=16).to(device)
    if os.path.exists("convlstm_extreme_weather.pt"):
        model.load_state_dict(torch.load("convlstm_extreme_weather.pt", map_location=device))
        model.eval()
        return model, device
    else:
        st.error("Model weights not found!")
        return None, None

model, device = load_ai_model()

# --- 2. THE AI LOGIC ---
def run_autoregressive_loop(initial_window, max_precip_scale=20.0):
    current_window = initial_window.clone()
    future_precip = []
    with torch.no_grad():
        for _ in range(24):
            raw_logits = model(current_window)
            pred_prob = torch.sigmoid(raw_logits)
            dampened_pred = pred_prob * 0.90 
            last_vars = current_window[:, -1:, 1:, :, :]
            new_step = torch.cat([dampened_pred.unsqueeze(2), last_vars], dim=2) 
            current_window = torch.cat([current_window[:, 1:, :, :, :], new_step], dim=1)
            future_precip.append(float(pred_prob[0, 0, 10, 10].numpy()) * max_precip_scale) 
    final_grid = current_window[0, -1, 0, :, :].numpy().tolist()
    return future_precip, final_grid

def fetch_and_predict(mode):
    if mode == "Live":
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
        
        x, y = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21))
        spatial_mask = np.exp(-((np.sqrt(x*x + y*y))**2 / 0.8)) 

        tensor_data = np.zeros((1, 24, 5, 21, 21), dtype=np.float32)
        for t in range(24):
            tensor_data[0, t, 0, :, :] = min(h_precip[t] / 35.0, 1.0) * spatial_mask
            tensor_data[0, t, 1, :, :] = ((h_dew[t] - 5) / 30.0) * spatial_mask 
            tensor_data[0, t, 2, :, :] = (h_u[t] + 20) / 40.0
            tensor_data[0, t, 3, :, :] = (h_v[t] + 20) / 40.0
            tensor_data[0, t, 4, :, :] = (h_pres[t] - 900) / 150.0

        future_precip, final_grid = run_autoregressive_loop(torch.tensor(tensor_data), max_precip_scale=5.0)
        current_time = pd.Timestamp.now(tz='Asia/Kolkata').strftime('%B %d, %Y at %I:%M %p IST')
        
        return {"target_date": current_time, "historical_24h": {"precip": h_precip, "dew": h_dew, "u_wind": h_u, "v_wind": h_v, "pressure": h_pres}, "future_24h_precip": future_precip, "probability_grid": final_grid}
    
    else: # Historical Mode
        ds = xr.open_dataset('telangana_weather_test_data.nc', engine='netcdf4')
        precip_array = ds['tp'].values
        max_idx = np.unravel_index(precip_array.argmax(), precip_array.shape)[0]
        start_idx = max(0, max_idx - 24)
        past_24h = ds.isel(valid_time=slice(start_idx, start_idx + 24))
        
        t_zero_np = past_24h['valid_time'].values[-1]
        t_zero_str = pd.to_datetime(t_zero_np).strftime('%B %d, %Y at %H:%M UTC')
        
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

        future_precip, final_grid = run_autoregressive_loop(torch.tensor(tensor_data), max_precip_scale=20.0)
        
        return {"target_date": t_zero_str, "historical_24h": {"precip": (h_precip[:, 10, 10] * 1000).tolist(), "dew": (h_dew[:, 10, 10] - 273.15).tolist(), "u_wind": h_u[:, 10, 10].tolist(), "v_wind": h_v[:, 10, 10].tolist(), "pressure": (h_pres[:, 10, 10] / 100).tolist()}, "future_24h_precip": future_precip, "probability_grid": final_grid}

# --- 3. THE UI FRONTEND ---
if 'api_data' not in st.session_state:
    st.session_state.api_data = None
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = None

st.sidebar.header("System Controls")
data_mode = st.sidebar.radio("Select Data Source:", ("Live Secunderabad Data (Current)", "Historical Storm Simulation (Sept 2023)"))
user_threshold = st.sidebar.slider("Warning Threshold (%)", 1.0, 100.0, 11.6, 1.0) / 100.0

if st.sidebar.button("Run Prediction Pipeline", type="primary"):
    if model is None:
        st.error("Cannot run: Model failed to load.")
    else:
        with st.spinner("Processing spatial grid and running ConvLSTM..."):
            mode_param = "Live" if "Live" in data_mode else "Historical"
            st.session_state.api_data = fetch_and_predict(mode_param)
            st.session_state.current_mode = data_mode

if st.session_state.api_data:
    data = st.session_state.api_data
    hist = data["historical_24h"]
    future = data["future_24h_precip"]
    prob_grid = np.array(data["probability_grid"])
    mode_label = st.session_state.current_mode.split('(')[0].strip()
    
    st.divider()
    st.subheader(f"Atmospheric Conditions ({mode_label})")
    st.caption(f"**T-Zero (Time of Prediction):** {data.get('target_date')}")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Dewpoint (Moisture)", f"{hist['dew'][-1]:.1f} °C") 
    m2.metric("Surface Pressure", f"{hist['pressure'][-1]:.0f} hPa")
    m3.metric("U-Wind Vector", f"{hist['u_wind'][-1]:.1f} m/s")
    m4.metric("V-Wind Vector", f"{hist['v_wind'][-1]:.1f} m/s")
    m5.metric("Last 24h Rainfall", f"{sum(hist['precip']):.1f} mm")
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Predicted Spatial Risk Heatmap")
        max_risk = np.max(prob_grid) * 100
        if max_risk < (user_threshold * 100):
            st.success(f"**Status: Clear.** Maximum regional risk is {max_risk:.1f}%. Map remains clear.")
        else:
            st.error(f"**Status: Extreme Danger Detected.** Areas exceeding the {user_threshold*100:.1f}% threshold are highlighted.")

        m = folium.Map(location=[17.5, 79.5], zoom_start=6, tiles="CartoDB dark_matter")
        lats, lons = np.linspace(20, 15, 21), np.linspace(77, 82, 21)
        
        try:
            geojson_url = "https://raw.githubusercontent.com/gpavanb1/Telangana-Visualisation/master/data/Telangana.geojson"
            tg_geo = requests.get(geojson_url).json()
            folium.GeoJson(tg_geo, style_function=lambda x: {'color': '#00FFFF', 'weight': 2, 'fillOpacity': 0.0}).add_to(m)
            clean_geoms = [shape(f['geometry']).buffer(0) for f in tg_geo['features']]
            tg_shape = unary_union(clean_geoms)
        except Exception:
            tg_shape = None

        heat_data = []
        for i in range(21):
            for j in range(21):
                prob = prob_grid[i, j]
                if prob >= user_threshold:
                    if tg_shape:
                        pt = Point(lons[j], lats[i])
                        if tg_shape.contains(pt): 
                            heat_data.append([float(lats[i]), float(lons[j]), float(prob)])
                    else:
                        heat_data.append([float(lats[i]), float(lons[j]), float(prob)])
        
        if heat_data:
            HeatMap(heat_data, radius=35, blur=20, min_opacity=0.4).add_to(m)

        st_folium(m, width=600, height=450)

    with col2:
        st.subheader("Precipitation Timeline Analysis")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(-24, 0)), y=hist['precip'], mode='lines', name='Actual (Past 24h)', line=dict(color='gray', width=3)))
        fig.add_trace(go.Scatter(x=list(range(0, 24)), y=future, mode='lines', name='Predicted (Next 24h)', line=dict(color='cyan', width=3, dash='dot')))
        fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red", annotation_text="T-Zero")
        fig.update_layout(xaxis_title="Hours from T-Zero", yaxis_title="Rainfall (mm)", height=450)
        st.plotly_chart(fig, use_container_width=True)