import streamlit as st
import requests
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.graph_objects as go
from shapely.geometry import shape, Point
from shapely.ops import unary_union

st.set_page_config(layout="wide", page_title="Extreme Weather AI")
st.title("🌩️ Extreme Precipitation Early Warning System")
st.markdown("**(Spatio-Temporal ConvLSTM Anomaly Detector)**")

if 'api_data' not in st.session_state:
    st.session_state.api_data = None
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = None

st.sidebar.header("System Controls")
data_mode = st.sidebar.radio(
    "Select Data Source:",
    ("Live Secunderabad Data (Current)", "Historical Storm Simulation (Sept 2023)")
)

user_threshold = st.sidebar.slider("Warning Threshold (%)", 1.0, 100.0, 11.6, 1.0) / 100.0

if st.sidebar.button("Run Prediction Pipeline", type="primary"):
    with st.spinner("Processing spatial grid and running ConvLSTM..."):
        api_endpoint = "predict_live" if "Live" in data_mode else "predict_historical"
        
        response = requests.get(f"http://127.0.0.1:8000/{api_endpoint}")
        if response.status_code == 200:
            st.session_state.api_data = response.json()
            st.session_state.current_mode = data_mode
        else:
            st.error("Backend API failed.")

if st.session_state.api_data:
    data = st.session_state.api_data
    hist = data["historical_24h"]
    future = data["future_24h_precip"]
    prob_grid = np.array(data["probability_grid"])
    mode_label = st.session_state.current_mode.split('(')[0].strip()
    
    st.divider()
    
    st.subheader(f"Atmospheric Conditions ({mode_label})")
    st.caption(f"**T-Zero (Time of Prediction):** {data.get('target_date', 'Unknown Date')}")
    
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
        
        # 1. Fetch Telangana GeoJSON
        geojson_url = "https://raw.githubusercontent.com/gpavanb1/Telangana-Visualisation/master/data/Telangana.geojson"
        tg_geo = requests.get(geojson_url).json()
        
        # Add the state outline to the map
        folium.GeoJson(
            tg_geo,
            name="Telangana Boundary",
            style_function=lambda x: {'color': '#00FFFF', 'weight': 2, 'fillOpacity': 0.0}
        ).add_to(m)

        # 2. Convert GeoJSON to Shapely Geometry AND FIX THE ERROR
        # The .buffer(0) mathematically cleans the messy geometry so unary_union won't crash
        try:
            clean_geoms = [shape(f['geometry']).buffer(0) for f in tg_geo['features']]
            tg_shape = unary_union(clean_geoms)
        except Exception as e:
            st.error(f"Geometry masking failed: {e}")
            tg_shape = None # Fallback in case the GeoJSON is entirely corrupt

        # 3. Filter the grid points
        heat_data = []
        for i in range(21):
            for j in range(21):
                prob = prob_grid[i, j]
                if prob >= user_threshold:
                    # If we successfully built the state shape, mask it!
                    if tg_shape:
                        pt = Point(lons[j], lats[i])
                        if tg_shape.contains(pt): 
                            heat_data.append([float(lats[i]), float(lons[j]), float(prob)])
                    else:
                        # Fallback: Just show the whole grid if masking fails
                        heat_data.append([float(lats[i]), float(lons[j]), float(prob)])
        
        # 4. Render the glowing HeatMap
        if heat_data:
            HeatMap(heat_data, radius=35, blur=20, min_opacity=0.4).add_to(m)

        st_folium(m, width=600, height=450)

    with col2:
        st.subheader("Precipitation Timeline Analysis")
        
        if max_risk < (user_threshold * 100):
            st.info("💡 **Graph Context:** Grey line is actual recent rainfall. Dotted cyan line is flat because the autoregressive model predicts no imminent disruption.")
        else:
            st.warning("💡 **Graph Context:** Grey line maps the storm's historical buildup. Dotted cyan curve represents the ConvLSTM's projected intensity over the next 24 hours.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(-24, 0)), y=hist['precip'], mode='lines', name='Actual (Past 24h)', line=dict(color='gray', width=3)))
        fig.add_trace(go.Scatter(x=list(range(0, 24)), y=future, mode='lines', name='Predicted (Next 24h)', line=dict(color='cyan', width=3, dash='dot')))
        fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red", annotation_text="T-Zero")
        
        fig.update_layout(xaxis_title="Hours from T-Zero", yaxis_title="Rainfall (mm)", height=450)
        st.plotly_chart(fig, use_container_width=True)