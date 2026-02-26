# app.py
import streamlit as st
import requests
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Extreme Weather AI")

st.title("🌩️ Extreme Precipitation Early Warning System")
st.markdown("Powered by ConvLSTM & Copernicus ERA5 Reanalysis Data")

# Sidebar Controls
st.sidebar.header("System Controls")
user_threshold = st.sidebar.slider(
    "Warning Threshold (%)", 
    min_value=1.0, max_value=100.0, value=11.6, step=1.0,
    help="Adjust the probability threshold required to trigger a red alert."
) / 100.0

if st.sidebar.button("Predict Next Hour", type="primary"):
    with st.spinner("Fetching 24-hour climate history & running ConvLSTM..."):
        
        # 1. Ping the FastAPI Backend
        response = requests.get("http://127.0.0.1:8000/predict_next_hour")
        
        if response.status_code == 200:
            data = response.json()
            prob_grid = np.array(data["probability_grid"])
            proof = data["proof_metrics"]
            
            col1, col2 = st.columns([2, 1])
            
            # --- COLUMN 1: The Map ---
            with col1:
                st.subheader("Geographical Risk Map")
                # Center map over Telangana
                m = folium.Map(location=[17.5, 79.5], zoom_start=6, tiles="CartoDB dark_matter")
                
                # Overlay the 21x21 grid
                # Bounding box: [20, 77] (NW) to [15, 82] (SE)
                lats = np.linspace(20, 15, 21)
                lons = np.linspace(77, 82, 21)
                
                for i in range(21):
                    for j in range(21):
                        prob = prob_grid[i, j]
                        if prob >= user_threshold:
                            # Draw a red rectangle for danger zones
                            folium.Rectangle(
                                bounds=[[lats[i], lons[j]], [lats[i]-0.25, lons[j]+0.25]],
                                color="red",
                                fill=True,
                                fill_opacity=prob, # Higher probability = darker red
                                tooltip=f"Risk: {prob*100:.1f}%"
                            ).add_to(m)
                            
                st_folium(m, width=700, height=500)

            # --- COLUMN 2: The "Proof" & Analytics ---
            with col2:
                st.subheader("Diagnostic Proof")
                st.info(f"Target Time: {data['target_time']}\n\nDominant Wind: {proof['dominant_wind_direction']}")
                
                st.markdown("**Why is this triggering?**")
                st.write("The model detected a sharp escalation in regional moisture (Dewpoint) over the last 6 hours, combined with converging wind vectors.")
                
                # Plot the 24-hour trend to prove it to the user
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=proof['moisture_trend_24h'], 
                    mode='lines+markers',
                    name='Moisture Trend',
                    line=dict(color='cyan', width=2)
                ))
                fig.update_layout(
                    title="24-Hour Regional Moisture Context",
                    xaxis_title="Hours Ago (T-24 to T-0)",
                    yaxis_title="Normalized Intensity",
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.error("Failed to connect to the prediction API.")