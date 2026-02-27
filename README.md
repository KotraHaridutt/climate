# 🌩️ Extreme Precipitation Early Warning System (v1.0)
**An AI-Driven Spatio-Temporal Anomaly Detector using ConvLSTM and ERA5 Climate Data**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)
![Google Cloud Run](https://img.shields.io/badge/GCP-Cloud%20Run-4285F4)



## 🚨 The Problem: Why not just use AccuWeather?
Standard weather applications rely on Numerical Weather Prediction (NWP)—massive, physics-based simulations that calculate global weather. While excellent for general temperature forecasts, NWPs inherently "smooth out" local data, frequently failing to predict hyper-local, non-linear extreme events like sudden flash floods. 

This project solves a highly specific problem in climate tech: **fast, localized extreme weather early warnings**. Instead of simulating global physics, this AI treats weather as a multi-channel video, learning the mathematical relationships between converging wind vectors and moisture to detect anomalies standard models miss.

## 🧠 Architecture & Methodology

### 1. Spatio-Temporal Data Modeling (The 5D Tensor)
The model does not take simple tabular data. It ingests a 24-hour localized geographical grid, creating a multi-dimensional tensor shaped strictly as `(Batch, Time, Channels, Height, Width)` or `(1, 24, 5, 21, 21)`. 
The 5 physical channels analyzed at every geographical pixel are:
* Total Precipitation
* Dewpoint Temperature (Moisture content)
* Surface Pressure
* U-Wind Component (Calculated via trigonometry from raw speed/direction)
* V-Wind Component

### 2. The Deep Learning Engine: ConvLSTM
Built from scratch in PyTorch, the network utilizes a **Convolutional Long Short-Term Memory (ConvLSTM)** architecture. 
* Standard CNNs understand spatial maps but have no concept of time.
* Standard LSTMs understand time but destroy spatial relationships.
* **ConvLSTM** replaces matrix multiplications with convolution operations inside the LSTM cell, allowing the network to watch the "movie" of clouds moving across the Deccan Plateau and predict the next logical frame.

### 3. The 24-Hour Autoregressive Loop
For future forecasting, the API does not make a single jump. It predicts `T+1`, dynamically injects its own prediction back into the spatial grid, shifts the memory window forward, and predicts `T+2`. It loops this mathematical sequence 24 times to map the storm's future trajectory.

## 📊 Model Evaluation & Metrics
Because extreme rainfall is incredibly rare (present in <1% of the dataset), standard "Accuracy" is a deceptive trap. A broken model that always guesses "No Rain" would be 99% accurate. 

The model was tested on unseen September 2023 ERA5 data, heavily prioritizing the identification of rare flood-risk events:
* **Recall (70.86%):** The golden metric for disaster prediction. The model successfully detected and flagged 7 out of 10 actual extreme rain events.
* **F1-Score (0.5359):** A highly robust harmonic mean for a baseline ConvLSTM dealing with severe spatial class imbalance.

## ⚙️ Features
* **Dual-Mode Inference:**
    * **Live Mode:** Connects to the Open-Meteo API to fetch the last 24 hours of real climate data, applies a Gaussian spatial mask, and runs real-time predictions.
    * **Historical Mode:** Loads `.nc` (NetCDF) ERA5 Reanalysis data to simulate a catastrophic September 2023 storm for demonstration purposes.
* **Geospatial Masking:** Uses `shapely` and GeoJSON boundaries to mathematically clip the prediction heatmap so it perfectly molds to the physical borders of the target state (Telangana).
* **Containerized Deployment:** Fused frontend and PyTorch backend, dockerized, and hosted entirely on Google Cloud Run.

## 🚀 Running Locally
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
2. Ensure you are in the `extreme_weather_app` folder.
   ```bash
   cd extreme_weather_app
3. Run the application locally:
   ```bash
   streamlit run app.py
