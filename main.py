import streamlit as st
import joblib
import numpy as np

# Label mapping for output
label_map = {0: 'Cloudy', 1: 'Rainy', 2: 'Snowy', 3: 'Sunny'}

# Encoding maps (from training step)
cloud_map = {'clear': 0, 'overcast': 1, 'partly cloudy': 2}
season_map = {'Autumn': 0, 'Spring': 1, 'Summer': 2, 'Winter': 3}
location_map = {'coastal': 0, 'inland': 1, 'mountain': 2}

# Load the trained model
model = joblib.load("best_rf.pkl")

# App title
st.title("Weather Classifier")

# Inputs for numerical features
temperature = st.slider('Temperature (°C)', -20.0, 50.0)
humidity = st.slider('Humidity (%)', 10.0, 100.0)
wind_speed = st.slider('Wind Speed (km/h)', 0.0, 30.0)
precipitation = st.slider('Precipitation (%)', 0.0, 100.0)
pressure = st.slider('Atmospheric Pressure (hPa)', 950.0, 1050.0)
uv_index = st.slider('UV Index', 0, 11)
visibility = st.slider('Visibility (km)', 0.0, 20.0)

# Inputs for categorical features
cloud_cover = st.selectbox('Cloud Cover', list(cloud_map.keys()))
season = st.selectbox('Season', list(season_map.keys()))
location = st.selectbox('Location', list(location_map.keys()))

# Encode categorical inputs
cloud_encoded = cloud_map[cloud_cover]
season_encoded = season_map[season]
location_encoded = location_map[location]

# Final feature array (in training order)
features = np.array([[temperature, humidity, wind_speed, precipitation,
                      cloud_encoded, pressure, uv_index, season_encoded,
                      visibility, location_encoded]])

# Prediction
if st.button("Classify"):
    result = model.predict(features)[0]
    label = label_map.get(result, "Unknown")
    st.success(f"Predicted Weather Type: {label}")
