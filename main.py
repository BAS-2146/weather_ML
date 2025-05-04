import streamlit as st
import joblib
import numpy as np
import os  

# Label mapping for output
label_map = {0: 'Cloudy', 1: 'Rainy', 2: 'Snowy', 3: 'Sunny'}

# Encoding maps
cloud_map = {'clear': 0, 'overcast': 1, 'partly cloudy': 2}
season_map = {'Autumn': 0, 'Spring': 1, 'Summer': 2, 'Winter': 3}
location_map = {'coastal': 0, 'inland': 1, 'mountain': 2}

# Load the trained Random Forest model
model = joblib.load("best_rf.pkl")

# App title
st.title("Weather Classifier")

# Numerical inputs
temperature = st.slider('Temperature (°C)', -20.0, 70.0)
humidity = st.slider('Humidity (%)', 20.0, 110.0)
wind_speed = st.slider('Wind Speed (km/h)', 0.0, 30.0)
precipitation = st.slider('Precipitation (%)', 0.0, 100.0)
pressure = st.slider('Atmospheric Pressure (hPa)', 960.0, 1050.0)
uv_index = st.slider('UV Index', 0, 13)
visibility = st.slider('Visibility (km)', 0.0, 20.0)

# Categorical inputs
cloud_cover = st.selectbox('Cloud Cover', list(cloud_map.keys()))
season = st.selectbox('Season', list(season_map.keys()))
location = st.selectbox('Location', list(location_map.keys()))

# One-hot encoding: Cloud Cover (3)
cloud_vector = [0, 0, 0]
cloud_vector[cloud_map[cloud_cover]] = 1

# One-hot encoding: Season (Spring, Summer, Winter — no Autumn column)
season_vector = [0, 0, 0]
if season == "Spring":
    season_vector[0] = 1
elif season == "Summer":
    season_vector[1] = 1
elif season == "Winter":
    season_vector[2] = 1
# Autumn is the base case (all zeros)

# One-hot encoding: Location (inland, mountain — no coastal column)
location_vector = [0, 0]
if location == "inland":
    location_vector[0] = 1
elif location == "mountain":
    location_vector[1] = 1
# Coastal is the base case (all zeros)

# Final feature array in training order
features = np.array([[temperature, humidity, wind_speed, precipitation, pressure,
                      uv_index, visibility, *cloud_vector, *season_vector, *location_vector]])
st.write("🔍 Model Input:", features)

# Prediction
if st.button("Classify"):
    result = model.predict(features)[0]
    label = label_map.get(result, "Unknown")
    st.success(f"Predicted Weather Type: {label}")

# File browser display
def list_files(startpath):
    tree = ""
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        tree += f"{indent}📁 {os.path.basename(root)}\n"
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            tree += f"{subindent}📄 {f}\n"
    return tree

# Show project file structure
st.subheader("📂 Project File Structure")
st.code(list_files("."), language="markdown")
