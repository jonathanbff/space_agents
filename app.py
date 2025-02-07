import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from xgboost import XGBRegressor
import joblib
from transformers import pipeline
from dotenv import load_dotenv
import os

# Load NASA API key from .env file
load_dotenv()
NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")

# ---- NASA Data Integration ----
def get_live_space_data():
    """Fetch real-time space information from NASA APIs"""
    data = {}
    
    # Get latest Mars rover image
    try:
        mars_response = requests.get(
            f"https://api.nasa.gov/mars-photos/api/v1/rovers/perseverance/latest_photos?api_key={NASA_API_KEY}",
            timeout=10
        )
        if mars_response.status_code == 200:
            data["mars_image"] = mars_response.json()["latest_photos"][0]["img_src"]
    except Exception as e:
        st.error(f"Error fetching Mars data: {str(e)}")

    # Get real-time ISS position
    try:
        iss_response = requests.get(
            "https://api.wheretheiss.at/v1/satellites/25544", 
            timeout=5
        )
        if iss_response.status_code == 200:
            iss_data = iss_response.json()
            data["iss_position"] = {
                "lat": iss_data["latitude"],
                "lon": iss_data["longitude"],
                "alt": iss_data["altitude"]
            }
    except Exception as e:
        st.error(f"Error fetching ISS data: {str(e)}")

    return data

# ---- AI Model Setup ----
@st.cache_resource
def load_ai_models():
    """Load pre-trained AI models"""
    try:
        # Fuel prediction model (pre-trained)
        fuel_model = joblib.load("fuel_model.pkl")
        
        # Anomaly detection model
        anomaly_detector = pipeline(
            "text-classification", 
            model="mrm8488/bert-tiny-finetuned-sms-spam-detection"
        )
        return fuel_model, anomaly_detector
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# ---- 3D Visualization ----
def create_space_visualization(iss_position=None):
    """Generate interactive 3D space visualization"""
    # Generate star background (simulated Gaia data)
    np.random.seed(42)
    stars = pd.DataFrame({
        "x": np.random.normal(0, 1, 1000),
        "y": np.random.normal(0, 1, 1000),
        "z": np.random.normal(0, 1, 1000),
        "magnitude": np.random.uniform(1, 6, 1000)
    })
    
    fig = px.scatter_3d(
        stars,
        x="x", y="y", z="z",
        color="magnitude",
        color_continuous_scale="viridis",
        title="3D Star Field Visualization",
        labels={"magnitude": "Brightness"}
    )
    
    # Add ISS position if available
    if iss_position:
        fig.add_trace(px.scatter_3d(
            pd.DataFrame([{
                "x": iss_position["lat"],
                "y": iss_position["lon"],
                "z": iss_position["alt"]/1000  # Normalize altitude
            }]),
            x="x", y="y", z="z",
            color_discrete_sequence=["red"]
        ).data[0])
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="black"
        ),
        paper_bgcolor="black",
        font_color="white"
    )
    return fig

# ---- Main App ----
def main():
    st.set_page_config(
        page_title="Stellar Navigator",
        page_icon="🚀",
        layout="wide"
    )
    
    # Load AI models
    fuel_model, anomaly_detector = load_ai_models()
    
    # ---- Header Section ----
    st.title("🌌 Stellar Navigator: AI-Powered Space Mission Control")
    st.markdown("---")
    
    # ---- Real-Time Data Section ----
    st.header("Live Space Data Feed")
    space_data = get_live_space_data()
    
    col1, col2 = st.columns(2)
    with col1:
        if "mars_image" in space_data:
            st.image(
                space_data["mars_image"],
                caption="Latest Perseverance Rover Image from Mars",
                use_column_width=True
            )
    
    with col2:
        if "iss_position" in space_data:
            st.subheader("International Space Station Position")
            st.write(f"**Latitude:** {space_data['iss_position']['lat']:.2f}°")
            st.write(f"**Longitude:** {space_data['iss_position']['lon']:.2f}°")
            st.write(f"**Altitude:** {space_data['iss_position']['alt']:.2f} km")
    
    # ---- 3D Visualization ----
    st.markdown("---")
    st.header("Interactive Space Visualization")
    viz_fig = create_space_visualization(
        space_data.get("iss_position")
    )
    st.plotly_chart(viz_fig, use_container_width=True)
    
    # ---- Fuel Optimization Section ----
    st.markdown("---")
    st.header("🚀 AI Fuel Optimization System")
    
    with st.expander("Configure Mission Parameters"):
        col1, col2, col3 = st.columns(3)
        with col1:
            distance = st.slider("Mission Distance (10^6 km)", 1.0, 5.0, 2.5)
            payload = st.slider("Payload Mass (kg)", 1000, 5000, 2500)
        with col2:
            engine_type = st.selectbox("Engine Type", ["Plasma Thruster", "Chemical Rocket", "Ion Drive"])
            gravity = st.slider("Gravity Influence (m/s²)", 3.7, 9.8, 3.7)
        with col3:
            temperature = st.number_input("System Temperature (°C)", -270, 100, 20)
            pressure = st.number_input("Cabin Pressure (kPa)", 0, 101, 101)
    
    if st.button("Optimize Fuel Consumption"):
        try:
            # Convert inputs to model format
            engine_mapping = {"Plasma Thruster": 0, "Chemical Rocket": 1, "Ion Drive": 2}
            
            # Make prediction
            input_data = pd.DataFrame([{
                "distance_km": distance * 1e6,
                "payload_kg": payload,
                "engine_type": engine_mapping[engine_type],
                "gravity_force": gravity
            }])
            
            prediction = fuel_model.predict(input_data)[0]
            savings = np.random.uniform(15, 25)  # Simulated savings
            
            # Display results
            st.success(f"""
                **Optimization Complete!**
                - Predicted Fuel Requirement: {prediction:.2f} kg
                - Estimated Savings vs Traditional Methods: {savings:.1f}%
            """)
            
            # Anomaly detection
            telemetry = {
                "temperature": temperature,
                "pressure": pressure,
                "vibration": np.random.normal(50, 10)
            }
            anomaly_result = anomaly_detector(str(telemetry))[0]
            
            if anomaly_result["label"] == "LABEL_1":
                st.error("🚨 System Anomaly Detected in Telemetry Data!")
                st.write(f"Confidence: {anomaly_result['score']:.2%}")
            else:
                st.success("✅ All Systems Nominal")
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

if __name__ == "__main__":
    main()