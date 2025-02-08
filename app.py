# app.py

import os
import logging
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBRegressor
import joblib
from transformers import pipeline
from dotenv import load_dotenv

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")

# ---- Data Sources ----
# Updated URL: use HYG v4.1 from the CURRENT folder.
HYG_STAR_DATABASE = "https://raw.githubusercontent.com/astronexus/HYG-Database/main/hyg/CURRENT/hyg_v41.csv"
ISS_API = "https://api.wheretheiss.at/v1/satellites/25544"
NASA_APOD_API = f"https://api.nasa.gov/planetary/apod?api_key={NASA_API_KEY}"


# ---- Live Data Fetching ----
def get_live_space_data() -> dict:
    """
    Fetch real-time space information from verified APIs:
    - ISS position from wheretheiss.at
    - NASA Astronomy Picture of the Day (APOD)
    """
    data = {"iss_position": None, "apod": None}
    try:
        # Get real-time ISS position
        iss_response = requests.get(ISS_API, timeout=5)
        if iss_response.status_code == 200:
            iss_data = iss_response.json()
            data["iss_position"] = {
                "lat": iss_data["latitude"],
                "lon": iss_data["longitude"],
                "alt": iss_data["altitude"]
            }
        else:
            st.warning("Could not fetch ISS data.")

        # Get NASA APOD
        apod_response = requests.get(NASA_APOD_API, timeout=10)
        if apod_response.status_code == 200:
            data["apod"] = apod_response.json()
        else:
            st.warning("Could not fetch NASA APOD data.")
    except Exception as e:
        st.error(f"Error fetching live space data: {e}")
    return data


# ---- Model Loading ----
@st.cache_resource
def load_ai_models() -> tuple:
    """
    Load or create AI models with error handling.
    - Fuel optimization model (XGBRegressor)
    - Anomaly detection model using a pretrained transformer pipeline
    """
    try:
        if not os.path.exists("fuel_model.pkl"):
            st.info("Training initial fuel model...")
            np.random.seed(42)
            data = {
                "distance_km": np.random.uniform(1e6, 5e6, 1000),
                "payload_kg": np.random.randint(1000, 5000, 1000),
                "engine_type": np.random.choice([0, 1, 2], 1000),
                "gravity_force": np.random.uniform(3.7, 9.8, 1000),
                "fuel_used_kg": np.random.uniform(5000, 20000, 1000)
            }
            df = pd.DataFrame(data)
            model = XGBRegressor()
            model.fit(df.drop("fuel_used_kg", axis=1), df["fuel_used_kg"])
            joblib.dump(model, "fuel_model.pkl")

        fuel_model = joblib.load("fuel_model.pkl")
        anomaly_detector = pipeline(
            "text-classification",
            model="mrm8488/bert-tiny-finetuned-sms-spam-detection"
        )
        return fuel_model, anomaly_detector
    except Exception as e:
        st.error(f"Error loading AI models: {e}")
        return None, None


# ---- Visualization Functions ----
def create_iss_globe(lat: float, lon: float, alt: float) -> go.Figure:
    """
    Create a 3D globe visualization of the ISS position.
    
    Args:
        lat (float): Latitude of the ISS.
        lon (float): Longitude of the ISS.
        alt (float): Altitude of the ISS.
    
    Returns:
        go.Figure: Plotly figure showing the ISS on a globe.
    """
    fig = go.Figure(go.Scattergeo(
        lat=[lat],
        lon=[lon],
        text=[f"ISS Position<br>Altitude: {alt:.2f} km"],
        marker=dict(color='red', size=12, symbol='circle')
    ))
    fig.update_geos(
        projection_type="orthographic",
        landcolor="rgb(243, 243, 243)",
        showland=True,
        oceancolor="LightBlue",
        lakecolor="LightBlue"
    )
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, b=0, t=40),
        title="Live ISS Position"
    )
    return fig


@st.cache_data
def load_star_data() -> pd.DataFrame:
    """
    Load HYG star catalog data from remote CSV.
    
    Returns:
        pd.DataFrame: Star catalog data.
    """
    df = pd.read_csv(HYG_STAR_DATABASE)
    return df


def create_starfield(max_mag: float = 6.5) -> go.Figure:
    """
    Create a 3D starfield visualization using the HYG catalog.
    
    Args:
        max_mag (float, optional): Maximum star magnitude to include. Defaults to 6.5.
    
    Returns:
        go.Figure: Plotly 3D scatter plot of stars.
    """
    try:
        stars = load_star_data()
        stars_filtered = stars[stars['mag'] < max_mag]
        fig = px.scatter_3d(
            stars_filtered,
            x='x', y='y', z='z',
            color='mag',
            color_continuous_scale='viridis',
            title="Real Star Field Visualization (HYG v4.1 Catalog)",
            opacity=0.8
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor='black'
            ),
            paper_bgcolor='black',
            font_color='white',
            margin=dict(l=0, r=0, b=0, t=40)
        )
        return fig
    except Exception as e:
        st.error(f"Error loading star data: {e}")
        return go.Figure()


def predict_fuel_usage(fuel_model, distance: float, payload: int, engine_type: str) -> float:
    """
    Predict fuel usage using the AI fuel model given mission parameters.
    
    Args:
        fuel_model: The preloaded fuel optimization model.
        distance (float): Mission distance in million km.
        payload (int): Payload mass in kg.
        engine_type (str): Type of engine selected.
    
    Returns:
        float: Predicted fuel usage (kg).
    """
    engine_mapping = {"Plasma Thruster": 0, "Chemical Rocket": 1, "Ion Drive": 2}
    input_data = pd.DataFrame([{
        "distance_km": distance * 1e6,
        "payload_kg": payload,
        "engine_type": engine_mapping.get(engine_type, 1),
        "gravity_force": 3.7  # Example: Mars gravity
    }])
    prediction = fuel_model.predict(input_data)[0]
    return prediction


def plot_fuel_usage_vs_distance(fuel_model, payload: int, engine_type: str) -> go.Figure:
    """
    Generate a line chart showing predicted fuel usage vs. distance for a given payload and engine type.
    
    Args:
        fuel_model: The fuel optimization model.
        payload (int): Payload mass in kg.
        engine_type (str): Engine type.
    
    Returns:
        go.Figure: Plotly line chart.
    """
    engine_mapping = {"Plasma Thruster": 0, "Chemical Rocket": 1, "Ion Drive": 2}
    distances = np.linspace(1, 5, 50)  # Distance in million km
    predictions = []
    for d in distances:
        input_data = pd.DataFrame([{
            "distance_km": d * 1e6,
            "payload_kg": payload,
            "engine_type": engine_mapping.get(engine_type, 1),
            "gravity_force": 3.7
        }])
        pred = fuel_model.predict(input_data)[0]
        predictions.append(pred)
    df_plot = pd.DataFrame({
        "Distance (million km)": distances,
        "Predicted Fuel Usage (kg)": predictions
    })
    fig = px.line(df_plot, x="Distance (million km)", y="Predicted Fuel Usage (kg)",
                  title="Fuel Usage vs. Distance")
    return fig


# ---- Main App ----
def main():
    # Configure the page
    st.set_page_config(
        page_title="Stellar Navigator",
        page_icon="🚀",
        layout="wide"
    )
    st.title("🌌 Stellar Navigator: AI-Powered Space Mission Control")
    st.markdown("---")

    # Load AI models
    fuel_model, anomaly_detector = load_ai_models()

    # ---- Real-Time Data Section ----
    with st.spinner("Fetching live space data..."):
        space_data = get_live_space_data()

    col1, col2 = st.columns(2)
    with col1:
        if space_data.get("apod"):
            st.image(
                space_data["apod"]["url"],
                caption=f"NASA Astronomy Picture of the Day: {space_data['apod']['title']}",
                use_container_width=True
            )
    with col2:
        if space_data.get("iss_position"):
            iss_pos = space_data["iss_position"]
            st.subheader("🛰️ International Space Station Tracker")
            iss_fig = create_iss_globe(iss_pos["lat"], iss_pos["lon"], iss_pos["alt"])
            st.plotly_chart(iss_fig, use_container_width=True)

    # ---- Starfield Visualization ----
    st.markdown("---")
    with st.expander("✨ Interactive Star Field Explorer", expanded=True):
        max_mag = st.slider("Maximum Star Magnitude (only brighter stars shown)", 0.0, 10.0, 6.5)
        star_fig = create_starfield(max_mag)
        st.plotly_chart(star_fig, use_container_width=True)

    # ---- Fuel Optimization Section ----
    st.markdown("---")
    st.header("🚀 AI Fuel Optimization System")
    
    # Initialize session state for storing prediction history
    if "fuel_predictions" not in st.session_state:
        st.session_state.fuel_predictions = []

    col1, col2 = st.columns([1, 2])
    with col1:
        with st.form("mission_parameters"):
            st.subheader("Mission Configuration")
            distance = st.slider("Distance (million km)", 1.0, 5.0, 2.5)
            payload = st.slider("Payload Mass (kg)", 1000, 5000, 2500)
            engine_type = st.selectbox("Engine Type", ["Plasma Thruster", "Chemical Rocket", "Ion Drive"])
            show_trend = st.checkbox("Show Fuel Usage Trend")
            submitted = st.form_submit_button("Optimize Fuel")
    with col2:
        if submitted and fuel_model:
            try:
                # Predict fuel usage and simulate fuel savings percentage
                predicted_fuel = predict_fuel_usage(fuel_model, distance, payload, engine_type)
                savings_percentage = np.random.uniform(15, 25)
                st.success(f"Predicted Fuel Usage: {predicted_fuel:.2f} kg")
                st.info(f"Estimated Fuel Savings: {savings_percentage:.1f}%")

                # Store the prediction in session state for history
                st.session_state.fuel_predictions.append({
                    "Distance (million km)": distance,
                    "Payload (kg)": payload,
                    "Engine": engine_type,
                    "Predicted Fuel (kg)": predicted_fuel,
                    "Savings (%)": savings_percentage
                })

                # Display gauge visualization for fuel savings
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=savings_percentage,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fuel Savings (%)"},
                    gauge={'axis': {'range': [0, 30]},
                           'bar': {'color': "darkblue"},
                           'steps': [
                               {'range': [0, 15], 'color': "red"},
                               {'range': [15, 25], 'color': "orange"},
                               {'range': [25, 30], 'color': "green"}
                           ]}
                ))
                st.plotly_chart(gauge_fig, use_container_width=True)

                # Optionally display a fuel usage trend chart
                if show_trend:
                    trend_fig = plot_fuel_usage_vs_distance(fuel_model, payload, engine_type)
                    st.plotly_chart(trend_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")

    if st.session_state.fuel_predictions:
        st.subheader("Fuel Optimization History")
        st.dataframe(pd.DataFrame(st.session_state.fuel_predictions))

    # ---- Anomaly Detection Section ----
    st.markdown("---")
    st.header("🕵️ Mission Log Anomaly Detection")
    with st.form("anomaly_detection"):
        user_text = st.text_area("Enter mission log or message for anomaly detection", height=100)
        detect_submitted = st.form_submit_button("Analyze Log")
    if detect_submitted and anomaly_detector:
        try:
            result = anomaly_detector(user_text)[0]
            label = result["label"]
            score = result["score"]
            st.write(f"**Analysis Result:** {label} (Confidence: {score:.2f})")
            if label.lower() == "spam":
                st.warning("This message appears to be anomalous or spam!")
            else:
                st.success("This message appears normal.")
        except Exception as e:
            st.error(f"Error in anomaly detection: {e}")


if __name__ == "__main__":
    main()
