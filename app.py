import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from streamlit_folium import st_folium
import folium

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "traffy_rf_model.joblib"
PREDICTIONS_PATH = BASE_DIR / "data" / "predictions" / "traffy_with_predictions.csv"

st.set_page_config(page_title="Traffy Late Prediction", page_icon="🍫", layout="wide")

st.title("Traffy Fondue Late Prediction")
st.markdown("Predict whether a Traffy ticket will be resolved **late** or **on-time**")

# Load model
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. Please run the training pipeline first.")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# Sidebar - Input features
st.sidebar.header("Input Ticket Features")

ticket_type = st.sidebar.selectbox("Type", ["ปัญหาสาธารณสุข", "ปัญหาถนน", "ปัญหาขยะ", "ปัญหาไฟฟ้า", "อื่นๆ"])
organization = st.sidebar.text_input("Organization", "Unknown")
district = st.sidebar.text_input("District", "บางรัก")

# Map picker for location
st.sidebar.subheader("Select Location on Map")
use_map = st.sidebar.checkbox("Use Map Picker", value=True)

if use_map:
    # Initialize session state for coordinates
    if 'lat' not in st.session_state:
        st.session_state.lat = 13.75
        st.session_state.lon = 100.50
    
    # Create map centered on Bangkok
    m = folium.Map(
        location=[st.session_state.lat, st.session_state.lon],
        zoom_start=12,
        width=300,
        height=300
    )
    
    # Add marker at current location
    folium.Marker(
        [st.session_state.lat, st.session_state.lon],
        popup="Selected Location",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)
    
    # Display map in sidebar
    map_data = st_folium(m, width=300, height=300, key="location_map")
    
    # Update coordinates if map was clicked
    if map_data and map_data.get("last_clicked"):
        st.session_state.lat = map_data["last_clicked"]["lat"]
        st.session_state.lon = map_data["last_clicked"]["lng"]
    
    lat = st.session_state.lat
    lon = st.session_state.lon
    
    # Display selected coordinates
    st.sidebar.text(f"Lat: {lat:.4f}, Lon: {lon:.4f}")
else:
    lat = st.sidebar.number_input("Latitude", value=13.75, format="%.4f")
    lon = st.sidebar.number_input("Longitude", value=100.50, format="%.4f")
star = st.sidebar.slider("Star Rating", 0, 5, 0)
count_reopen = st.sidebar.number_input("Count Reopen", 0, 10, 0)

hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
dayofweek = st.sidebar.slider("Day of Week (0=Mon)", 0, 6, 0)
month = st.sidebar.slider("Month", 1, 12, 6)
year = st.sidebar.number_input("Year", 2020, 2030, 2023)

num_hospitals = st.sidebar.number_input("Hospitals in District", 0, 50, 5)
rain_mm = st.sidebar.number_input("Rain (mm)", 0.0, 100.0, 0.0, step=0.1)
is_rainy = st.sidebar.checkbox("Is Rainy Hour")
rain_last_3h = st.sidebar.number_input("Rain Last 3H (mm)", 0.0, 300.0, 0.0, step=0.1)
temperature = st.sidebar.number_input("Temperature (°C)", 15.0, 45.0, 30.0, step=0.5)
high_temp = st.sidebar.checkbox("High Temperature (>33°C)")
wind_speed = st.sidebar.number_input("Wind Speed (m/s)", 0.0, 30.0, 2.0, step=0.5)

# Create input dataframe
input_data = pd.DataFrame({
    "type": [ticket_type],
    "organization": [organization],
    "district": [district],
    "lat": [lat],
    "lon": [lon],
    "star": [star],
    "count_reopen": [count_reopen],
    "hour": [hour],
    "dayofweek": [dayofweek],
    "month": [month],
    "year": [year],
    "num_hospitals_in_district": [num_hospitals],
    "rain_mm": [rain_mm],
    "is_rainy_hour": [int(is_rainy)],
    "rain_last_3h": [rain_last_3h],
    "temperature": [temperature],
    "high_temperature": [int(high_temp)],
    "wind_speed": [wind_speed],
})

# Predict
if st.sidebar.button("Predict", type="primary"):
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0, 1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Prediction", "LATE" if prediction == 1 else "ON-TIME")
        
        with col2:
            st.metric("Probability of Late", f"{probability:.1%}")
        
        # Visual indicator
        if prediction == 1:
            st.error("This ticket is likely to be resolved **LATE** (>7 days)")
        else:
            st.success("This ticket is likely to be resolved **ON-TIME** (≤7 days)")
        
        # Show input summary
        with st.expander("Input Summary"):
            st.dataframe(input_data.T, use_container_width=True)
            
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Display recent predictions
st.markdown("---")
st.header("📊 Recent Predictions from Training")

if PREDICTIONS_PATH.exists():
    df_pred = pd.read_csv(PREDICTIONS_PATH)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Tickets", len(df_pred))
    with col2:
        late_pct = (df_pred["pred_is_late"].sum() / len(df_pred)) * 100
        st.metric("Predicted Late %", f"{late_pct:.1f}%")
    with col3:
        avg_prob = df_pred["pred_proba_late"].mean()
        st.metric("Avg Late Probability", f"{avg_prob:.1%}")
    
    # Show sample predictions
    st.subheader("Sample Predictions")
    display_cols = ["type", "district", "organization", "is_late", "pred_is_late", "pred_proba_late"]
    available_cols = [c for c in display_cols if c in df_pred.columns]
    st.dataframe(df_pred[available_cols].head(100), use_container_width=True, height=400)
else:
    st.info("No predictions file found. Run the ML training pipeline first.")

# Footer
st.markdown("---")
st.caption("Powered by Random Forest | Built with Streamlit")
