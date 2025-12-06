import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "traffy_rf_model.joblib"
PREDICTIONS_PATH = BASE_DIR / "data" / "predictions" / "traffy_with_predictions.csv"

# Bangkok districts with approximate center coordinates
BANGKOK_DISTRICTS = {
    "บางรัก": (13.7248, 100.5265),
    "ปทุมวัน": (13.7469, 100.5362),
    "ดุสิต": (13.7777, 100.5155),
    "บางกอกน้อย": (13.7681, 100.4844),
    "บางกอกใหญ่": (13.7294, 100.4989),
    "บางขุนเทียน": (13.6469, 100.4230),
    "บางเขน": (13.8868, 100.6072),
    "บางแค": (13.7053, 100.3960),
    "บางคอแหลม": (13.7066, 100.5155),
    "บางซื่อ": (13.8032, 100.5345),
    "บางนา": (13.6670, 100.6099),
    "บางบอน": (13.6640, 100.3952),
    "บางพลัด": (13.7859, 100.4980),
    "บางระหว่าง": (13.6775, 100.5120),
    "บางกะปิ": (13.7622, 100.6424),
    "ภาษีเจริญ": (13.7310, 100.4413),
    "พญาไท": (13.7697, 100.5446),
    "ป้อมปราบศัตรูพ่าย": (13.7538, 100.5131),
    "ประเวศ": (13.6924, 100.6726),
    "ราชเทวี": (13.7507, 100.5349),
    "ราษฎร์บูรณะ": (13.6811, 100.5103),
    "ลาดกระบัง": (13.7285, 100.7489),
    "ลาดพร้าว": (13.8165, 100.6051),
    "วังทองหลาง": (13.7759, 100.5967),
    "วัฒนา": (13.7236, 100.5842),
    "สะพานสูง": (13.8222, 100.6657),
    "สวนหลวง": (13.7367, 100.6473),
    "สาทร": (13.7192, 100.5319),
    "สายไหม": (13.9015, 100.6542),
    "สัมพันธวงศ์": (13.7400, 100.5123),
    "คลองเตย": (13.7221, 100.5686),
    "คลองสาน": (13.7246, 100.5067),
    "คลองสามวา": (13.8448, 100.7204),
    "คันนายาว": (13.8298, 100.6976),
    "จตุจักร": (13.8155, 100.5542),
    "จอมทอง": (13.6663, 100.4538),
    "ดอนเมือง": (13.9174, 100.5976),
    "ดินแดง": (13.7664, 100.5586),
    "ทวีวัฒนา": (13.7731, 100.3655),
    "ทุ่งครุ": (13.6285, 100.5040),
    "ธนบุรี": (13.7300, 100.4893),
    "บึงกุ่ม": (13.8086, 100.6438),
    "ประเวศ": (13.6924, 100.6726),
    "มีนบุรี": (13.8120, 100.7424),
    "ยานนาวา": (13.6965, 100.5385),
    "หนองแขม": (13.6928, 100.3444),
    "หนองจอก": (13.8567, 100.8384),
    "หลักสี่": (13.8722, 100.5736),
    "ห้วยขวาง": (13.7747, 100.5819),
    "คลองสามวา": (13.8448, 100.7204),
}

st.set_page_config(page_title="Traffy Late Prediction", layout="wide")

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

# District selection with auto-coordinates
st.sidebar.subheader("Location")
district = st.sidebar.selectbox("District", sorted(BANGKOK_DISTRICTS.keys()))
lat, lon = BANGKOK_DISTRICTS[district]
st.sidebar.caption(f"📍 Coordinates: {lat:.4f}, {lon:.4f}")

# Option to manually override coordinates
with st.sidebar.expander("Override Coordinates (Advanced)"):
    custom_coords = st.checkbox("Use custom coordinates")
    if custom_coords:
        lat = st.number_input("Latitude", min_value=13.0, max_value=14.5, value=lat, step=0.01, format="%.4f")
        lon = st.number_input("Longitude", min_value=100.0, max_value=101.0, value=lon, step=0.01, format="%.4f")

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
st.header("Recent Predictions from Training")

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
