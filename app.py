import streamlit as st
from datetime import datetime
from pricecode import CropPricePredictionSystem

st.set_page_config(page_title="Crop Price Predictor", layout="centered")

st.title("🌾 Crop Price Prediction System")
st.markdown("Predict future crop prices using weather data and LSTM models.")

# Initialize model
predictor = CropPricePredictionSystem()
predictor.load_data()
predictor.load_model("crop_price")  # Ensure crop_price_model.h5 and scalers are present

# Input fields
crop = st.selectbox("Select Crop", ["Wheat", "Rice", "Maize", "Soybean", "Gram"])
district = st.text_input("Enter District Name")
date_str = st.date_input("Select Prediction Date", datetime.today())
api_key = st.text_input("Enter OpenWeatherMap API Key", type="password")

# Predict single day
if st.button("Predict Price"):
    try:
        result = predictor.predict_price(crop, district, date_str, api_key)
        st.success(f"Predicted price on {date_str.strftime('%Y-%m-%d')} in {district}: ₹{result['predicted_price']:.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Predict full week
if st.button("Predict Week Ahead"):
    try:
        results = predictor.predict_week_ahead(crop, district, date_str, api_key)
        st.markdown("### 📅 Weekly Forecast")
        for item in results['predictions']:
            st.write(f"{item['date']}: ₹{item['predicted_price']:.2f}")
    except Exception as e:
        st.error(f"Week prediction error: {e}")
