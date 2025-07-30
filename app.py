from flask import Flask, request, jsonify
from datetime import datetime
import numpy as np
import pickle
import tensorflow as tf

# Import your class from pricecode.py
from pricecode import CropPricePredictionSystem

app = Flask(__name__)

# Load model and scalers
predictor = CropPricePredictionSystem()
predictor.load_data()
predictor.load_model('crop_price')  # Assumes crop_price_model.h5 and crop_price_scalers.pkl exist

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    crop = data.get('crop')
    district = data.get('district')
    date_str = data.get('date')
    api_key = data.get('api_key', None)

    try:
        prediction_date = datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD.'}), 400

    try:
        result = predictor.predict_price(crop, district, prediction_date, api_key)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    return jsonify(result)

@app.route('/predict_week', methods=['POST'])
def predict_week():
    data = request.get_json(force=True)
    crop = data.get('crop')
    district = data.get('district')
    date_str = data.get('date')
    api_key = data.get('api_key', None)

    try:
        start_date = datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD.'}), 400

    try:
        results = predictor.predict_week_ahead(crop, district, start_date, api_key)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)