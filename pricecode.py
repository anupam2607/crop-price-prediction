# Complete Crop Price Prediction System
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle


class CropPricePredictionSystem:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.price_scaler = StandardScaler()
        self.crop_encoder = LabelEncoder()
        self.district_encoder = LabelEncoder()
        self.sequence_length = 30
        self.feature_columns = []
        self.districts = []
        self.crops = []

    def load_data(self):
        """Load and preprocess the training data"""
        print("Loading data...")

        # Load data
        self.crop_data = pd.read_csv(r'C:\Users\Anupa\Desktop\croppricepredictor\augmented_crop_data_2022 (2).csv')
        self.weather_data = pd.read_csv(r'C:\Users\Anupa\Desktop\croppricepredictor\MP_Weather_2022_2023.csv')

        # Convert date columns
        self.crop_data['Date'] = pd.to_datetime(self.crop_data['Date'], format='%d-%m-%Y')
        self.weather_data['Date'] = pd.to_datetime(self.weather_data['Date'])

        # Store unique values
        self.districts = sorted(self.crop_data['District'].unique())
        self.crops = sorted(self.crop_data['Crop_Name'].unique())

        print(f"Data loaded successfully!")
        print(f"Districts: {self.districts}")
        print(f"Crops: {self.crops}")

    def engineer_features(self, df):
        """Create engineered features from weather and price data"""
        df = df.copy()

        # Weather features
        df['Temp_Range'] = df['MaxTemp_C'] - df['MinTemp_C']
        df['GDD'] = np.maximum(0, (df['MaxTemp_C'] + df['MinTemp_C']) / 2 - 10)  # Base temp 10°C
        df['Heat_Stress'] = (df['MaxTemp_C'] > 35).astype(int)
        df['Drought_Risk'] = (df['Rainfall_mm'] < 1).astype(int)
        df['High_Humidity'] = (df['AvgHumidity'] > 80).astype(int)

        # Rolling weather features (7-day window)
        df['Temp_MA7'] = df['MaxTemp_C'].rolling(window=7, min_periods=1).mean()
        df['Rainfall_Sum7'] = df['Rainfall_mm'].rolling(window=7, min_periods=1).sum()
        df['Humidity_MA7'] = df['AvgHumidity'].rolling(window=7, min_periods=1).mean()

        # Time features
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['Day_of_Year'] = df['Date'].dt.dayofyear
        df['Season'] = df['Month'].apply(lambda x:
            1 if x in [12, 1, 2] else  # Winter
            2 if x in [3, 4, 5] else   # Spring
            3 if x in [6, 7, 8] else   # Summer
            4)                         # Autumn

        return df

    def create_sequences(self, data, target_col='Modal_Price'):
        """Create sequences for LSTM training"""
        sequences = []
        targets = []

        for crop in self.crops:
            for district in self.districts:
                crop_district_data = data[
                    (data['Crop_Name'] == crop) &
                    (data['District'] == district)
                ].sort_values('Date')

                if len(crop_district_data) < self.sequence_length:
                    continue

                # Create sequences
                for i in range(self.sequence_length, len(crop_district_data)):
                    seq = crop_district_data.iloc[i-self.sequence_length:i][self.feature_columns].values
                    target = crop_district_data.iloc[i][target_col]
                    sequences.append(seq)
                    targets.append(target)

        return np.array(sequences), np.array(targets)

    def prepare_training_data(self):
        """Prepare data for training"""
        print("Preparing training data...")

        # Merge crop and weather data
        merged_data = pd.merge(self.crop_data, self.weather_data, on='Date', how='left')

        # Fill missing weather data with forward fill
        merged_data = merged_data.fillna(method='ffill')

        # Engineer features
        merged_data = self.engineer_features(merged_data)

        # Encode categorical variables
        merged_data['Crop_Encoded'] = self.crop_encoder.fit_transform(merged_data['Crop_Name'])
        merged_data['District_Encoded'] = self.district_encoder.fit_transform(merged_data['District'])

        # Add price lags and rolling features
        merged_data = merged_data.sort_values(['Crop_Name', 'District', 'Date'])

        for crop in self.crops:
            for district in self.districts:
                mask = (merged_data['Crop_Name'] == crop) & (merged_data['District'] == district)
                if mask.sum() > 0:
                    # Price lags
                    for lag in range(1, 8):
                        merged_data.loc[mask, f'Price_Lag_{lag}'] = merged_data.loc[mask, 'Modal_Price'].shift(lag)

                    # Rolling price features
                    merged_data.loc[mask, 'Price_MA7'] = merged_data.loc[mask, 'Modal_Price'].rolling(7, min_periods=1).mean()
                    merged_data.loc[mask, 'Price_MA14'] = merged_data.loc[mask, 'Modal_Price'].rolling(14, min_periods=1).mean()
                    merged_data.loc[mask, 'Price_Volatility'] = merged_data.loc[mask, 'Modal_Price'].rolling(7, min_periods=1).std()

        # Define feature columns
        self.feature_columns = [
            'MaxTemp_C', 'MinTemp_C', 'AvgHumidity', 'Rainfall_mm',
            'Temp_Range', 'GDD', 'Heat_Stress', 'Drought_Risk', 'High_Humidity',
            'Temp_MA7', 'Rainfall_Sum7', 'Humidity_MA7',
            'Month', 'Quarter', 'Day_of_Year', 'Season',
            'Crop_Encoded', 'District_Encoded'
        ]

        # Add price features
        price_features = [f'Price_Lag_{i}' for i in range(1, 8)] + ['Price_MA7', 'Price_MA14', 'Price_Volatility']
        self.feature_columns.extend(price_features)

        # Remove rows with NaN values
        merged_data = merged_data.dropna()

        # Create sequences
        X, y = self.create_sequences(merged_data)

        # Scale features and target
        X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_scaled = self.price_scaler.fit_transform(y.reshape(-1, 1)).flatten()

        return X_scaled, y_scaled, merged_data

    def build_model(self, input_shape):
        """Build the LSTM model"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def train_model(self):
        """Train the LSTM model"""
        print("Starting model training...")

        # Prepare data
        X, y, merged_data = self.prepare_training_data()

        print(f"Training data shape: {X.shape}, Target shape: {y.shape}")

        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Build model
        self.model = self.build_model((X.shape[1], X.shape[2]))

        # Train model
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )

        # Evaluate model
        y_pred = self.model.predict(X_test)
        y_pred_original = self.price_scaler.inverse_transform(y_pred)
        y_test_original = self.price_scaler.inverse_transform(y_test.reshape(-1, 1))

        mse = mean_squared_error(y_test_original, y_pred_original)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)
        mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100

        print(f"\nModel Performance:")
        print(f"RMSE: ₹{np.sqrt(mse):.2f}")
        print(f"MAE: ₹{mae:.2f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")

        return history

    def fetch_weather_data(self, api_key, district_coords, start_date, end_date):
        """Fetch weather data from OpenWeatherMap API"""
        lat, lon = district_coords
        weather_data = []

        current_date = start_date
        while current_date <= end_date:
            # Convert to timestamp
            timestamp = int(current_date.timestamp())

            # API call for historical data
            url = f"http://api.openweathermap.org/data/2.5/onecall/timemachine"
            params = {
                'lat': lat,
                'lon': lon,
                'dt': timestamp,
                'appid': api_key,
                'units': 'metric'
            }

            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    weather_info = data['current']

                    weather_data.append({
                        'Date': current_date,
                        'MaxTemp_C': weather_info.get('temp', 25),
                        'MinTemp_C': weather_info.get('temp', 25) - 5,  # Approximation
                        'AvgHumidity': weather_info.get('humidity', 60),
                        'Rainfall_mm': weather_info.get('rain', {}).get('1h', 0)
                    })

            except Exception as e:
                print(f"Error fetching weather data: {e}")
                # Use dummy data
                weather_data.append({
                    'Date': current_date,
                    'MaxTemp_C': 30,
                    'MinTemp_C': 20,
                    'AvgHumidity': 60,
                    'Rainfall_mm': 0
                })

            current_date += timedelta(days=1)

        return pd.DataFrame(weather_data)

    def predict_price(self, crop, district, prediction_date, api_key=None):
        """Predict crop price for given parameters"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # District coordinates (approximate)
        district_coords = {
            'Bhopal': (23.2599, 77.4126),
            'Indore': (22.7196, 75.8577),
            'Dewas': (22.9676, 76.0534),
            'Sehore': (23.2007, 77.0853),
            'Hoshangabad': (22.7445, 77.7249),
            'Ratlam': (23.3315, 75.0367),
            'Chhatarpur': (24.9178, 79.5941),
            'Damoh': (23.8315, 79.4421),
            'Pipariya': (22.7736, 78.3559)
        }

        if district not in district_coords:
            raise ValueError(f"District {district} not supported")

        # Get historical data for the last 30 days
        end_date = prediction_date - timedelta(days=1)
        start_date = end_date - timedelta(days=30)

        # Fetch weather data
        if api_key:
            weather_df = self.fetch_weather_data(api_key, district_coords[district], start_date, end_date)
        else:
            # Use dummy weather data for demonstration
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            weather_df = pd.DataFrame({
                'Date': dates,
                'MaxTemp_C': np.random.normal(32, 5, len(dates)),
                'MinTemp_C': np.random.normal(22, 3, len(dates)),
                'AvgHumidity': np.random.normal(65, 10, len(dates)),
                'Rainfall_mm': np.random.exponential(2, len(dates))
            })

        # Get recent price data (using last available prices)
        recent_prices = self.crop_data[
            (self.crop_data['Crop_Name'] == crop) &
            (self.crop_data['District'] == district)
        ].sort_values('Date').tail(30)

        if len(recent_prices) == 0:
            raise ValueError(f"No historical data for {crop} in {district}")

        # Create prediction data
        prediction_data = []
        for i, (_, weather_row) in enumerate(weather_df.iterrows()):
            # Use the last available price or interpolate
            if i < len(recent_prices):
                price = recent_prices.iloc[i]['Modal_Price']
            else:
                price = recent_prices.iloc[-1]['Modal_Price']

            prediction_data.append({
                'Date': weather_row['Date'],
                'Crop_Name': crop,
                'District': district,
                'Modal_Price': price,
                'MaxTemp_C': weather_row['MaxTemp_C'],
                'MinTemp_C': weather_row['MinTemp_C'],
                'AvgHumidity': weather_row['AvgHumidity'],
                'Rainfall_mm': weather_row['Rainfall_mm']
            })

        prediction_df = pd.DataFrame(prediction_data)

        # Engineer features
        prediction_df = self.engineer_features(prediction_df)

        # Encode categorical variables
        prediction_df['Crop_Encoded'] = self.crop_encoder.transform(prediction_df['Crop_Name'])
        prediction_df['District_Encoded'] = self.district_encoder.transform(prediction_df['District'])

        # Add price lags and rolling features
        for lag in range(1, 8):
            prediction_df[f'Price_Lag_{lag}'] = prediction_df['Modal_Price'].shift(lag)

        prediction_df['Price_MA7'] = prediction_df['Modal_Price'].rolling(7, min_periods=1).mean()
        prediction_df['Price_MA14'] = prediction_df['Modal_Price'].rolling(14, min_periods=1).mean()
        prediction_df['Price_Volatility'] = prediction_df['Modal_Price'].rolling(7, min_periods=1).std()

        # Fill NaN values
        prediction_df = prediction_df.fillna(method='ffill').fillna(method='bfill')

        # Prepare sequence
        sequence = prediction_df[self.feature_columns].values
        if len(sequence) < self.sequence_length:
            # Pad with last row if needed
            while len(sequence) < self.sequence_length:
                sequence = np.vstack([sequence, sequence[-1]])

        sequence = sequence[-self.sequence_length:]
        sequence_scaled = self.scaler.transform(sequence.reshape(1, -1)).reshape(1, self.sequence_length, -1)

        # Make prediction
        prediction_scaled = self.model.predict(sequence_scaled)
        prediction = self.price_scaler.inverse_transform(prediction_scaled)[0][0]

        return {
            'crop': crop,
            'district': district,
            'prediction_date': prediction_date.strftime('%Y-%m-%d'),
            'predicted_price': round(prediction, 2),
            'current_price': round(recent_prices.iloc[-1]['Modal_Price'], 2),
            'price_change': round(prediction - recent_prices.iloc[-1]['Modal_Price'], 2),
            'confidence': 'Medium'  # Placeholder
        }

    def predict_week_ahead(self, crop, district, start_date, api_key=None):
        """Predict prices for the next 7 days"""
        predictions = []

        for i in range(7):
            pred_date = start_date + timedelta(days=i)
            try:
                prediction = self.predict_price(crop, district, pred_date, api_key)
                predictions.append(prediction)
            except Exception as e:
                print(f"Error predicting for {pred_date}: {e}")
                continue

        return predictions

    def save_model(self, filepath):
        """Save the trained model and scalers"""
        self.model.save(f"{filepath}_model.h5")

        with open(f"{filepath}_scalers.pkl", 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'price_scaler': self.price_scaler,
                'crop_encoder': self.crop_encoder,
                'district_encoder': self.district_encoder,
                'feature_columns': self.feature_columns,
                'districts': self.districts,
                'crops': self.crops
            }, f)

    def load_model(self, filepath):
        """Load the trained model and scalers"""
        self.model = tf.keras.models.load_model(f"{filepath}_model.h5")

        with open(f"{filepath}_scalers.pkl", 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.price_scaler = data['price_scaler']
            self.crop_encoder = data['crop_encoder']
            self.district_encoder = data['district_encoder']
            self.feature_columns = data['feature_columns']
            self.districts = data['districts']
            self.crops = data['crops']

# Initialize and train the system
print("Initializing Crop Price Prediction System...")
predictor = CropPricePredictionSystem()
predictor.load_data()

# Train the model
print("Training the model...")
history = predictor.train_model()

# Save the model
predictor.model.save('crop_price_model.h5')
print("Model saved successfully!")

# Now let's create the prediction functions with API integration
def fetch_weather_forecast(api_key, lat, lon, days=7):
    """Fetch weather forecast from OpenWeatherMap API"""
    url = f"http://api.openweathermap.org/data/2.5/forecast"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key,
        'units': 'metric',
        'cnt': days * 8  # 8 forecasts per day (3-hour intervals)
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            daily_forecasts = []

            # Group forecasts by date
            current_date = None
            daily_data = {'temps': [], 'humidity': [], 'rain': []}

            for forecast in data['list']:
                forecast_date = datetime.fromtimestamp(forecast['dt']).date()

                if current_date is None:
                    current_date = forecast_date

                if forecast_date != current_date:
                    # Process previous day's data
                    if daily_data['temps']:
                        daily_forecasts.append({
                            'date': current_date,
                            'max_temp': max(daily_data['temps']),
                            'min_temp': min(daily_data['temps']),
                            'avg_humidity': sum(daily_data['humidity']) / len(daily_data['humidity']),
                            'rainfall': sum(daily_data['rain'])
                        })

                    # Reset for new day
                    current_date = forecast_date
                    daily_data = {'temps': [], 'humidity': [], 'rain': []}

                # Add current forecast data
                daily_data['temps'].append(forecast['main']['temp'])
                daily_data['humidity'].append(forecast['main']['humidity'])
                daily_data['rain'].append(forecast.get('rain', {}).get('3h', 0))

            # Process last day
            if daily_data['temps']:
                daily_forecasts.append({
                    'date': current_date,
                    'max_temp': max(daily_data['temps']),

                    'min_temp': min(daily_data['temps']),
                    'avg_humidity': sum(daily_data['humidity']) / len(daily_data['humidity']),
                    'rainfall': sum(daily_data['rain'])
                })

            return daily_forecasts
        else:
            print(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching weather forecast: {e}")
        return None

def predict_crop_price(crop, district, prediction_date, api_key="d07723cd885b4c70accc897d9e992a56"):
    """Predict crop price using the trained model"""

    # District coordinates for Madhya Pradesh
    district_coords = {
        'Bhopal': (23.2599, 77.4126),
        'Indore': (22.7196, 75.8577),
        'Dewas': (22.9676, 76.0534),
        'Sehore': (23.2007, 77.0853),
        'Hoshangabad': (22.7445, 77.7249),
        'Ratlam': (23.3315, 75.0367),
        'Chhatarpur': (24.9178, 79.5941),
        'Damoh': (23.8315, 79.4421),
        'Pipariya': (22.7736, 78.3559),
        'Gwalior': (26.2124, 78.1772),
    }

    if district not in district_coords:
        return {"error": f"District '{district}' not supported"}

    if crop not in predictor.crops:
        return {"error": f"Crop '{crop}' not supported"}

    # Get current weather data
    lat, lon = district_coords[district]

    # Fetch weather forecast for the prediction date (or closest available)
    weather_forecast = fetch_weather_forecast(api_key, lat, lon, days=1) # Fetch forecast for 1 day

    if not weather_forecast:
         return {"error": f"Could not fetch weather forecast for {district} on {prediction_date.strftime('%Y-%m-%d')}"}

     # Get last known price for the crop in the district
    last_price_data = predictor.crop_data[
        (predictor.crop_data['Crop_Name'] == crop) &
        (predictor.crop_data['District'] == district)
    ].sort_values('Date')

    if len(last_price_data) == 0:
        return {"error": f"No historical data available for {crop} in {district}"}

    last_price = last_price_data.iloc[-1]['Modal_Price']

    # Simple prediction based on seasonal patterns and weather
    # This is a simplified version - in production, use the full LSTM model

    # Seasonal adjustment
    month = prediction_date.month
    seasonal_factor = 1.0

    if crop == 'Wheat':
        if month in [3, 4, 5]:  # Harvest season
            seasonal_factor = 0.95
        elif month in [9, 10, 11]:  # Sowing season
            seasonal_factor = 1.05
    elif crop == 'Bengal Gram':
        if month in [4, 5, 6]:  # Harvest season
            seasonal_factor = 0.92
        elif month in [10, 11, 12]:  # Sowing season
            seasonal_factor = 1.08
    elif crop == 'Garlic':
        if month in [3, 4, 5]:  # Harvest season
            seasonal_factor = 0.90
        elif month in [7, 8, 9]:  # Storage period
            seasonal_factor = 1.10

    # Weather impact
    weather_factor = 1.0
    if weather_forecast[0]['max_temp'] > 40:  # Extreme heat
        weather_factor *= 0.5
    if weather_forecast[0]['rainfall'] > 20:  # Heavy rain
        weather_factor *= 0.5

    # Calculate predicted price
    predicted_price = last_price * seasonal_factor * weather_factor

    # Add some random variation (±2%)
    variation = np.random.normal(0, 0.02)
    predicted_price *= (1 + variation)

    return {
        'crop': crop,
        'district': district,
        'prediction_date': prediction_date.strftime('%Y-%m-%d'),
        'predicted_price': round(predicted_price, 2),
        'last_known_price': round(last_price, 2),
        'price_change': round(predicted_price - last_price, 2),
        'price_change_percent': round(((predicted_price - last_price) / last_price) * 100, 2),
        'weather_conditions': weather_forecast[0],
        'confidence': 'Medium',
        'factors': {
            'seasonal_factor': seasonal_factor,
            'weather_factor': weather_factor
        }
    }

def predict_week_ahead(crop, district, start_date, api_key="d07723cd885b4c70accc897d9e992a56"):
    """Predict crop prices for the next 7 days"""
    predictions = []

    for i in range(7):
        pred_date = start_date + timedelta(days=i)
        prediction = predict_crop_price(crop, district, pred_date, api_key)
        predictions.append(prediction)

    return predictions

# Test the prediction system
print("=== Testing Crop Price Prediction System ===")

print("=== Testing Crop Price Prediction System ===")

# Read crop name, district, and date from user
user_crop = input("Enter crop name (e.g., Wheat): ")
user_district = input("Enter district (e.g., Gwalior): ")
user_date_str = input("Enter prediction date (YYYY-MM-DD): ")

try:
    user_date = datetime.strptime(user_date_str, "%Y-%m-%d")
except ValueError:
    print("Invalid date format. Please use YYYY-MM-DD.")
    exit(1)

# Test single day prediction
prediction = predict_crop_price(user_crop, user_district, user_date)
print(f"\nSingle Day Prediction:")
if 'error' in prediction:
    print(f"Error: {prediction['error']}")
else:
    print(f"Crop: {prediction['crop']}")
    print(f"District: {prediction['district']}")
    print(f"Date: {prediction['prediction_date']}")
    print(f"Predicted Price: ₹{prediction['predicted_price']}")
    print(f"Price Change: ₹{prediction['price_change']} ({prediction['price_change_percent']}%)")

# Test weekly prediction
print(f"\n=== 7-Day Price Forecast ===")
weekly_predictions = predict_week_ahead(user_crop, user_district, user_date)

for i, pred in enumerate(weekly_predictions):
    if 'error' in pred:
        print(f"Day {i+1}: Error - {pred['error']}")
    else:
        print(f"Day {i+1}: {pred['prediction_date']} - ₹{pred['predicted_price']} ({pred['price_change_percent']:+.1f}%)")
