from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta  # Add datetime import
import os

app = Flask(__name__)
CORS(app)

# Load model, scaler, and feature names - update paths for production
MODEL_PATH = os.getenv('MODEL_PATH', 'models/fraud_model.pkl')
SCALER_PATH = os.getenv('SCALER_PATH', 'models/scaler.pkl')
FEATURES_PATH = os.getenv('FEATURES_PATH', 'models/feature_names.pkl')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)

def create_features(df):
    try:
        df = df.copy()
        print("\nProcessing Transaction:")
        
        # Time-based features - simplified
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
        df['hour'] = df['timestamp'].dt.hour  # Create hour first
        print(f"Hour value created: {df['hour'].iloc[0]}")
        
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        print(f"Hour extracted: {df['hour'].iloc[0]}")
        
        # Handle merchant category
        df['merchant_category'] = df['merchant_category'].fillna('unknown')
        
        # Amount-based features
        df['amount_cents'] = (df['amount'] * 100 % 100).astype(int)
        df['amount_round'] = (df['amount'] % 10 < 1).astype(int)
        df['amount_999'] = (df['amount'].between(900, 999)).astype(int)
        df['amount_high'] = (df['amount'] > 1000).astype(int)
        df['small_amount'] = (df['amount'] < 10).astype(int)
        df['amount_percentile'] = 0.5  # Default for single transaction
        df['amount_zscore'] = 0  # Default for single transaction
        
        # Transaction type features
        df['is_atm'] = (df['transaction_type'] == 'atm').astype(int)
        df['is_online'] = (df['transaction_type'] == 'online').astype(int)
        df['is_pos'] = (df['transaction_type'] == 'pos').astype(int)
        df['is_recurring'] = (df['transaction_type'] == 'recurring').astype(int)  # Added recurring
        
        # Merchant category features
        df['is_digital_goods'] = (df['merchant_category'] == 'digital_goods').astype(int)
        df['is_gas'] = (df['merchant_category'] == 'gas').astype(int)
        
        # Location risk features
        high_risk_locations = ['UK', 'RU', 'BR', 'CN']
        df['is_high_risk_location'] = df['location'].isin(high_risk_locations).astype(int)
        
        # Suspicious combinations - using hour directly
        df['risky_hour_atm'] = ((df['hour'] < 6) & (df['transaction_type'] == 'atm')).astype(int)
        df['risky_hour_online'] = ((df['hour'] < 6) & (df['transaction_type'] == 'online')).astype(int)
        df['small_online'] = df['small_amount'] & df['is_online']
        
        # Risk score - updated without is_night
        df['risk_score'] = (
            ((df['hour'] < 6).astype(int)) + 
            df['amount_999'].astype(int) * 2 +
            df['amount_high'].astype(int) * 2 +
            df['is_high_risk_location'].astype(int) * 3
        )
        
        # Default values for single transaction
        df['user_avg_amount'] = df['amount']
        df['user_std_amount'] = 0
        df['user_avg_interval'] = 0
        df['user_tx_count'] = 1
        df['user_tx_types'] = 1
        df['user_merchant_types'] = 1
        df['high_velocity'] = 0
        df['location_count'] = 1
        
        # Composite risk indicators
        df['card_testing_risk'] = (df['small_amount'] + df['high_velocity'] + df['is_online']) / 3
        df['location_time_risk'] = (df['is_high_risk_location'] + (df['hour'] < 6).astype(int)) / 2
        df['merchant_risk'] = (df['is_digital_goods'] + df['is_gas']) / 2
        
        # Location dummies
        location_dummies = pd.get_dummies(df['location'], prefix='loc')
        
        # Ensure all location columns exist
        expected_locations = ['NY', 'LA', 'CH', 'HO', 'SF', 'UK', 'RU', 'BR', 'CN']
        for loc in expected_locations:
            col = f'loc_{loc}'
            if col not in location_dummies.columns:
                location_dummies[col] = 0
        
        # Ensure all required columns exist with defaults
        required_columns = {
            'is_weekend': 0,
            'amount_high': 0,
            'small_amount': 0,
            'is_atm': 0,
            'is_online': 0,
            'is_pos': 0,
            'is_recurring': 0,
            'is_digital_goods': 0,
            'is_gas': 0,
            'risky_hour_atm': 0,
            'risky_hour_online': 0,
            'small_online': 0,
            'is_high_risk_location': 0,
            'card_testing_risk': 0,
            'location_time_risk': 0,
            'merchant_risk': 0
        }
        
        for col, default in required_columns.items():
            if col not in df.columns:
                df[col] = default
        
        # Add any missing merchant categories with default values
        if 'merchant_category' in df.columns:
            valid_categories = ['retail', 'dining', 'travel', 'online', 'entertainment']
            df['merchant_category'] = df['merchant_category'].fillna('unknown')
            df['merchant_category'] = df['merchant_category'].where(
                df['merchant_category'].isin(valid_categories), 
                'unknown'
            )
        
        # Ensure transaction type is valid
        if 'transaction_type' in df.columns:
            valid_types = ['pos', 'online', 'atm', 'recurring']
            df['transaction_type'] = df['transaction_type'].fillna('pos')
            df['transaction_type'] = df['transaction_type'].where(
                df['transaction_type'].isin(valid_types), 
                'pos'
            )
        
        # Update composite features after ensuring columns exist
        df['risky_hour_atm'] = ((df['hour'] < 6) & (df['transaction_type'] == 'atm')).astype(int)
        df['risky_hour_online'] = ((df['hour'] < 6) & (df['transaction_type'] == 'online')).astype(int)
        df['small_online'] = df['small_amount'] & df['is_online']
        
        # Combine all features
        features = [
            'amount', 'hour', 'day_of_week', 'is_weekend',
            'amount_cents', 'amount_round', 'amount_999', 'amount_high', 'small_amount',
            'amount_percentile', 'amount_zscore',
            'is_atm', 'is_online', 'is_pos', 'is_recurring',  # Added is_recurring
            'is_digital_goods', 'is_gas',
            'risky_hour_atm', 'risky_hour_online', 'small_online',
            'is_high_risk_location', 'risk_score',
            'user_avg_amount', 'user_std_amount', 'user_avg_interval',
            'user_tx_count', 'user_tx_types', 'user_merchant_types',
            'high_velocity', 'location_count',
            'card_testing_risk', 'location_time_risk', 'merchant_risk'
        ]
        
        return pd.concat([df[features], location_dummies], axis=1)
        
    except Exception as e:
        print(f"\nDetailed error information:")
        print(f"Input data: {df.to_dict('records')}")
        print(f"Error: {str(e)}")
        raise

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"Received request data: {data}")  # Debug logging
        
        if 'timestamp' in data:
            try:
                datetime.strptime(data['timestamp'], '%Y-%m-%d %H:%M:%S')
            except ValueError:
                return jsonify({'error': 'Invalid timestamp format. Use YYYY-MM-DD HH:MM:SS'}), 400
        
        # Create DataFrame and extract hour immediately
        df = pd.DataFrame([data])
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
        df['hour'] = df['timestamp'].dt.hour
        
        # Create features
        X = create_features(df)
        X = X.reindex(columns=feature_names, fill_value=0)
        X_scaled = scaler.transform(X)
        prediction = model.predict_proba(X_scaled)[0][1]
        
        # Risk patterns using hour from original DataFrame
        pattern_risks = {
            'location_anomaly': bool(df['location'].iloc[0] in ['RU', 'BR', 'UK', 'CN']),
            'transaction_type_anomaly': bool(
                df['transaction_type'].iloc[0] in ['atm', 'recurring'] and 
                (0 <= df['hour'].iloc[0] <= 5)  # Use hour range check
            ),
            'merchant_category_anomaly': bool(
                df['merchant_category'].iloc[0] not in 
                ['retail', 'dining', 'travel', 'online', 'entertainment']
            )
        }
        
        # Thresholds
        base_threshold = 0.25
        if pattern_risks['location_anomaly']:
            base_threshold = 0.2
        if sum(pattern_risks.values()) >= 2:
            base_threshold = 0.15
        
        # Fraud level
        fraud_level = "none"
        if prediction > 0.7:
            fraud_level = "high"
        elif prediction > 0.4:
            fraud_level = "medium"
        elif prediction > base_threshold:
            fraud_level = "low"
        
        return jsonify({
            'transaction_id': data.get('transaction_id'),
            'fraud_probability': float(prediction),
            'fraud_level': fraud_level,
            'is_fraud': bool(prediction > base_threshold),
            'risk_patterns': pattern_risks
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
