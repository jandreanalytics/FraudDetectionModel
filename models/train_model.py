import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class FraudDetectionModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def create_features(self, df):
        df = df.copy()
        
        # Time-based features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_night'] = df['hour'].between(0, 5).astype(int)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Amount-based features
        df['amount_cents'] = (df['amount'] * 100 % 100).astype(int)
        df['amount_round'] = (df['amount'] % 10 < 1).astype(int)
        df['amount_999'] = (df['amount'].between(900, 999)).astype(int)
        df['amount_high'] = (df['amount'] > 1000).astype(int)
        df['small_amount'] = (df['amount'] < 10).astype(int)  # Card testing indicator
        
        # Transaction type features
        df['is_atm'] = (df['transaction_type'] == 'atm').astype(int)
        df['is_online'] = (df['transaction_type'] == 'online').astype(int)
        df['is_pos'] = (df['transaction_type'] == 'pos').astype(int)
        df['is_recurring'] = (df['transaction_type'] == 'recurring').astype(int)  # Added recurring
        
        # Merchant category features
        df['is_digital_goods'] = (df['merchant_category'] == 'digital_goods').astype(int)
        df['is_gas'] = (df['merchant_category'] == 'gas').astype(int)
        
        # Suspicious combinations
        df['night_atm'] = df['is_night'] & df['is_atm']
        df['night_online'] = df['is_night'] & df['is_online']
        df['small_online'] = df['small_amount'] & df['is_online']
        
        # Location risk features
        high_risk_locations = ['UK', 'RU', 'BR', 'CN']
        df['is_high_risk_location'] = df['location'].isin(high_risk_locations).astype(int)
        
        # Combine risk factors
        df['risk_score'] = (
            df['is_night'].astype(int) + 
            df['amount_999'].astype(int) * 2 +
            df['amount_high'].astype(int) * 2 +
            df['is_high_risk_location'].astype(int) * 3
        )
        
        # Add amount percentile features
        df['amount_percentile'] = df['amount'].rank(pct=True)
        df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
        
        # User behavior features
        user_stats = df.groupby('user_id').agg({
            'amount': ['mean', 'std', 'count'],
            'timestamp': lambda x: x.diff().dt.total_seconds().mean(),
            'transaction_type': 'nunique',
            'merchant_category': 'nunique'
        }).fillna(0)
        
        # Flatten column names
        user_stats.columns = [
            'user_avg_amount', 'user_std_amount', 'user_tx_count',
            'user_avg_interval', 'user_tx_types', 'user_merchant_types'
        ]
        df = df.merge(user_stats, left_on='user_id', right_index=True, how='left')
        
        # Transaction velocity features
        df['tx_velocity'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
        df['high_velocity'] = (df['tx_velocity'] < 300).astype(int)  # Less than 5 minutes
        
        # Location features
        df['location_count'] = df.groupby('user_id')['location'].transform('nunique')
        location_dummies = pd.get_dummies(df['location'], prefix='loc')
        
        # Combine features
        features = [
            'amount', 'hour', 'day_of_week', 'is_night',
            'amount_cents', 'amount_round', 'amount_999',
            'user_avg_amount', 'user_std_amount', 'user_avg_interval',
            'location_count', 'amount_percentile', 'amount_zscore',
            'is_weekend', 'amount_high', 
            'is_high_risk_location', 'risk_score',
            'small_amount', 'is_atm', 'is_online', 'is_pos', 'is_recurring',  # Added is_recurring
            'is_digital_goods', 'is_gas', 'night_atm', 'night_online',
            'small_online', 'user_tx_count', 'user_tx_types',
            'user_merchant_types', 'high_velocity'
        ]
        
        return pd.concat([df[features], location_dummies], axis=1)

    def train(self, data_path):
        df = pd.read_csv(data_path)
        
        # Validate night transactions before feature creation
        df['temp_hour'] = pd.to_datetime(df['timestamp']).dt.hour
        night_stats = {
            'total_transactions': len(df),
            'night_transactions': len(df[df['temp_hour'].between(0, 5)]),
            'night_frauds': len(df[(df['temp_hour'].between(0, 5)) & (df['is_fraud'] == 1)])
        }
        print("\nNight Transaction Statistics:")
        print(f"Total Transactions: {night_stats['total_transactions']}")
        print(f"Night Transactions: {night_stats['night_transactions']} ({night_stats['night_transactions']/night_stats['total_transactions']*100:.2f}%)")
        print(f"Night Frauds: {night_stats['night_frauds']} ({night_stats['night_frauds']/night_stats['night_transactions']*100:.2f}% of night transactions)")
        
        X = self.create_features(df)
        y = df['is_fraud']
        
        # Adjusted feature weights based on real transaction distributions
        feature_weights = {
            # Location-based weights (rare locations have higher weight)
            'is_high_risk_location': 8,    # Increased due to <1% occurrence
            'loc_RU': 5,
            'loc_BR': 5,
            'loc_UK': 5,
            'loc_CN': 5,
            
            # Transaction type weights (based on distribution)
            'is_pos': 1,      # Most common (60%)
            'is_online': 2,   # Common (30%)
            'is_atm': 4,      # Rare (5%)
            'is_recurring': 4, # Rare (5%)
            
            # Merchant category weights (all roughly equal distribution ~20%)
            'is_retail': 1,
            'is_dining': 1,
            'is_travel': 1,
            'is_digital_goods': 1,
            'is_entertainment': 1,
            
            # Suspicious combinations
            'night_atm': 6,
            'night_online': 4,
            'small_online': 4,
            'high_velocity': 3
        }
        
        # Apply feature weights
        for feature, weight in feature_weights.items():
            if feature in X.columns:
                X[feature] = X[feature] * weight
        
        # Composite risk indicators based on real patterns
        X['location_risk'] = (
            X['is_high_risk_location'] * 
            (1 + X['is_night'] + X['high_velocity'])
        )
        
        X['transaction_type_risk'] = (
            (X['is_atm'] + X['is_recurring']) * 
            (1 + X['is_night'] + X['amount_high'])
        )
        
        # Update model parameters based on real distributions
        self.model = RandomForestClassifier(
            n_estimators=1000,
            max_depth=20,
            min_samples_leaf=1,
            min_samples_split=2,
            class_weight={
                0: 1,
                1: 20  # Increased weight for fraud cases due to natural imbalance
            },
            random_state=42
        )
        
        # Make sure to save the final column names after all feature engineering
        self.feature_names = X.columns.tolist()
        print(f"\nTotal features: {len(self.feature_names)}")
        print("Sample features:", self.feature_names[:10])
        
        # Stratified split to maintain fraud ratio
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            stratify=y,
            random_state=42
        )
        
        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate and print feature importances
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,  # Use saved feature names
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        # Save all artifacts
        print("\nSaving model artifacts...")
        joblib.dump(self.model, 'models/fraud_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.feature_names, 'models/feature_names.pkl')
        print(f"Saved {len(self.feature_names)} features to feature_names.pkl")

if __name__ == "__main__":
    model = FraudDetectionModel()
    model.train('data/transactions.csv')
