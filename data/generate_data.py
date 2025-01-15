import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_user_profile():
    return {
        'avg_amount': abs(np.random.normal(100, 50)),
        'preferred_locations': np.random.choice(['NY', 'LA', 'CH', 'HO', 'SF'], 
                                             size=np.random.randint(1, 3)),
        'active_hours': np.random.choice(['day', 'night', 'both'], p=[0.7, 0.1, 0.2]),
        'transaction_frequency': np.random.normal(5, 2),  # transactions per day
        'risk_level': np.random.choice(['low', 'medium', 'high'], p=[0.8, 0.15, 0.05]),
        'typical_merchants': np.random.randint(3, 10),  # number of regular merchants
        'spending_pattern': np.random.choice(['consistent', 'variable', 'periodic'], p=[0.6, 0.3, 0.1]),
        'income_level': np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2]),
        'account_age_days': np.random.randint(1, 3650),  # Up to 10 years
        'card_type': np.random.choice(['debit', 'credit', 'premium'], p=[0.4, 0.4, 0.2]),
        'merchant_categories': np.random.choice(
            ['retail', 'dining', 'travel', 'online', 'entertainment'],
            size=np.random.randint(2, 5),
            replace=False
        ),
        'international_travel': np.random.choice([True, False], p=[0.1, 0.9])
    }

def generate_normal_transaction(user_id, timestamp, user_profile):
    # Time-based amount adjustments (people spend more during day/weekends)
    hour = timestamp.hour
    is_weekend = timestamp.weekday() >= 5
    time_multiplier = 1.2 if (9 <= hour <= 18) else 0.8
    weekend_multiplier = 1.3 if is_weekend else 1.0
    
    amount = abs(np.random.normal(
        user_profile['avg_amount'] * time_multiplier * weekend_multiplier,
        user_profile['avg_amount'] * 0.2
    ))
    
    # Location selection based on user profile
    location = np.random.choice(user_profile['preferred_locations'])
    
    # Adjust timing based on user profile
    if user_profile['active_hours'] == 'day' and hour < 6:
        timestamp = timestamp.replace(hour=np.random.randint(9, 18))
    elif user_profile['active_hours'] == 'night' and 9 <= hour <= 18:
        timestamp = timestamp.replace(hour=np.random.randint(19, 24))
    
    return {
        'amount': amount,
        'location': location,
        'timestamp': timestamp,
        'user_id': user_id,
        'is_fraud': 0
    }

def generate_fraudulent_pattern(user_profile):
    patterns = [
        # Classic large amount fraud
        lambda: {
            'amount': user_profile['avg_amount'] * np.random.uniform(8, 15),
            'location': np.random.choice(['UK', 'RU', 'BR', 'CN']),
            'is_fraud': 1
        },
        # After hours + location change
        lambda: {
            'amount': user_profile['avg_amount'] * np.random.uniform(3, 6),
            'location': np.random.choice(['UK', 'RU', 'BR', 'CN']),
            'timestamp': datetime.now().replace(hour=np.random.randint(1, 5)),
            'is_fraud': 1
        },
        # Card testing pattern
        lambda: {
            'amount': np.random.uniform(1, 5),
            'location': np.random.choice(['UK', 'RU', 'BR', 'CN']),
            'is_fraud': 1,
            'is_card_testing': True
        },
        # Account takeover
        lambda: {
            'amount': user_profile['avg_amount'] * np.random.uniform(0.8, 1.2),
            'location': np.random.choice(user_profile['preferred_locations']),
            'is_fraud': 1,
            'merchant_count_spike': True
        },
        # Digital goods fraud
        lambda: {
            'amount': float(np.random.randint(1, 10) * 50),
            'location': 'NY',
            'is_fraud': 1,
            'is_digital': True
        },
        # ATM withdrawal pattern
        lambda: {
            'amount': float(np.random.randint(20, 50)) * 20,
            'location': np.random.choice(['UK', 'RU', 'BR', 'CN']),
            'is_fraud': 1,
            'transaction_type': 'atm'
        },
        # E-commerce fraud
        lambda: {
            'amount': np.random.uniform(50, 200),
            'location': 'NY',
            'is_fraud': 1,
            'transaction_type': 'online',
            'merchant_category': 'digital_goods'
        },
        # Gas station fraud
        lambda: {
            'amount': np.random.uniform(20, 60),
            'location': user_profile['preferred_locations'][0],
            'is_fraud': 1,
            'transaction_type': 'gas',
            'is_skimmed': True
        }
    ]
    
    # Weight patterns based on user risk level (adjusted for 8 patterns)
    if user_profile['risk_level'] == 'high':
        weights = [0.2, 0.2, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05]
    elif user_profile['risk_level'] == 'medium':
        weights = [0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1]
    else:
        weights = [0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1]
    
    return np.random.choice(patterns, p=weights)()

def generate_transactions(n_samples=50000):  # Increased sample size
    np.random.seed(42)
    transactions = []
    base = datetime.now() - timedelta(days=90)  # Extended to 90 days
    
    # Generate more diverse user profiles
    n_users = n_samples // 200  # More transactions per user
    users = {i: generate_user_profile() for i in range(1000, 1000 + n_users)}
    
    # Calculate fraud probability based on user profile
    def get_fraud_probability(profile):
        base_prob = {'low': 0.001, 'medium': 0.01, 'high': 0.05}[profile['risk_level']]
        
        # Adjust based on card type
        card_multiplier = {
            'debit': 1.2,
            'credit': 1.0,
            'premium': 0.8
        }[profile['card_type']]
        
        # Adjust based on account age
        if profile['account_age_days'] < 30:
            base_prob *= 2
        elif profile['account_age_days'] > 365:
            base_prob *= 0.7
            
        return base_prob * card_multiplier
    
    # Generate seasonal patterns
    holiday_periods = [
        (base + timedelta(days=x), base + timedelta(days=x+5))
        for x in [20, 45, 70]  # Three holiday periods
    ]

    def is_holiday_period(timestamp):
        return any(start <= timestamp <= end for start, end in holiday_periods)

    # Add debug logging for night transactions
    night_transactions = []

    # Generate transactions with seasonal patterns
    for user_id, profile in users.items():
        fraud_prob = get_fraud_probability(profile)
        
        # Calculate number of transactions for this user
        n_user_transactions = int(np.random.normal(
            profile['transaction_frequency'] * 30,  # 30 days
            profile['transaction_frequency'] * 5
        ))
        
        # Generate timestamps for this user
        user_timestamps = sorted([
            base + timedelta(
                days=np.random.uniform(0, 30),
                hours=np.random.uniform(0, 24)
            )
            for _ in range(n_user_transactions)
        ])
        
        # Generate transactions for this user
        for timestamp in user_timestamps:
            # Debug logging for night hours
            hour = timestamp.hour
            if 0 <= hour <= 5:
                night_transactions.append({
                    'hour': hour,
                    'user_id': user_id,
                    'risk_level': profile['risk_level']
                })

            transaction = {
                'transaction_id': len(transactions) + 1,
                **generate_normal_transaction(user_id, timestamp, profile)
            }
            
            # Introduce fraud patterns
            if np.random.random() < fraud_prob:
                fraud_pattern = generate_fraudulent_pattern(profile)
                transaction.update(fraud_pattern)
                
                # Card testing pattern generates multiple small transactions
                if fraud_pattern.get('is_card_testing'):
                    for j in range(np.random.randint(5, 15)):
                        fraud_follow_up = {
                            'transaction_id': len(transactions) + 2 + j,
                            'amount': np.random.uniform(1, 5),
                            'location': transaction['location'],
                            'timestamp': timestamp + timedelta(minutes=j*2),
                            'user_id': user_id,
                            'is_fraud': 1
                        }
                        transactions.append(fraud_follow_up)
            
            # Add holiday spending patterns
            if is_holiday_period(timestamp):
                transaction['amount'] *= np.random.uniform(1.5, 3.0)  # Higher spending during holidays
            
            # Add merchant category
            transaction['merchant_category'] = np.random.choice(profile['merchant_categories'])
            
            # Add transaction type
            transaction['transaction_type'] = np.random.choice(
                ['pos', 'online', 'atm', 'recurring'],
                p=[0.6, 0.3, 0.05, 0.05]
            )
            
            transactions.append(transaction)
    
    # Convert to DataFrame and sort by timestamp
    df = pd.DataFrame(transactions)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Save to CSV
    df.to_csv('transactions.csv', index=False)
    print(f"Generated {len(df)} transactions for {len(users)} users")
    print(f"Fraud rate: {(df['is_fraud'].sum() / len(df)) * 100:.2f}%")
    
    # Print additional statistics
    print("\nTransaction Statistics:")
    print(f"Total Users: {len(users)}")
    print(f"Average Transactions per User: {len(df) / len(users):.1f}")
    print(f"Transaction Types Distribution:\n{df['transaction_type'].value_counts(normalize=True)}")
    print(f"Merchant Categories Distribution:\n{df['merchant_category'].value_counts(normalize=True)}")
    print(f"Location Distribution:\n{df['location'].value_counts(normalize=True)}")

    # Print night transaction statistics
    print("\nNight Transaction Analysis:")
    print(f"Total night transactions: {len(night_transactions)}")
    print("Night hours distribution:")
    hours_dist = pd.DataFrame(night_transactions)['hour'].value_counts().sort_index()
    print(hours_dist)
    
    return df

if __name__ == "__main__":
    generate_transactions()
