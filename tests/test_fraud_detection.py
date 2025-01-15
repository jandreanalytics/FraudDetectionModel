import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def generate_test_cases():
    base_time = datetime.now()
    test_cases = [
        # Normal POS Transactions (60% of normal transactions)
        {
            "scenario": "Regular retail POS",
            "transaction_id": 1,
            "timestamp": (base_time.replace(hour=14)).strftime("%Y-%m-%d %H:%M:%S"),
            "amount": 75.50,
            "location": "SF",  # Most common location
            "user_id": 1234,
            "merchant_category": "retail",  # Most common category
            "transaction_type": "pos",
            "expected_fraud": False
        },
        {
            "scenario": "Dining POS transaction",
            "transaction_id": 2,
            "timestamp": (base_time.replace(hour=19)).strftime("%Y-%m-%d %H:%M:%S"),
            "amount": 120.00,
            "location": "CH",  # Second most common location
            "user_id": 1234,
            "merchant_category": "dining",
            "transaction_type": "pos",
            "expected_fraud": False
        },
        
        # Normal Online Transactions (30% of normal transactions)
        {
            "scenario": "Regular online purchase",
            "transaction_id": 3,
            "timestamp": (base_time.replace(hour=15)).strftime("%Y-%m-%d %H:%M:%S"),
            "amount": 199.99,
            "location": "NY",
            "user_id": 1234,
            "merchant_category": "online",
            "transaction_type": "online",
            "expected_fraud": False
        },
        
        # Suspicious Patterns
        # Card Testing (multiple small online transactions)
        *[{
            "scenario": f"Card testing sequence {i+1}/3",
            "transaction_id": 10 + i,
            "timestamp": (base_time + timedelta(minutes=i*2)).strftime("%Y-%m-%d %H:%M:%S"),
            "amount": np.random.uniform(1, 5),
            "location": "RU",  # Rare location (0.18%)
            "user_id": 1234,
            "merchant_category": "online",
            "transaction_type": "online",
            "expected_fraud": True
        } for i in range(3)],
        
        # Suspicious ATM (5% of transactions are ATM)
        {
            "scenario": "Suspicious ATM withdrawal",
            "transaction_id": 20,
            "timestamp": (base_time.replace(hour=2)).strftime("%Y-%m-%d %H:%M:%S"),
            "amount": 400.00,
            "location": "BR",  # Rare location (0.21%)
            "user_id": 1234,
            "merchant_category": None,
            "transaction_type": "atm",
            "expected_fraud": True
        },
        
        # Recurring Payment Anomaly (5% of transactions are recurring)
        {
            "scenario": "Abnormal recurring charge",
            "transaction_id": 21,
            "timestamp": (base_time.replace(hour=3)).strftime("%Y-%m-%d %H:%M:%S"),
            "amount": 299.99,
            "location": "UK",  # Rare location (0.23%)
            "user_id": 1234,
            "merchant_category": "entertainment",
            "transaction_type": "recurring",
            "expected_fraud": True
        },
        
        # Location Anomaly
        {
            "scenario": "Location anomaly",
            "transaction_id": 22,
            "timestamp": (base_time.replace(hour=23)).strftime("%Y-%m-%d %H:%M:%S"),
            "amount": 850.00,
            "location": "CN",  # Rare location (0.33%)
            "user_id": 1234,
            "merchant_category": "travel",
            "transaction_type": "pos",
            "expected_fraud": True
        }
    ]
    return test_cases

def test_model():
    test_cases = generate_test_cases()
    results = []
    
    for case in test_cases:
        try:
            api_input = case.copy()
            scenario = api_input.pop('scenario')
            expected_fraud = api_input.pop('expected_fraud')
            
            # Ensure timestamp is properly formatted
            if 'timestamp' in api_input:
                try:
                    datetime.strptime(api_input['timestamp'], '%Y-%m-%d %H:%M:%S')
                except ValueError as e:
                    print(f"Invalid timestamp format in scenario '{scenario}': {e}")
                    continue
            
            print(f"\nTesting scenario: {scenario}")
            print(f"Input data: {api_input}")
            
            response = requests.post(
                'http://localhost:5000/api/v1/predict',
                json=api_input
            )
            
            if response.status_code == 200:
                result = response.json()
                results.append({
                    'scenario': scenario,
                    'expected_fraud': expected_fraud,
                    'predicted_fraud': result['is_fraud'],
                    'fraud_probability': result['fraud_probability'],
                    **api_input
                })
            else:
                print(f"Error testing scenario '{scenario}': {response.json().get('error', 'Unknown error')}")
                print(f"Response status code: {response.status_code}")
                print(f"Response content: {response.text}")
        
        except Exception as e:
            print(f"Exception in scenario '{case['scenario']}': {str(e)}")
            print(f"Input data: {api_input}")
    
    # Only create DataFrame if we have results
    if results:
        return pd.DataFrame(results)
    else:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'scenario', 'expected_fraud', 'predicted_fraud', 
            'fraud_probability', 'transaction_id', 'timestamp',
            'amount', 'location', 'user_id', 'merchant_category',
            'transaction_type'
        ])

def analyze_results(results_df):
    if len(results_df) == 0:
        print("No results to analyze - all test cases failed")
        return
        
    # Print detailed results
    print("\nDetailed Test Results:")
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(f"\nScenario: {row['scenario']}")
        print(f"Expected Fraud: {row['expected_fraud']}")
        print(f"Predicted Fraud: {row['predicted_fraud']}")
        print(f"Fraud Probability: {row['fraud_probability']:.2f}")
    
    # Calculate and print metrics
    y_true = results_df['expected_fraud']
    y_pred = results_df['predicted_fraud']
    
    print("\nModel Performance Metrics:")
    print("-" * 80)
    print(classification_report(y_true, y_pred))
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=results_df,
        x='amount',
        y='fraud_probability',
        hue='expected_fraud',
        style='predicted_fraud',
        s=100
    )
    plt.title('Fraud Detection Results')
    plt.savefig('test_results.png')
    plt.close()
    
    # Add pattern-based analysis
    print("\nFraud Detection by Pattern:")
    print("-" * 80)
    pattern_results = results_df.groupby(
        results_df['scenario'].apply(lambda x: x.split()[0])
    ).agg({
        'predicted_fraud': ['mean', 'count'],
        'fraud_probability': ['mean', 'std']
    })
    print(pattern_results)
    
    # Visualize results with enhanced plotting
    plt.figure(figsize=(12, 8))
    
    # Create main scatter plot
    scatter = plt.subplot(2, 1, 1)
    sns.scatterplot(
        data=results_df,
        x='amount',
        y='fraud_probability',
        hue='expected_fraud',
        style='predicted_fraud',
        size='amount',
        sizes=(50, 400),
        alpha=0.6
    )
    plt.title('Fraud Detection Results by Amount and Probability')
    
    # Add pattern distribution
    pattern_dist = plt.subplot(2, 1, 2)
    sns.boxplot(
        data=results_df,
        x='scenario',
        y='fraud_probability',
        hue='expected_fraud'
    )
    plt.xticks(rotation=45, ha='right')
    plt.title('Fraud Probability Distribution by Scenario')
    
    plt.tight_layout()
    plt.savefig('test_results.png', bbox_inches='tight')
    plt.close()
    
    # Add distribution analysis
    print("\nTest Case Distribution Analysis:")
    print("-" * 80)
    print("\nTransaction Types:")
    print(results_df['transaction_type'].value_counts(normalize=True))
    print("\nMerchant Categories:")
    print(results_df['merchant_category'].value_counts(normalize=True))
    print("\nLocations:")
    print(results_df['location'].value_counts(normalize=True))

if __name__ == "__main__":
    print("Starting Fraud Detection System Tests...")
    results = test_model()
    analyze_results(results)
    print("\nTest results have been saved to 'test_results.png'")
