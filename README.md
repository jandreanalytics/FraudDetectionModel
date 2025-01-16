# /README.md
# RiskSentry ML

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.1-purple)

## Portfolio Project Overview

RiskSentry ML is a machine learning demonstration project that showcases transaction risk assessment capabilities. Built as a portfolio piece, it implements pattern recognition and risk analysis using Python and modern web technologies.

## Live Demo

[Try the Demo](https://jandreanalytics.github.io/FraudDetectionModel/)

*Note: Initial API request may take up to 60 seconds during cold start. Subsequent requests will be instantaneous.*

## Key Features

### Pattern Recognition
- Location-based anomalies
- Time-based risk factors (night transactions)
- Amount pattern analysis (suspicious ranges)
- Transaction type combinations

### Risk Analysis
- Multi-factor risk scoring
- Behavioral pattern detection
- Composite risk indicators
- Probability-based assessment

### Interactive Interface
- Real-time risk visualization
- Transaction history tracking
- Pattern detection display
- Risk factor breakdown

## Technical Stack

### Backend
- Python 3.8+
- Flask (API Framework)
- scikit-learn (Machine Learning)
- Pandas/NumPy (Data Processing)

### Frontend
- HTML5/CSS3
- JavaScript (ES6+)
- Bootstrap 5.1
- Interactive Charts

### Deployment
- Render Cloud Platform
- RESTful API
- CORS enabled
- Environment Configuration

## Machine Learning Implementation

### Feature Engineering
- 30+ engineered features
- Time-based analysis
- Location risk assessment
- Amount pattern detection
- Transaction type analysis

### Risk Patterns Detected
- Small amount transactions (<$10)
- Suspicious ranges ($900-999)
- Night transactions (00:00-06:00)
- High-risk locations
- ATM withdrawal patterns
- Online transaction anomalies

## Project Structure

```
├── api/
│   └── app.py                # Flask API implementation
├── models/
│   └── train_model.py        # ML model training
├── data/
│   └── generate_data.py      # Synthetic data generation
├── tests/
│   └── test_fraud_detection.py  # Test scenarios
├── index.html               # Interactive frontend
└── README.md
```

## API Usage

### Endpoint
`POST /api/v1/predict`

### Request Format
```json
{
  "transaction_id": "12345",
  "timestamp": "2024-01-01 14:30:00",
  "amount": 999.99,
  "location": "NY",
  "transaction_type": "online",
  "merchant_category": "retail"
}
```

### Response Format
```json
{
  "transaction_id": "12345",
  "fraud_probability": 0.85,
  "is_fraud": true,
  "risk_patterns": {
    "location_anomaly": true,
    "time_anomaly": false,
    "amount_anomaly": true,
    "transaction_type_risk": true
  },
  "risk_score": 4.5
}
```

## Local Development

1. Clone the repository
```bash
git clone https://github.com/yourusername/risksentry-ml.git
cd risksentry-ml
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
python api/app.py
```

4. Open `index.html` in your browser

## Testing

Run the test suite:
```bash
python tests/test_fraud_detection.py
```

## Portfolio Focus

This project demonstrates:
- Machine Learning Implementation
- API Development
- Complex Feature Engineering
- Pattern Recognition
- Interactive Visualization
- Cloud Deployment
- Testing Methodology

## License

MIT License - See LICENSE file for details

## Acknowledgments

Built with:
- scikit-learn for machine learning
- Flask for API development
- Bootstrap for frontend design
- Python data science stack