# Financial Fraud Detection System

A real-time fraud detection system using machine learning to identify suspicious financial transactions.

## Overview

This project implements a machine learning-based fraud detection system that:

- Processes financial transaction data
- Uses supervised learning to identify fraudulent patterns
- Provides real-time fraud detection via REST API
- Includes a web dashboard for visualization

## Tech Stack

- Python 3.8+
- Scikit-Learn for ML models
- Flask for API
- Pandas for data processing
- Plotly/Dash for visualization
- Heroku for deployment

## Installation

```bash
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system
pip install -r requirements.txt
```

## Project Structure

```
.
├── data/               # Dataset storage
├── models/            # ML models and preprocessing
├── api/              # Flask API implementation
├── web/              # Dashboard frontend
└── requirements.txt  # Python dependencies
```

## Usage

1. Train the model:

```bash
python models/train_model.py
```

2. Start the API server:

```bash
python api/app.py
```

3. Access the dashboard:

```bash
python web/dashboard.py
```

## API Endpoints

- POST `/api/v1/predict`
  - Input: JSON transaction data
  - Output: Fraud probability score

## Dataset

The included dataset (`data/transactions.csv`) contains synthetic transaction data with the following features:

- transaction_id: Unique identifier
- timestamp: Transaction time (YYYY-MM-DD HH:MM:SS)
- amount: Transaction amount in USD
- location: Transaction location
- user_id: Customer identifier
- is_fraud: Binary flag (1 = fraudulent, 0 = legitimate)

## Model Details

- Algorithm: Random Forest Classifier
- Features: Amount, time patterns, location changes
- Metrics: Precision, Recall, F1-Score

## License

MIT License
