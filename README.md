# Fraud Detection System

Full-stack fraud detection system with ML-powered API and interactive frontend.

## Features
- Real-time transaction analysis
- ML-powered risk assessment
- Pattern recognition
- Interactive demo interface
- RESTful API architecture

## Tech Stack
- Backend: Python, Flask, scikit-learn
- Frontend: HTML5, Bootstrap, JavaScript
- ML: Random Forest Classifier
- Deployment: Docker, Render

## Live Demo
[Link to your deployed demo]

## API Documentation
[Link to API docs or include them here]

## Screenshots
[Add screenshots of the interface]

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
