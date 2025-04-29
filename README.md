# Gaussian Anomaly Detector

A lightweight Gaussian-based anomaly detection system for binary classification tasks.  
Automatically selects the best detection threshold based on F1 score.  
Designed by Fatemeh Zahra Safaeipour.

## Installation

Clone the repository:

```
git clone https://github.com/fzarasp/gaussian_anomaly_detector.git
cd gaussian_anomaly_detector
pip install -r requirements.txt
```

## Dataset

This model was developed and tested using the [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset from Kaggle.

Download it using KaggleHub:

```python
import kagglehub
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

```
## Usage

```python
from gaussian_anomaly_detector import GaussianAnomalyDetector

# Create and fit model
model = GaussianAnomalyDetector()
model.fit(X_train, y_train)

# Auto-select threshold
f1_score = model.score(X_val, y_val)

# Predict
y_pred = model.predict(X_test)

# Get log probabilities
log_probs = model.predict_proba(X_test)

# Plot precision-recall curve
model.plot_precision_recall(X_val, y_val)

# Save and load
model.save('detector.pkl')
loaded_model = GaussianAnomalyDetector.load('detector.pkl')
```

## License

MIT License © 2024 Fatemeh Zahra Safaeipour
