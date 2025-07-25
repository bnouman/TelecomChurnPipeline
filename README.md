# Telco Customer Churn Prediction

This repository provides a complete machine learning pipeline for predicting customer churn using the Telco Customer Churn dataset. The pipeline is implemented in Python and Jupyter Notebook.

---

## Dataset

We use the **Telco Customer Churn** dataset from IBM Sample Data. It contains customer demographics, account information, service details, and whether the customer has churned or not.

---

## Project Features

- Load and clean real-world Telco data
- Handle missing and incorrect data types
- Encode categorical variables
- Scale numeric features
- Train classification models:
  - Logistic Regression
  - Random Forest Classifier
- Evaluate with:
  - Classification Report
  - Confusion Matrix
  - ROC-AUC Score & Curve
- Save the trained model pipeline with `joblib`

---

## Installation

```bash
git clone https://github.com/bnouman/TelecomChurnPipeline
cd TelecomChurnPipeline
pip install -r requirements.txt
```

---

## Usage

To run the full pipeline:

```bash
python pipeline_for_predicting_customer_churn.py
```

To run in notebook mode:

```bash
jupyter notebook pipeline_for_predicting_customer_churn.ipynb
```

---

## Input Features

- Categorical: gender, SeniorCitizen, Partner, Dependents, PhoneService, InternetService, Contract, etc.
- Numeric: tenure, MonthlyCharges, TotalCharges

Target label: `Churn` (Yes/No)

---

## Output

- Trained model file: `best_churn_pipeline.joblib`
- Evaluation metrics printed in notebook/terminal
- Visualization: ROC curves, Confusion Matrix

---

## Sample Prediction

```python
from joblib import load
model = load("best_churn_pipeline.joblib")
model.predict(X_new)
```

---

## Acknowledgements

Dataset: IBM Sample Telco Churn Dataset  
Inspired by customer retention problems in the telecom sector.

---

## Author

Nouman Bashir
noumanbashir923@gmail.com
www.linkedin.com/in/nouman-bashir
