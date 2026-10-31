import os
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# ============================================================
# 1. LOAD MODEL
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "best_churn_model.pkl"
)

FEATURES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "features.pkl"
)


model = joblib.load(
    MODEL_PATH
)

features = joblib.load(
    FEATURES_PATH
)


# ============================================================
# 2. CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn probability",
    version="1.0.0"
)


# ============================================================
# 3. INPUT DATA MODEL
# ============================================================

class CustomerData(BaseModel):

    recency: float
    frequency: float
    monetary: float
    purchase_days: float
    customer_lifetime_days: float


# ============================================================
# 4. HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API",
        "status": "running"
    }


# ============================================================
# 5. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "loaded"
    }


# ============================================================
# 6. PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(customer: CustomerData):

    # Convert input into DataFrame
    input_data = pd.DataFrame(
        [[
            customer.recency,
            customer.frequency,
            customer.monetary,
            customer.purchase_days,
            customer.customer_lifetime_days
        ]],
        columns=features
    )

    # Prediction
    prediction = model.predict(
        input_data
    )[0]

    # Churn probability
    probability = model.predict_proba(
        input_data
    )[0][1]

    # Risk level
    if probability >= 0.70:

        risk = "High"

    elif probability >= 0.40:

        risk = "Medium"

    else:

        risk = "Low"

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(
            float(probability),
            4
        ),
        "risk_level": risk
    }