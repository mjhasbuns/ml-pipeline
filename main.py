"""Serve the scikit-learn churn pipeline with FastAPI."""

from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

# Loaded once when the app starts, not on every request
model = joblib.load("model.joblib")

app = FastAPI(title="Telco Churn Predictor")

YesNo = Literal["Yes", "No"]
InternetExtra = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetExtra
    OnlineBackup: InternetExtra
    DeviceProtection: InternetExtra
    TechSupport: InternetExtra
    StreamingTV: InternetExtra
    StreamingMovies: InternetExtra
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float
    TotalCharges: float

    # Pre-fills "Try it out" in /docs with a realistic customer
    model_config = ConfigDict(json_schema_extra={"example": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 70.70, "TotalCharges": 151.65,
    }})


@app.post("/predict")
def predict(customer: Customer):
    df = pd.DataFrame([customer.model_dump()])
    prediction = model.predict(df)[0]
    churn_probability = model.predict_proba(df)[0][list(model.classes_).index("Yes")]
    return {"churn": prediction, "churn_probability": round(float(churn_probability), 3)}
