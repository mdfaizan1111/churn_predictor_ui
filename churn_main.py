from churn_custom_transformers import Winsorizer, TopNCategories, DistributionPreservingImputer
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import Optional

# Load the trained pipeline
model = joblib.load('model_pipeline2.pkl')

# Create the FastAPI app
app = FastAPI()

# Define expected input schema (all fields are optional now)
class InputData(BaseModel):
    # Numerical (Winsorized)
    monthly_spend_inr: Optional[float]
    transactions_last_90d: Optional[float]
    avg_session_duration_min: Optional[float]

    # Numerical (Non-Winsorized)
    support_tickets_last_year: Optional[float]

    # Nominal Categorical
    city: Optional[str]
    preferred_payment_method: Optional[str]
    referral_source: Optional[str]

    # Ordinal Categorical
    tenure_bucket: Optional[str]
    satisfaction_level: Optional[str]
    risk_segment: Optional[str]

    # Derived numerical features
    account_vintage: Optional[float]
    activity_vintage: Optional[float]

@app.post("/predict")
def predict(data: InputData):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # ✅ Convert None → np.nan (so imputer and pipeline can handle missing values)
    df = df.replace({None: np.nan})

    # Make prediction
    prediction = model.predict(df)

    return {"prediction": prediction.tolist()}
