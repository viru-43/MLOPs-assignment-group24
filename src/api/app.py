from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import logging

# -------------------------------------------------
# App & logging setup
# -------------------------------------------------
app = FastAPI(title="Heart Disease Prediction API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart-api")

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
model = joblib.load("models/heart_model.joblib")

# -------------------------------------------------
# Simple monitoring counter
# -------------------------------------------------
request_count = 0

# -------------------------------------------------
# Input schema
# -------------------------------------------------
class PatientData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: int

# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------
@app.post("/predict")
def predict(data: PatientData):
    global request_count
    request_count += 1

    input_dict = {
        "age": data.age,
        "sex": data.sex,
        "cp": data.cp,
        "trestbps": data.trestbps,
        "chol": data.chol,
        "fbs": data.fbs,
        "restecg": data.restecg,
        "thalach": data.thalach,
        "exang": data.exang,
        "oldpeak": data.oldpeak,
        "slope": data.slope,
        "ca": data.ca,
        "thal": data.thal
    }

    logger.info(f"Incoming prediction request #{request_count}: {input_dict}")

    # IMPORTANT: DataFrame input for ColumnTransformer
    X = pd.DataFrame([input_dict])

    prediction = model.predict(X)[0]

    logger.info(f"Prediction result: {int(prediction)}")

    return {
        "heart_disease_prediction": int(prediction)
    }

# -------------------------------------------------
# Metrics endpoint (simple monitoring)
# -------------------------------------------------
@app.get("/metrics")
def metrics():
    return {
        "total_prediction_requests": request_count
    }
