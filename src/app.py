from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import logging
import os
import time

# ---------------- Create required directories ----------------
os.makedirs("logs", exist_ok=True)

# ---------------- Logging Configuration ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
# -------------------------------------------------------

# ---------------- Load Trained Model -------------------
MODEL_PATH = "models/heart_disease_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise e
# -------------------------------------------------------

# ---------------- FastAPI App --------------------------
app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps Task-8: Model Inference Service",
    version="1.0"
)
# -------------------------------------------------------

# ---------------- Input Schema -------------------------
class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int
# -------------------------------------------------------

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

# ---------------- Health Check -------------------------
@app.get("/")
def home():
    logger.info("Health check endpoint accessed")
    return {"message": "Heart Disease Prediction API is running"}
# -------------------------------------------------------

# ---------------- Prediction Endpoint ------------------
@app.post("/predict")
def predict(data: PatientData):
    global REQUEST_COUNT, TOTAL_PREDICTION_TIME

    start_time = time.time()
    REQUEST_COUNT += 1

    try:
        logger.info(f"Request #{REQUEST_COUNT}: {data.dict()}")

        input_df = pd.DataFrame([data.dict()])

        prediction = int(model.predict(input_df)[0])

        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_df)[0][1])

        duration = time.time() - start_time
        TOTAL_PREDICTION_TIME += duration

        logger.info(
            f"Prediction={prediction}, Probability={probability}, Time={duration:.4f}s"
        )

        return {
            "prediction": prediction,
            "probability": probability
        }

    except Exception as e:
        logger.error("Prediction failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

# -------------------------------------------------------
