from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import time
import logging
import os

# ================== Logging Setup ==================
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/api.log")
    ]
)

logger = logging.getLogger(__name__)
# ===================================================

# ================== Monitoring Counters ==================
REQUEST_COUNT = 0
TOTAL_PREDICTION_TIME = 0.0
# =========================================================

# ================== Load Model ==================
try:
    model = joblib.load("models/heart_disease_model.pkl")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error("Failed to load model", exc_info=True)
    raise e
# =================================================

app = FastAPI(title="Heart Disease Prediction API")

# ================== Input Schema ==================
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
# ==================================================

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}

@app.post("/predict")
def predict(data: PatientData):
    global REQUEST_COUNT, TOTAL_PREDICTION_TIME

    try:
        start_time = time.time()

        input_df = pd.DataFrame([data.dict()])
        prediction = int(model.predict(input_df)[0])

        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = float(model.predict_proba(input_df)[0][1])

        elapsed_time = time.time() - start_time

        REQUEST_COUNT += 1
        TOTAL_PREDICTION_TIME += elapsed_time

        logger.info(
            f"Request #{REQUEST_COUNT} | "
            f"Time={elapsed_time:.4f}s | "
            f"Prediction={prediction} | "
            f"Confidence={confidence}"
        )

        # ✅ TEST EXPECTS "confidence"
        return {
            "prediction": prediction,
            "confidence": confidence
        }

    except Exception:
        logger.error("Prediction failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
