"""
FastAPI application serving the heart disease prediction model.
Exposes a /predict endpoint that accepts patient data and returns
a prediction (0/1) plus a confidence score.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import pandas as pd
import os
import logging
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import time

app = FastAPI(title="Heart Disease Prediction API")

# Configure logging: writes to both a file and the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api_requests.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("heart-disease-api")

# Prometheus metrics: these are counters/timers that accumulate over
# the life of the running process
REQUEST_COUNT = Counter("prediction_requests_total", "Total number of prediction requests")
PREDICTION_COUNT = Counter("predictions_by_class", "Predictions grouped by outcome", ["diagnosis"])
REQUEST_LATENCY = Histogram("prediction_request_latency_seconds", "Time taken to process a prediction request")

# Load model artifacts once, at startup (not on every request — that would be slow)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
with open(os.path.join(MODEL_DIR, "model_config.json")) as f:
    config = json.load(f)


class PatientData(BaseModel):
    """
    Defines exactly what fields a request must contain, and their types.
    FastAPI automatically validates incoming requests against this —
    reject anything malformed before it ever reaches the model.
    """
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
    ca: float
    thal: float


@app.get("/")
def root():
    """Simple health check — confirms the API is alive and responding."""
    return {"status": "ok", "message": "Heart Disease Prediction API is running"}


@app.post("/predict")
def predict(patient: PatientData):
    """
    Accepts one patient's data, applies the same cleaning/encoding used
    during training, and returns a prediction + confidence score.
    """
    start_time = time.time()
    REQUEST_COUNT.inc()

    raw_df = pd.DataFrame([patient.model_dump()])

    categorical_cols = ["cp", "restecg", "slope", "thal", "ca"]
    encoded_df = pd.get_dummies(raw_df, columns=categorical_cols, drop_first=True)

    for col in config["feature_columns"]:
        if col not in encoded_df.columns:
            encoded_df[col] = 0

    encoded_df = encoded_df[config["feature_columns"]]

    scaled = scaler.transform(encoded_df)
    probability = model.predict_proba(scaled)[0, 1]
    prediction = int(probability >= config["decision_threshold"])
    diagnosis = "Disease Present" if prediction == 1 else "No Disease"

    PREDICTION_COUNT.labels(diagnosis=diagnosis).inc()
    elapsed = time.time() - start_time
    REQUEST_LATENCY.observe(elapsed)

    logger.info(
        f"Prediction request | input_age={patient.age} | "
        f"prediction={prediction} | confidence={round(float(probability), 4)} | "
        f"latency={elapsed:.4f}s"
    )

    return {
        "prediction": prediction,
        "confidence": round(float(probability), 4),
        "diagnosis": diagnosis,
        "threshold_used": config["decision_threshold"]
    }

@app.get("/metrics")
def metrics():
    """
    Exposes metrics in Prometheus's standard text format.
    A real Prometheus server would call this URL periodically to
    collect these numbers automatically.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)