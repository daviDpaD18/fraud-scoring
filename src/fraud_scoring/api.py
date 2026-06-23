from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time


app = FastAPI(title="Fraud Scoring API")
model = joblib.load("models/model.joblib")


PREDICTIONS = Counter("fraud_predictions_total", "Total predictions", ["result"])
LATENCY = Histogram("fraud_prediction_latency_seconds", "Prediction latency")
CONFIDENCE = Histogram("fraud_prediction_confidence", "Confidence of predictions", buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

class PredictRequest(BaseModel):
    features: list[float]

class PredictResponse(BaseModel):
    fraud: bool
    confidence: float
    prediction_time_ms: float


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start = time.time()
    X = np.array(request.features).reshape(1, -1)
    proba = model.predict_proba(X)[0][1]
    fraud = bool(proba > 0.5)
    elapsed = (time.time() - start) * 1000  # Convert to milliseconds
    PREDICTIONS.labels(result="fraud" if fraud else "legitimate").inc()
    LATENCY.observe(elapsed / 1000)  # Convert to seconds for Prometheus
    CONFIDENCE.observe(proba)
    return PredictResponse(fraud=fraud, confidence=float(proba), prediction_time_ms=float(elapsed))

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health():
    return {"status": "ok"}