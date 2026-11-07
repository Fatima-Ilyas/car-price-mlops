# app.py - car-price API that also keeps a diary of every prediction
# and exposes metrics for Prometheus

import csv
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

app = FastAPI(title="Car Price API")

# Load the model ONCE when the API starts (not on every request)
model = joblib.load("champion.pkl")

# The prediction diary: one row per request (used later for drift checks)
LOG_FILE = "predictions.csv"

# Metrics: numbers Prometheus collects every few seconds
PREDICTIONS = Counter("predictions_total", "How many predictions the API has made")
LATENCY = Histogram("prediction_seconds", "How long one prediction takes")
ENGINE = Histogram("input_engine_cc", "Engine sizes users ask about",
                   buckets=[700, 1000, 1300, 1600, 2000, 2500, 3000, 4000, 5000])
PRICES = Histogram("predicted_price_lakh", "Prices the model predicts",
                   buckets=[25, 50, 100, 150, 200, 300, 500, 800])


# The shape of a request. FastAPI rejects anything that doesn't match.
class Car(BaseModel):
    age_years: int = Field(gt=0, description="Age of the car in years")
    mileage_thousand_km: float = Field(gt=0, description="Distance driven, in thousand km")
    engine_cc: int = Field(gt=0, description="Engine size in cc")


def log_prediction(car, price):
    is_new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["time", "age_years", "mileage_thousand_km", "engine_cc", "predicted_price_lakh"])
        writer.writerow([datetime.now(timezone.utc).isoformat(), car.age_years,
                         car.mileage_thousand_km, car.engine_cc, price])


# Door 1: "are you alive?"
@app.get("/health")
def health():
    return {"status": "ok"}


# Door 2: the real job
@app.post("/predict")
def predict(car: Car):
    with LATENCY.time():
        data = pd.DataFrame([{"age_years": car.age_years,
                              "mileage_thousand_km": car.mileage_thousand_km,
                              "engine_cc": car.engine_cc}])
        price = round(float(model.predict(data)[0]), 1)

    PREDICTIONS.inc()
    ENGINE.observe(car.engine_cc)
    PRICES.observe(price)
    log_prediction(car, price)
    return {"price_lakh": price}


# Door 3: metrics for Prometheus
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
