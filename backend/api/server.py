from __future__ import annotations

import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.models.surrogate_model import SurrogateModel

MODEL_PATH = os.environ.get("SURROGATE_MODEL_PATH", "backend/models/surrogate.joblib")

app = FastAPI(title="Material AI Lab API", version="0.1.0")
model_cache: SurrogateModel | None = None


class PredictRequest(BaseModel):
    features: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    global model_cache
    if model_cache is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=400, detail="Model file not found. Train model first.")
        model_cache = SurrogateModel.load(MODEL_PATH)

    x_df = pd.DataFrame([req.features])
    pred = float(model_cache.predict(x_df)[0])
    return {"prediction": pred, "target": model_cache.target_column}
