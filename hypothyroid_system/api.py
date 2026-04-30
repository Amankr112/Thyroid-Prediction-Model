"""
api.py
------
FastAPI REST endpoint for hypothyroid prediction.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Docs:
    http://localhost:8000/docs
"""

import os
import logging
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from preprocessing import preprocess
from feature_engineering import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR  = os.path.join(BASE_DIR, "artifacts")

# ── Load artifacts at startup ──────────────────────────────────────────────────
pre_art = joblib.load(os.path.join(ART_DIR, "preprocess_artifacts.pkl"))
scaler  = joblib.load(os.path.join(ART_DIR, "scaler.pkl"))
meta    = joblib.load(os.path.join(ART_DIR, "best_model_meta.pkl"))
xgb     = joblib.load(os.path.join(ART_DIR, "xgb_model.pkl"))

if meta["type"] == "keras":
    from tensorflow import keras as _keras
    best_model = _keras.models.load_model(os.path.join(ART_DIR, "best_model.keras"))
else:
    best_model = joblib.load(os.path.join(ART_DIR, "best_model.pkl"))

logger.info("Artifacts loaded. Best model: %s", meta["name"])

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="HypoDetect API",
    description="REST API for hypothyroid prediction using machine learning.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────
class PatientRecord(BaseModel):
    age: float = Field(..., example=45)
    sex: str = Field(..., example="F")
    on_thyroxine: str = Field("f", alias="on thyroxine", example="f")
    query_on_thyroxine: str = Field("f", alias="query on thyroxine", example="f")
    on_antithyroid_medication: str = Field("f", alias="on antithyroid medication", example="f")
    sick: str = Field("f", example="f")
    pregnant: str = Field("f", example="f")
    thyroid_surgery: str = Field("f", alias="thyroid surgery", example="f")
    I131_treatment: str = Field("f", alias="I131 treatment", example="f")
    query_hypothyroid: str = Field("f", alias="query hypothyroid", example="f")
    query_hyperthyroid: str = Field("f", alias="query hyperthyroid", example="f")
    lithium: str = Field("f", example="f")
    goitre: str = Field("f", example="f")
    tumor: str = Field("f", example="f")
    hypopituitary: str = Field("f", example="f")
    psych: str = Field("f", example="f")
    TSH_measured: str = Field("t", alias="TSH measured", example="t")
    TSH: float = Field(2.0, example=2.0)
    T3_measured: str = Field("t", alias="T3 measured", example="t")
    T3: float = Field(2.0, example=2.0)
    TT4_measured: str = Field("t", alias="TT4 measured", example="t")
    TT4: float = Field(100.0, example=100.0)
    T4U_measured: str = Field("t", alias="T4U measured", example="t")
    T4U: float = Field(1.0, example=1.0)
    FTI_measured: str = Field("t", alias="FTI measured", example="t")
    FTI: float = Field(100.0, example=100.0)
    TBG_measured: str = Field("f", alias="TBG measured", example="f")
    TBG: str = Field("?", example="?")
    referral_source: str = Field("other", alias="referral source", example="other")

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    model_used: str


class BatchRecord(BaseModel):
    records: List[PatientRecord]


def _run_prediction(record_dict: dict) -> tuple[str, float]:
    df = pd.DataFrame([record_dict])
    X, _, _ = preprocess(df, fit=False, artifacts=pre_art)
    X = engineer_features(X)

    need_scale = meta["name"] in ("Logistic Regression", "SVM", "Deep Learning")
    X_in = scaler.transform(X) if need_scale else X.values

    if meta["type"] == "keras":
        prob = float(best_model.predict(X_in, verbose=0).ravel()[0])
    elif hasattr(best_model, "predict_proba"):
        prob = float(best_model.predict_proba(X_in)[:, 1][0])
    else:
        d = best_model.decision_function(X_in)
        prob = float((d - d.min()) / (d.max() - d.min() + 1e-9))

    label = "Hypothyroid" if prob > 0.5 else "Normal"
    return label, round(prob, 4)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": meta["name"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientRecord):
    try:
        record = patient.dict(by_alias=True)
        label, prob = _run_prediction(record)
        return PredictionResponse(
            prediction=label,
            probability=prob,
            model_used=meta["name"],
        )
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(batch: BatchRecord):
    try:
        results = []
        for record in batch.records:
            rd = record.dict(by_alias=True)
            label, prob = _run_prediction(rd)
            results.append({"prediction": label, "probability": prob})
        return {"results": results, "count": len(results), "model_used": meta["name"]}
    except Exception as e:
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
def model_info():
    import pandas as pd, os
    comp_path = os.path.join(ART_DIR, "model_comparison.csv")
    comp = pd.read_csv(comp_path, index_col=0).to_dict() if os.path.exists(comp_path) else {}
    return {"best_model": meta["name"], "model_type": meta["type"], "comparison": comp}
