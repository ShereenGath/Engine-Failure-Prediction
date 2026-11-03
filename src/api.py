"""Web API that predicts remaining useful life (RUL) for one engine.

Run from the project's main folder:
    uvicorn src.api:app --reload
Then open http://127.0.0.1:8000/docs
"""
import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import XGBRegressor

from src.features import make_features

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

# Load the model ONCE when the API starts (not on every request)
meta = json.loads((MODELS / "metadata.json").read_text())
model = XGBRegressor()
model.load_model(MODELS / "model.json")

app = FastAPI(
    title="Engine RUL API",
    description="Predicts how many cycles a turbofan engine has left before failure "
                "(NASA C-MAPSS FD001, XGBoost).",
)


class Cycle(BaseModel):
    """Sensor readings of one engine at one cycle."""
    cycle: int = Field(..., ge=1, description="Cycle number (1 = first flight)")
    s2: float
    s3: float
    s4: float
    s7: float
    s8: float
    s9: float
    s11: float
    s12: float
    s13: float
    s14: float
    s15: float
    s17: float
    s20: float
    s21: float


class EngineHistory(BaseModel):
    """Recent history of one engine, oldest cycle first. Send up to 30 cycles."""
    cycles: list[Cycle] = Field(..., min_length=1)


class Prediction(BaseModel):
    predicted_rul: float
    cycles_received: int
    note: str


@app.get("/health")
def health():
    """Quick check that the API is running."""
    return {"status": "ok", "model": "xgboost", "n_features": len(meta["features"])}


@app.post("/predict", response_model=Prediction)
def predict(history: EngineHistory):
    # 1. Turn the JSON list into a table, one row per cycle
    df = pd.DataFrame([c.model_dump() for c in history.cycles])
    if df["cycle"].duplicated().any():
        raise HTTPException(status_code=422, detail="Each cycle number must appear only once.")
    df["unit"] = 1

    # 2. Build exactly the same features as in training, keep the latest cycle
    feats = make_features(df, sensors=meta["sensors"]).tail(1)

    # 3. Predict and keep the answer in a sensible range
    rul = float(model.predict(feats[meta["features"]])[0])
    rul = min(max(rul, 0.0), float(meta["rul_cap"]))

    note = ("Healthy: no clear degradation yet." if rul >= meta["rul_cap"] - 5
            else "Plan maintenance soon." if rul < 30
            else "Degradation visible: monitor.")
    return Prediction(predicted_rul=round(rul, 1), cycles_received=len(df), note=note)
