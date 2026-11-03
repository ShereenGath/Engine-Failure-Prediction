"""API tests. Run `py -m src.train` once before `pytest` so the model exists."""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)
EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "engine_1.json"


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_a_number():
    r = client.post("/predict", json=json.loads(EXAMPLE.read_text()))
    assert r.status_code == 200
    rul = r.json()["predicted_rul"]
    assert 0 <= rul <= 125


def test_predict_rejects_missing_sensor():
    bad = {"cycles": [{"cycle": 1, "s2": 642.0}]}   # most sensors missing
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
