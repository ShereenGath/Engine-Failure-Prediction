"""Train the final RUL model and save it for the API.

Run from the project's main folder:
    py -m src.train
"""
import json
from pathlib import Path

from xgboost import XGBRegressor

from src.features import SENSORS, add_rul, feature_columns, load_cmapss, make_features

ROOT = Path(__file__).resolve().parent.parent      # the project's main folder
DATA = ROOT / "data"
MODELS = ROOT / "models"
EXAMPLES = ROOT / "examples"

RUL_CAP = 125
XGB_PARAMS = {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.05,
              "subsample": 0.8, "colsample_bytree": 0.8}


def main():
    # 1. Load training data and create the target
    train = add_rul(load_cmapss(DATA / "train_FD001.txt"), cap=RUL_CAP)

    # 2. Build the same features as in the notebook
    train_feat = make_features(train)
    feat_cols = feature_columns(train_feat)

    # 3. Train the chosen model on ALL training engines
    model = XGBRegressor(**XGB_PARAMS, random_state=42, n_jobs=-1)
    model.fit(train_feat[feat_cols], train_feat["RUL"])

    # 4. Save the model + the information the API needs to use it
    MODELS.mkdir(exist_ok=True)
    model.save_model(MODELS / "model.json")
    meta = {"features": feat_cols, "sensors": SENSORS, "rul_cap": RUL_CAP,
            "xgb_params": XGB_PARAMS}
    (MODELS / "metadata.json").write_text(json.dumps(meta, indent=2))

    # 5. Save an example request (last 30 cycles of test engine 1) to try the API
    test = load_cmapss(DATA / "test_FD001.txt")
    engine = test[test.unit == 1].tail(30)
    example = {"cycles": engine[["cycle"] + SENSORS].to_dict(orient="records")}
    EXAMPLES.mkdir(exist_ok=True)
    (EXAMPLES / "engine_1.json").write_text(json.dumps(example, indent=2))

    print(f"Model trained on {train['unit'].nunique()} engines, {len(feat_cols)} features.")
    print(f"Saved: {MODELS / 'model.json'}")
    print(f"Saved: {MODELS / 'metadata.json'}")
    print(f"Saved example request: {EXAMPLES / 'engine_1.json'}")


if __name__ == "__main__":
    main()
