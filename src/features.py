
"""Data loading and feature engineering for NASA C-MAPSS (FD001)."""
import pandas as pd

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
 
# Columns with (almost) no variation in FD001, found in 01_eda.ipynb
DROP_COLS = ["s1","op3","s5","s10","s16","s18","s19","s6","op1", "op2"]
SENSORS = [c for c in COLS if c.startswith("s") and c not in DROP_COLS]
 
def load_cmapss(path):
    """Read one C-MAPSS text file (train_FD001.txt or test_FD001.txt)."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS)

def add_rul(df, cap=125):
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["RUL"] = (max_cycle - df["cycle"]).clip(upper=cap)
    return df

def make_features(df, sensors=SENSORS, windows=(5, 10, 30)):
    """Rolling mean/std and last-cycle change per sensor, computed per engine."""
    df = df.sort_values(["unit", "cycle"]).copy()
    g = df.groupby("unit")
    new_cols = {}
    for s in sensors:
        for w in windows:
            new_cols[f"{s}_mean{w}"] = g[s].transform(lambda x: x.rolling(w, min_periods=1).mean())
            new_cols[f"{s}_std{w}"] = g[s].transform(lambda x: x.rolling(w, min_periods=1).std()).fillna(0)
        new_cols[f"{s}_diff"] = g[s].diff().fillna(0)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

def feature_columns(df):
    """All model inputs: everything except the engine id and the target."""
    return [c for c in df.columns if c not in ["unit", "RUL"] + DROP_COLS]