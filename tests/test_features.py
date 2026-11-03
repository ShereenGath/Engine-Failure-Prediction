import pandas as pd

from src.features import SENSORS, add_rul, make_features


def small_engine_data():
    """Two fake engines: engine 1 lives 5 cycles, engine 2 lives 3 cycles."""
    rows = []
    for unit, life in [(1, 5), (2, 3)]:
        for c in range(1, life + 1):
            rows.append({"unit": unit, "cycle": c, **{s: float(c) for s in SENSORS}})
    return pd.DataFrame(rows)


def test_rul_is_zero_at_last_cycle():
    df = add_rul(small_engine_data())
    last_rows = df.groupby("unit").tail(1)
    assert (last_rows["RUL"] == 0).all()


def test_rul_is_capped():
    df = add_rul(small_engine_data(), cap=2)
    assert df["RUL"].max() == 2


def test_make_features_keeps_row_count():
    df = small_engine_data()
    assert len(make_features(df)) == len(df)


def test_features_do_not_mix_engines():
    df = make_features(small_engine_data())
    first_cycle_engine_2 = df[(df.unit == 2) & (df.cycle == 1)]
    # rolling mean at engine 2's first cycle must use only engine 2's value (1.0)
    assert first_cycle_engine_2["s2_mean5"].iloc[0] == 1.0
