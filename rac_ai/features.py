from __future__ import annotations
import numpy as np
import pandas as pd

BASE_FEATURES = [
    "room_temp_c", "relative_humidity_pct", "outdoor_temp_c", "ac_set_temp_c",
    "occupants", "ac_running", "ac_runtime_min",
]
HISTORY_FEATURES = [
    "room_temp_lag_5m_c", "room_temp_lag_10m_c", "room_temp_lag_15m_c",
    "room_temp_change_15m_c", "room_temp_mean_15m_c",
]
FEATURE_COLUMNS = BASE_FEATURES + HISTORY_FEATURES


def make_feature_frame(df: pd.DataFrame, include_targets: bool = True) -> pd.DataFrame:
    """Use only information available at time t to make feature rows.

    Future targets are attached separately at t+30 and t+60 minutes. Lag
    temperatures are looked up by exact timestamp, rather than simply by row,
    so an omitted observation is not mistakenly treated as a 5-minute lag.
    """
    out = df.sort_values("timestamp").copy().reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    temp_by_time = pd.Series(out["room_temp_c"].to_numpy(), index=out["timestamp"])
    ts = out["timestamp"]
    out["room_temp_lag_5m_c"] = (ts - pd.Timedelta(minutes=5)).map(temp_by_time)
    out["room_temp_lag_10m_c"] = (ts - pd.Timedelta(minutes=10)).map(temp_by_time)
    out["room_temp_lag_15m_c"] = (ts - pd.Timedelta(minutes=15)).map(temp_by_time)
    out["room_temp_change_15m_c"] = out["room_temp_c"] - out["room_temp_lag_15m_c"]
    out["room_temp_mean_15m_c"] = out[["room_temp_c", "room_temp_lag_5m_c", "room_temp_lag_10m_c", "room_temp_lag_15m_c"]].mean(axis=1)
    if include_targets:
        out["target_timestamp_30m"] = ts + pd.Timedelta(minutes=30)
        out["target_room_temp_30m_c"] = (ts + pd.Timedelta(minutes=30)).map(temp_by_time)
        out["target_timestamp_60m"] = ts + pd.Timedelta(minutes=60)
        out["target_room_temp_60m_c"] = (ts + pd.Timedelta(minutes=60)).map(temp_by_time)
    return out


def make_single_feature_row(values: dict) -> pd.DataFrame:
    """Build a single model input from a dashboard/manual forecast form."""
    current = float(values["room_temp_c"])
    lag5 = float(values["room_temp_lag_5m_c"])
    lag10 = float(values["room_temp_lag_10m_c"])
    lag15 = float(values["room_temp_lag_15m_c"])
    row = {
        "room_temp_c": current,
        "relative_humidity_pct": float(values["relative_humidity_pct"]),
        "outdoor_temp_c": float(values["outdoor_temp_c"]),
        "ac_set_temp_c": float(values["ac_set_temp_c"]),
        "occupants": int(values["occupants"]),
        "ac_running": int(bool(values["ac_running"])),
        "ac_runtime_min": float(values["ac_runtime_min"]) if bool(values["ac_running"]) else 0.0,
        "room_temp_lag_5m_c": lag5,
        "room_temp_lag_10m_c": lag10,
        "room_temp_lag_15m_c": lag15,
        "room_temp_change_15m_c": current - lag15,
        "room_temp_mean_15m_c": float(np.mean([current, lag5, lag10, lag15])),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
