from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp", "room_temp_c", "relative_humidity_pct", "outdoor_temp_c",
    "ac_set_temp_c", "occupants", "ac_running", "ac_runtime_min",
]
NUMERIC_COLUMNS = REQUIRED_COLUMNS[1:]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate a timestamped RAC dataset from CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "CSV is missing required column(s): " + ", ".join(missing) +
            ". See data/real_data_template.csv and README.md."
        )
    df = df[REQUIRED_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        bad_count = int(df["timestamp"].isna().sum())
        raise ValueError(f"{bad_count} timestamp(s) could not be parsed. Use dates such as 2026-10-12 10:00:00.")
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[NUMERIC_COLUMNS].isna().any().any():
        bad = df[NUMERIC_COLUMNS].isna().sum()
        bad = [f"{col} ({int(n)} missing/non-numeric)" for col, n in bad.items() if n]
        raise ValueError("Missing or non-numeric measurement(s): " + ", ".join(bad))
    numeric_values = df[NUMERIC_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Numeric measurements must be finite numbers; NaN and infinity are not allowed.")
    if df["timestamp"].duplicated().any():
        examples = df.loc[df["timestamp"].duplicated(keep=False), "timestamp"].head(4).astype(str).tolist()
        raise ValueError(f"Duplicate timestamps found (examples: {examples}). Keep one observation per timestamp.")
    if not df["relative_humidity_pct"].between(0, 100).all():
        raise ValueError("relative_humidity_pct must be between 0 and 100.")
    if not df["room_temp_c"].between(-10, 60).all():
        raise ValueError("room_temp_c must be between -10 and 60 °C. Check units and sensor values.")
    if not df["outdoor_temp_c"].between(-30, 65).all():
        raise ValueError("outdoor_temp_c must be between -30 and 65 °C. Check units and sensor values.")
    if not df["ac_set_temp_c"].between(16, 32).all():
        raise ValueError("ac_set_temp_c must be between 16 and 32 °C for this RAC project.")
    if not df["ac_running"].isin([0, 1]).all():
        raise ValueError("ac_running must be 1 (ON) or 0 (OFF).")
    if (df[["occupants", "ac_runtime_min"]] < 0).any().any():
        raise ValueError("occupants and ac_runtime_min cannot be negative.")
    if (df["occupants"] > 100).any():
        raise ValueError("occupants cannot exceed 100 per room. Check whether the column is in a different unit.")
    if (df["ac_runtime_min"] > 1440).any():
        raise ValueError("ac_runtime_min cannot exceed 1440 minutes (24 hours).")
    if not (df["occupants"] % 1 == 0).all():
        raise ValueError("occupants must be a whole number.")
    if (df.loc[df["ac_running"] == 0, "ac_runtime_min"] > 0).any():
        raise ValueError("Use ac_runtime_min=0 when ac_running=0.")
    df = df.sort_values("timestamp").reset_index(drop=True)
    if len(df) < 100:
        raise ValueError(f"Only {len(df)} rows found. Add substantially more readings before model training.")
    return df
