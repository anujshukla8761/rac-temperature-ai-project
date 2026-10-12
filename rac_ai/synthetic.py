from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .data import REQUIRED_COLUMNS


def generate_synthetic_data(days: int = 30, seed: int = 42) -> pd.DataFrame:
    """Generate physically inspired but SIMULATED educational room data."""
    if days < 3:
        raise ValueError("Use at least 3 days for a useful example dataset.")
    rng = np.random.default_rng(seed)
    n = days * 24 * 12  # one row every five minutes
    times = pd.date_range("2026-09-01 00:00:00", periods=n, freq="5min")
    rows = []
    room = 29.2
    was_on = False
    runtime = 0
    setpoint = 24.0
    previous_day = -1
    for i, ts in enumerate(times):
        hour = ts.hour + ts.minute / 60.0
        day = ts.dayofyear
        if ts.day != previous_day:
            setpoint = float(rng.choice([23.0, 24.0, 25.0, 26.0]))
            previous_day = ts.day
        # Synthetic warm-climate day: warmer in the afternoon, cooler at night.
        outdoor = 32.0 + 5.0 * np.sin(2 * np.pi * (hour - 9.0) / 24.0) + rng.normal(0, 0.35)
        outdoor = float(np.clip(outdoor, 23.0, 42.0))
        # Variable daily use schedule and occupancy; not a representation of a real building.
        occupied_window = (8 <= hour < 13.5) or (15.5 <= hour < 23)
        occupants = int(rng.choice([0, 1, 2, 3, 4, 5], p=[0.20, 0.12, 0.26, 0.22, 0.14, 0.06])) if occupied_window else int(rng.choice([0, 1], p=[0.92, 0.08]))
        ac_scheduled = (8.5 <= hour < 14.0) or (16.0 <= hour < 23.0)
        # Occasional interruptions and cycling near setpoint.
        if ac_scheduled:
            ac_on = bool(rng.random() > 0.06)
            if room <= setpoint - 0.25 and rng.random() < 0.7:
                ac_on = False
            elif room >= setpoint + 0.8:
                ac_on = True
        else:
            ac_on = False
        if ac_on:
            runtime = runtime + 5 if was_on else 0
        else:
            runtime = 0
        was_on = ac_on
        # Illustrative first-order heat balance: outdoor/solar/internal gains minus cooling.
        solar_gain = max(0.0, np.sin(np.pi * (hour - 6.0) / 13.0)) * 0.035
        envelope_gain = 0.012 * (outdoor - room)
        people_gain = 0.012 * occupants
        cooling = (0.105 + 0.00015 * min(runtime, 120)) if ac_on and room > setpoint else (0.012 if ac_on else 0.0)
        room = room + envelope_gain + solar_gain + people_gain - cooling + rng.normal(0, 0.035)
        room = float(np.clip(room, 18.0, 39.0))
        # RH is simulated: AC tends to reduce humidity while outdoor/occupant load raises it.
        previous_rh = rows[-1]["relative_humidity_pct"] if rows else 62.0
        rh = previous_rh + (0.30 if not ac_on else -0.11) + 0.10 * occupants + 0.02 * (outdoor - 30) + rng.normal(0, 0.65)
        rh = float(np.clip(rh, 30.0, 85.0))
        rows.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "room_temp_c": round(room, 3),
            "relative_humidity_pct": round(rh, 2),
            "outdoor_temp_c": round(outdoor, 2),
            "ac_set_temp_c": setpoint,
            "occupants": occupants,
            "ac_running": int(ac_on),
            "ac_runtime_min": runtime if ac_on else 0,
        })
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def main():
    parser = argparse.ArgumentParser(description="Generate SIMULATED RAC temperature practice data (not real measurements).")
    parser.add_argument("--output", default="data/sample_temperature_data.csv")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_data(args.days, args.seed)
    df.to_csv(out, index=False)
    print("SIMULATED dataset written (not real sensor measurements):", out.resolve())
    print(f"Rows: {len(df):,}; interval: 5 minutes; span: {df.timestamp.iloc[0]} to {df.timestamp.iloc[-1]}")
    print("Room temperature range:", f"{df.room_temp_c.min():.2f}–{df.room_temp_c.max():.2f} °C")

if __name__ == "__main__":
    main()
