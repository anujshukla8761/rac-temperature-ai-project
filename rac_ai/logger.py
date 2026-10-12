from __future__ import annotations
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import time
import pandas as pd
from .data import REQUIRED_COLUMNS


def ask_float(label, default=None, minimum=None, maximum=None):
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{label}{suffix}: ").strip()
        if not raw and default is not None:
            val = float(default)
        else:
            try:
                val = float(raw)
            except ValueError:
                print("Enter a numeric value.")
                continue
        if minimum is not None and val < minimum:
            print(f"Value must be at least {minimum}.")
            continue
        if maximum is not None and val > maximum:
            print(f"Value must be no greater than {maximum}.")
            continue
        return val


def log_loop(output: str | Path, interval_minutes: int = 5, once: bool = False):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not path.exists() or path.stat().st_size == 0
    print("Manual RAC data logger. Enter measured values; do not make up measurements.")
    print("Saving to:", path.resolve())
    print("Stop with Ctrl+C. Use consistent 5-minute intervals for the supplied modelling workflow.\n")
    previous_on = False
    runtime = 0
    while True:
        try:
            # Timestamp the observation at the beginning of manual entry.
            # Scheduling against this timestamp keeps logger rows on a regular
            # interval unless a manual entry takes longer than the interval.
            observation_time = datetime.now().replace(second=0, microsecond=0)
            print(f"--- New observation at {observation_time:%Y-%m-%d %H:%M:%S} ---")
            room = ask_float("Room temperature (°C)", minimum=-5, maximum=60)
            rh = ask_float("Indoor relative humidity (%)", minimum=0, maximum=100)
            outdoor = ask_float("Outdoor temperature (°C)", minimum=-20, maximum=60)
            setpoint = ask_float("AC set temperature (°C)", default=24, minimum=16, maximum=32)
            occupants = int(ask_float("Number of occupants", default=0, minimum=0, maximum=100))
            while True:
                ac_raw = input("Is AC running? Enter 1=ON or 0=OFF: ").strip()
                if ac_raw in {"0", "1"}:
                    ac_on = int(ac_raw)
                    break
                print("Enter only 1 or 0.")
            if ac_on:
                runtime = runtime + interval_minutes if previous_on else 0
            else:
                runtime = 0
            previous_on = bool(ac_on)
            stamp = observation_time.strftime("%Y-%m-%d %H:%M:%S")
            row = {"timestamp": stamp, "room_temp_c": room, "relative_humidity_pct": rh,
                   "outdoor_temp_c": outdoor, "ac_set_temp_c": setpoint, "occupants": occupants,
                   "ac_running": ac_on, "ac_runtime_min": runtime if ac_on else 0}
            pd.DataFrame([row], columns=REQUIRED_COLUMNS).to_csv(path, mode="a", header=need_header, index=False)
            need_header = False
            print(f"Saved {stamp}. AC runtime recorded as {row['ac_runtime_min']} min.\n")
            if once:
                break
            next_due = observation_time + timedelta(minutes=interval_minutes)
            remaining = (next_due - datetime.now()).total_seconds()
            if remaining > 0:
                print(f"Next observation at approximately {next_due:%H:%M:%S}.")
                time.sleep(remaining)
            else:
                print("Manual entry took longer than the sampling interval; the next observation will start now. Check timestamp gaps before training.")
        except KeyboardInterrupt:
            print("\nLogger stopped. File saved:", path.resolve())
            break


def main():
    parser = argparse.ArgumentParser(description="Collect manual RAC observations into a CSV file.")
    parser.add_argument("--output", default="data/real_temperature_data.csv")
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--once", action="store_true", help="Record one row and exit.")
    args = parser.parse_args()
    if args.interval_minutes <= 0:
        parser.error("--interval-minutes must be positive")
    log_loop(args.output, args.interval_minutes, args.once)

if __name__ == "__main__":
    main()
