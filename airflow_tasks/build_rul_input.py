"""
Task 3: build_rul_input
Reuses enrich_telemetry() from train_rul_multihead_v2.py — the exact same
6h feature engineering used during training — then extracts the last 48
rows per machine as the RUL input sequence.
Saves to data/processed/rul_input.json — one entry per (machine, component).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml_models.rul_regression.train_rul_multihead_v2 import (
    enrich_telemetry,
    FEATURE_COLS,
    SEQUENCE_LENGTH,
    COMPONENTS,
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def run():
    print("[build_rul_input] Loading raw telemetry and errors...")
    telemetry = pd.read_csv(RAW_DIR / "PdM_telemetry.csv", parse_dates=["datetime"])
    errors = pd.read_csv(RAW_DIR / "PdM_errors.csv", parse_dates=["datetime"])

    telemetry = telemetry.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    errors = errors.sort_values(["machineID", "datetime"]).reset_index(drop=True)

    print("[build_rul_input] Engineering 6h features (reusing training code)...")
    enriched = enrich_telemetry(telemetry, errors)

    print("[build_rul_input] Extracting last 48 rows per machine...")
    records = []
    for machine_id in enriched["machineID"].unique():
        machine_data = enriched[enriched["machineID"] == machine_id].tail(SEQUENCE_LENGTH)

        if len(machine_data) < SEQUENCE_LENGTH:
            print(f"  [WARN] machine {machine_id} only has {len(machine_data)} rows — skipping")
            continue

        sequence = machine_data[FEATURE_COLS].values.tolist()

        for comp in COMPONENTS:
            records.append({
                "machine_id": int(machine_id),
                "component": comp,
                "sequence": sequence,
            })

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "rul_input.json"
    with open(out_path, "w") as f:
        json.dump(records, f)

    print(f"[build_rul_input] Saved {len(records)} sequences to {out_path}")
    return str(out_path)


if __name__ == "__main__":
    run()