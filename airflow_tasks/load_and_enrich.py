"""
Task 1: load_and_enrich
Calls build_features() from feature_engineering.py, optionally filters
to a simulation cutoff datetime (as_of), and saves to features_latest.csv.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml_models.feature_engineering import build_features

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def run(as_of: str | None = None, **context):
    """
    as_of: ISO datetime string e.g. '2015-08-01 06:00:00'
           If None, uses all available data.
           When called from Airflow, read from dag_run.conf.
    """
    # ── Read as_of from Airflow dag_run conf if available ────────────────────
    if as_of is None and context.get("dag_run"):
        as_of = context["dag_run"].conf.get("as_of")

    print(f"[load_and_enrich] as_of={as_of or 'ALL DATA'}")
    print("[load_and_enrich] Building features from raw CSVs...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    features, nan_counts, nan_rows, rows_after = build_features()

    # ── Filter to simulation cutoff if provided ───────────────────────────────
    if as_of:
        cutoff = pd.Timestamp(as_of)
        before = len(features)
        features = features[features["datetime"] <= cutoff].reset_index(drop=True)
        print(f"[load_and_enrich] Filtered {before} → {len(features)} rows (cutoff: {cutoff})")

    out_path = PROCESSED_DIR / "features_latest.csv"
    features.to_csv(out_path, index=False)
    print(f"[load_and_enrich] Done. Saved {len(features)} rows to {out_path}")
    return str(out_path)


if __name__ == "__main__":
    # Test with a specific date
    run(as_of="2015-08-01 06:00:00")