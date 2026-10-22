"""
Task 4: run_classifier
Reads classifier_input.json, calls POST /classify/batch,
saves results to data/processed/classifier_results.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
API_BASE = "http://localhost:8000/api/v1"
TIMEOUT = 120


def run():
    input_path = PROCESSED_DIR / "classifier_input.json"
    print(f"[run_classifier] Reading {input_path}...")

    with open(input_path) as f:
        records = json.load(f)

    print(f"[run_classifier] Calling POST {API_BASE}/classify/batch with {len(records)} machines...")

    payload = {"requests": records}

    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(f"{API_BASE}/classify/batch", json=payload)
        resp.raise_for_status()

    response_data = resp.json()

    if response_data.get("status") != "success":
        raise RuntimeError(f"Classifier API error: {response_data.get('error')}")

    results = response_data["data"]
    out_path = PROCESSED_DIR / "classifier_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f)

    print(f"[run_classifier] Saved {len(results)} results to {out_path}")
    return str(out_path)


if __name__ == "__main__":
    run()