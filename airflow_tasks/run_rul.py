from __future__ import annotations
import json
import math
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

API_BASE = "http://localhost:8000/api/v1"
BATCH_SIZE  = 100   # BatchRULRequest max_length
TIMEOUT     = 120   # seconds per batch call


def run():
    # ── 1. Load the 400 sequences produced by Task 3 ─────────────────────────
    input_path = PROCESSED_DIR / "rul_input.json"
    print(f"[run_rul] Reading {input_path}...")
    with open(input_path) as f:
        records = json.load(f)
    print(f"[run_rul] Loaded {len(records)} sequences")

    # ── 2. Rename 'sequence' → 'sensor_sequence' (what the API expects) ──────
    api_records = [
        {
            "machine_id": r["machine_id"],
            "component":  r["component"],
            "sensor_sequence": r["sequence"],   # rename here
        }
        for r in records
    ]

    # ── 3. Split into chunks of 100 and call the API ──────────────────────────
    num_batches = math.ceil(len(api_records) / BATCH_SIZE)
    print(f"[run_rul] Calling POST {API_BASE}/rul/batch in {num_batches} batches of {BATCH_SIZE}...")

    all_results = []

    with httpx.Client(timeout=TIMEOUT) as client:
        for i in range(num_batches):
            chunk = api_records[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            print(f"  [run_rul] Batch {i + 1}/{num_batches} — {len(chunk)} sequences...")

            resp = client.post(f"{API_BASE}/rul/batch", json={"requests": chunk})
            resp.raise_for_status()

            response_data = resp.json()
            if response_data.get("status") != "success":
                raise RuntimeError(
                    f"RUL API error on batch {i + 1}: {response_data.get('error')}"
                )

            batch_results = response_data["data"]
            all_results.extend(batch_results)
            print(f"  [run_rul] Batch {i + 1} done — got {len(batch_results)} results")

    # ── 4. Save all 400 results ───────────────────────────────────────────────
    out_path = PROCESSED_DIR / "rul_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f)
    print(f"[run_rul] Saved {len(all_results)} RUL results to {out_path}")
    return str(out_path)


if __name__ == "__main__":
    run()