from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

import psycopg2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DB_CONFIG = {
    "host":     "127.0.0.1",
    "port":     5432,
    "dbname":   "predictiq_db",
    "user":     "postgres",
    "password": "Goodluck@1234",
}


def run():
    run_at = datetime.now(timezone.utc)

    # ── 1. Load results from Tasks 4 and 5 ───────────────────────────────────
    print("[save_to_postgres] Loading classifier_results.json...")
    with open(PROCESSED_DIR / "classifier_results.json") as f:
        classifier_results = json.load(f)

    print("[save_to_postgres] Loading rul_results.json...")
    with open(PROCESSED_DIR / "rul_results.json") as f:
        rul_results = json.load(f)

    print(f"[save_to_postgres] {len(classifier_results)} classifier rows, {len(rul_results)} RUL rows")

    # ── 2. Insert into PostgreSQL ─────────────────────────────────────────────
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn:  # auto commit/rollback
            cur = conn.cursor()

            # ── Classifier results ────────────────────────────────────────────
            classifier_sql = """
                INSERT INTO classifier_results (
                    machine_id, overall_status, priority,
                    comp1_prob, comp1_alert, comp1_status,
                    comp2_prob, comp2_alert, comp2_status,
                    comp3_prob, comp3_alert, comp3_status,
                    comp4_prob, comp4_alert, comp4_status,
                    pipeline_run_at
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s
                )
            """
            classifier_rows = [
                (
                    r["machine_id"],
                    r["overall_status"],
                    r["priority"],
                    r["comp1"]["failure_probability"], r["comp1"]["alert"], r["comp1"]["status"],
                    r["comp2"]["failure_probability"], r["comp2"]["alert"], r["comp2"]["status"],
                    r["comp3"]["failure_probability"], r["comp3"]["alert"], r["comp3"]["status"],
                    r["comp4"]["failure_probability"], r["comp4"]["alert"], r["comp4"]["status"],
                    run_at,
                )
                for r in classifier_results
            ]
            cur.executemany(classifier_sql, classifier_rows)
            print(f"[save_to_postgres] Inserted {len(classifier_rows)} classifier rows")

            # ── RUL results ───────────────────────────────────────────────────
            rul_sql = """
                INSERT INTO rul_results (
                    machine_id, component,
                    predicted_rul_hours, day_bucket, day_label, interpretation,
                    pipeline_run_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            rul_rows = [
                (
                    r["machine_id"],
                    r["component"],
                    r["predicted_rul_hours"],
                    r["day_bucket"],
                    r["day_label"],
                    r["interpretation"],
                    run_at,
                )
                for r in rul_results
            ]
            cur.executemany(rul_sql, rul_rows)
            print(f"[save_to_postgres] Inserted {len(rul_rows)} RUL rows")

    finally:
        conn.close()

    print(f"[save_to_postgres] Done. pipeline_run_at={run_at.isoformat()}")
    return run_at.isoformat()


if __name__ == "__main__":
    run()