from __future__ import annotations

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from airflow_tasks.load_and_enrich        import run as load_and_enrich
from airflow_tasks.build_classifier_input import run as build_classifier_input
from airflow_tasks.build_rul_input        import run as build_rul_input
from airflow_tasks.run_classifier         import run as run_classifier
from airflow_tasks.run_rul                import run as run_rul
from airflow_tasks.save_to_postgres       import run as save_to_postgres

default_args = {
    "owner": "predictiq",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "email_on_failure": False,
}

with DAG(
    dag_id="predictiq_pipeline",
    description="Hourly predictive maintenance — classify + RUL for 100 machines",
    schedule="@hourly",
    start_date=datetime(2026, 10, 22),
    catchup=False,
    default_args=default_args,
    tags=["predictiq", "maintenance"],
) as dag:

    t1 = PythonOperator(
        task_id="load_and_enrich",
        python_callable=load_and_enrich,
        op_kwargs={},           # as_of comes from dag_run.conf via **context
        provide_context=True,   # injects dag_run, ti, etc. into run()
    )

    t2 = PythonOperator(
        task_id="build_classifier_input",
        python_callable=build_classifier_input,
    )

    t3 = PythonOperator(
        task_id="build_rul_input",
        python_callable=build_rul_input,
    )

    t4 = PythonOperator(
        task_id="run_classifier",
        python_callable=run_classifier,
    )

    t5 = PythonOperator(
        task_id="run_rul",
        python_callable=run_rul,
    )

    t6 = PythonOperator(
        task_id="save_to_postgres",
        python_callable=save_to_postgres,
    )

    t1 >> t2 >> t3 >> t4 >> t5 >> t6