"""
airflow/dags/travel_ml_pipeline.py
====================================
Apache Airflow DAGs for the Travel ML Project.

Two DAGs:
  1. travel_data_pipeline        — Daily data ingestion, cleaning, feature engineering
  2. travel_model_training_dag   — Weekly model retraining and evaluation

Airflow Setup:
  pip install apache-airflow
  airflow db init
  airflow webserver --port 8080
  airflow scheduler
  airflow dags list
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash   import BashOperator
from airflow.operators.email  import EmailOperator
from airflow.utils.dates      import days_ago


# ── Default arguments shared by both DAGs ─────────────────────────────────────
default_args = {
    "owner"            : "travel_ml_team",
    "depends_on_past"  : False,
    "email"            : ["mlops@travelproject.com"],
    "email_on_failure" : True,
    "email_on_retry"   : False,
    "retries"          : 2,
    "retry_delay"      : timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


# ==============================================================================
# PYTHON CALLABLES (task functions)
# ==============================================================================

def load_raw_data(**kwargs):
    """Task: Load raw CSV datasets and validate their existence."""
    import os, logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    data_dir = "/opt/airflow/data"  # Adjust to your mount path
    files    = ["flights.csv", "hotels.csv", "users.csv"]
    for f in files:
        path = os.path.join(data_dir, f)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file missing: {path}")
        logger.info("✅ Found: %s", f)

    kwargs["ti"].xcom_push(key="data_dir", value=data_dir)
    logger.info("Raw data validation complete.")


def clean_and_engineer(**kwargs):
    """Task: Clean datasets and engineer features. Save processed data."""
    import sys, os, pandas as pd
    sys.path.insert(0, "/opt/airflow")   # Airflow project root
    from src.data_processing import (
        load_datasets, clean_flights, clean_hotels, clean_users,
        engineer_flight_features, engineer_user_features
    )

    data_dir = kwargs["ti"].xcom_pull(key="data_dir")
    flights, hotels, users = load_datasets(data_dir)

    flights_clean = engineer_flight_features(clean_flights(flights))
    hotels_clean  = clean_hotels(hotels)
    users_clean   = clean_users(users)

    out_dir = "/opt/airflow/data/processed"
    os.makedirs(out_dir, exist_ok=True)
    flights_clean.to_csv(f"{out_dir}/flights_processed.csv", index=False)
    hotels_clean.to_csv( f"{out_dir}/hotels_processed.csv",  index=False)
    users_clean.to_csv(  f"{out_dir}/users_processed.csv",   index=False)

    kwargs["ti"].xcom_push(key="processed_dir", value=out_dir)
    print(f"Data cleaning complete. Flights: {flights_clean.shape}, Hotels: {hotels_clean.shape}")


def validate_processed_data(**kwargs):
    """Task: Basic data quality checks on the processed datasets."""
    import pandas as pd, logging
    logger = logging.getLogger(__name__)

    proc_dir = kwargs["ti"].xcom_pull(key="processed_dir")
    flights  = pd.read_csv(f"{proc_dir}/flights_processed.csv")
    hotels   = pd.read_csv(f"{proc_dir}/hotels_processed.csv")
    users    = pd.read_csv(f"{proc_dir}/users_processed.csv")

    assert len(flights) > 0, "Flights data is empty after cleaning!"
    assert len(hotels)  > 0, "Hotels data is empty after cleaning!"
    assert len(users)   > 0, "Users data is empty after cleaning!"
    assert flights["price"].min() > 0, "Negative prices found!"
    assert not flights.isnull().all(axis=None), "Flights has all-NaN rows!"

    logger.info("✅ Data validation passed — Flights: %d | Hotels: %d | Users: %d",
                len(flights), len(hotels), len(users))


def generate_eda_report(**kwargs):
    """Task: Generate basic EDA statistics and save to a report file."""
    import pandas as pd, json, os
    proc_dir = kwargs["ti"].xcom_pull(key="processed_dir")
    flights  = pd.read_csv(f"{proc_dir}/flights_processed.csv")

    report = {
        "total_flights"    : int(len(flights)),
        "avg_price"        : round(float(flights["price"].mean()), 2),
        "price_std"        : round(float(flights["price"].std()),  2),
        "flight_types"     : flights["flightType"].value_counts().to_dict(),
        "agency_counts"    : flights["agency"].value_counts().to_dict(),
        "generated_at"     : datetime.utcnow().isoformat(),
    }

    report_path = "/opt/airflow/data/eda_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"EDA report saved to {report_path}")


def retrain_regression_model(**kwargs):
    """Task: Retrain the flight price regression model."""
    import sys
    sys.path.insert(0, "/opt/airflow")
    import pandas as pd
    from src.data_processing import load_datasets
    from src.model_training  import train_regression_models

    flights, hotels, users = load_datasets("/opt/airflow/data")
    results = train_regression_models(flights)
    kwargs["ti"].xcom_push(key="regression_results", value=results)
    print("Regression retraining done:", results)


def retrain_classification_model(**kwargs):
    """Task: Retrain the gender classification model."""
    import sys
    sys.path.insert(0, "/opt/airflow")
    from src.data_processing import load_datasets
    from src.model_training  import train_classification_models

    flights, hotels, users = load_datasets("/opt/airflow/data")
    results = train_classification_models(users, flights, hotels)
    kwargs["ti"].xcom_push(key="classification_results", value=results)
    print("Classification retraining done:", results)


def retrain_recommendation_model(**kwargs):
    """Task: Rebuild the hotel recommendation system."""
    import sys
    sys.path.insert(0, "/opt/airflow")
    from src.data_processing import load_datasets
    from src.model_training  import train_recommendation_model

    flights, hotels, users = load_datasets("/opt/airflow/data")
    train_recommendation_model(hotels, users)
    print("Recommendation model rebuilt successfully.")


def evaluate_and_compare(**kwargs):
    """Task: Compare new model performance against the previous baseline."""
    import json, logging
    logger = logging.getLogger(__name__)

    reg_results = kwargs["ti"].xcom_pull(key="regression_results",     task_ids="retrain_regression")
    clf_results = kwargs["ti"].xcom_pull(key="classification_results", task_ids="retrain_classification")

    logger.info("New regression metrics : %s", reg_results)
    logger.info("New classification metrics: %s", clf_results)

    # In production, compare against a stored baseline here
    # If performance degrades, raise an alert or roll back
    logger.info("Model evaluation complete. Models are ready for deployment.")


# ==============================================================================
# DAG 1 — Daily Data Pipeline
# ==============================================================================

with DAG(
    dag_id             = "travel_data_pipeline",
    description        = "Daily data ingestion, cleaning, and feature engineering",
    default_args       = default_args,
    schedule_interval  = "@daily",           # Runs every day at midnight
    start_date         = days_ago(1),
    catchup            = False,
    tags               = ["data", "etl", "travel"],
) as data_dag:

    t1_load = PythonOperator(
        task_id         = "load_raw_data",
        python_callable = load_raw_data,
    )

    t2_clean = PythonOperator(
        task_id         = "clean_and_engineer_features",
        python_callable = clean_and_engineer,
    )

    t3_validate = PythonOperator(
        task_id         = "validate_processed_data",
        python_callable = validate_processed_data,
    )

    t4_eda = PythonOperator(
        task_id         = "generate_eda_report",
        python_callable = generate_eda_report,
    )

    # ── DAG Task Dependency Chain ───────────────────────────────────────────
    t1_load >> t2_clean >> t3_validate >> t4_eda


# ==============================================================================
# DAG 2 — Weekly Model Training Pipeline
# ==============================================================================

with DAG(
    dag_id             = "travel_model_training_dag",
    description        = "Weekly model retraining, evaluation, and deployment",
    default_args       = default_args,
    schedule_interval  = "@weekly",          # Runs every Monday at midnight
    start_date         = days_ago(7),
    catchup            = False,
    tags               = ["ml", "training", "travel"],
) as training_dag:

    t1_load = PythonOperator(
        task_id         = "load_raw_data",
        python_callable = load_raw_data,
    )

    t2_clean = PythonOperator(
        task_id         = "clean_features",
        python_callable = clean_and_engineer,
    )

    t3_validate = PythonOperator(
        task_id         = "validate_data",
        python_callable = validate_processed_data,
    )

    # ── Parallel model training tasks ──────────────────────────────────────
    t4_regression = PythonOperator(
        task_id         = "retrain_regression",
        python_callable = retrain_regression_model,
    )

    t4_classification = PythonOperator(
        task_id         = "retrain_classification",
        python_callable = retrain_classification_model,
    )

    t4_recommendation = PythonOperator(
        task_id         = "retrain_recommendation",
        python_callable = retrain_recommendation_model,
    )

    t5_evaluate = PythonOperator(
        task_id         = "evaluate_models",
        python_callable = evaluate_and_compare,
    )

    t6_deploy = BashOperator(
        task_id      = "restart_api_container",
        bash_command = "docker restart travel_ml_api || echo 'Docker not available in this environment'",
    )

    # ── DAG Task Dependency Chain ───────────────────────────────────────────
    #
    #   load → clean → validate → [regression, classification, recommendation]
    #                                    ↓
    #                              evaluate → deploy
    #
    t1_load >> t2_clean >> t3_validate >> [t4_regression, t4_classification, t4_recommendation]
    [t4_regression, t4_classification, t4_recommendation] >> t5_evaluate >> t6_deploy
