from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.empty import EmptyOperator

from pipeline_tasks.external_scraping import run as external_scraping_run
from pipeline_tasks.feature_engineering import run as feature_engineering_run
from pipeline_tasks.ml_training import run as ml_training_run
from pipeline_tasks.prep import run as prep_run


def check_model_exists(**context):
    """Check if model already exists, skip training if it does."""
    model_path = Path("/opt/airflow/models/traffy_rf_model.joblib")
    
    if model_path.exists():
        print(f"✓ Model already exists at {model_path}")
        print("Skipping training pipeline. To retrain, manually trigger this DAG.")
        return "skip_training"
    else:
        print(f"✗ Model not found at {model_path}")
        print("Starting training pipeline...")
        return "prep_data"


with DAG(
    dag_id="traffy_model_pipeline",
    description="End-to-end Traffy prep -> external -> feature -> training pipeline",
    start_date=datetime(2023, 1, 1),  # Past date to trigger immediately
    schedule="@once",  # Run exactly once on startup
    catchup=True,  # Enable catchup to trigger on first startup
    default_args={"owner": "data-platform"},
    max_active_runs=1,
) as dag:
    
    # Check if model exists before running pipeline
    check_model = BranchPythonOperator(
        task_id="check_model_exists",
        python_callable=check_model_exists,
    )
    
    # Skip task if model exists
    skip_training = EmptyOperator(
        task_id="skip_training",
    )
    
    prep_task = PythonOperator(
        task_id="prep_data",
        python_callable=prep_run,
        do_xcom_push=False,  # avoid serializing Path
    )

    external_task = PythonOperator(
        task_id="external_scraping",
        python_callable=external_scraping_run,
        do_xcom_push=False,
    )

    feature_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering_run,
        do_xcom_push=False,
    )

    training_task = PythonOperator(
        task_id="ml_training",
        python_callable=ml_training_run,
        do_xcom_push=False,
    )

    # Workflow: Check model -> either skip or run full pipeline
    check_model >> [skip_training, prep_task]
    prep_task >> external_task >> feature_task >> training_task