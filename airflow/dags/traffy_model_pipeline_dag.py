from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from pipeline_tasks.external_scraping import run as external_scraping_run
from pipeline_tasks.feature_engineering import run as feature_engineering_run
from pipeline_tasks.ml_training import run as ml_training_run
from pipeline_tasks.prep import run as prep_run
from pipeline_tasks.launch_app import run as launch_app_run


with DAG(
    dag_id="traffy_model_pipeline",
    description="End-to-end Traffy prep -> external -> feature -> training pipeline",
    start_date=datetime(2025, 12, 6),
    schedule=None,
    catchup=False,
    default_args={"owner": "data-platform"},
    max_active_runs=1,
) as dag:
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

    app_task = PythonOperator(
        task_id="launch_streamlit_app",
        python_callable=launch_app_run,
        do_xcom_push=False,
    )

    prep_task >> external_task >> feature_task >> training_task >> app_task