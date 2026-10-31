
from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

PROJECT_DIR = "/mnt/d/Projects/Ecommerce-Sales-Analytics"
PYTHON = f"{PROJECT_DIR}/venv/Scripts/python.exe"
PASSWORD_FILE = "/home/patee/airflow/.postgres_password"

with DAG(
    dag_id="ecommerce_pipeline",
    start_date=datetime(2026, 9, 27),
    schedule="@daily",
    catchup=False,
) as dag:

    extract = BashOperator(
        task_id="extract",
        bash_command=f"""
            set -e
            test -f "{PROJECT_DIR}/scripts/extract.py"
            test -f "{PROJECT_DIR}/data/raw/Online Retail.xlsx"
            cd "{PROJECT_DIR}"
            "{PYTHON}" scripts/extract.py
        """,
    )

    transform = BashOperator(
        task_id="transform",
        bash_command=f"""
            set -e
            cd "{PROJECT_DIR}"
            "{PYTHON}" scripts/transform.py
        """,
    )

    load_rejected = BashOperator(
        task_id="load_rejected",
        bash_command=f"""
            set -e
            test -f "{PASSWORD_FILE}"
            cd "{PROJECT_DIR}"
            export POSTGRES_PASSWORD="$(cat "{PASSWORD_FILE}")"
            "{PYTHON}" scripts/load_rejected.py
        """,
    )

    load = BashOperator(
        task_id="load",
        bash_command=f"""
            set -e
            test -f "{PASSWORD_FILE}"
            cd "{PROJECT_DIR}"
            export POSTGRES_PASSWORD="$(cat "{PASSWORD_FILE}")"
            "{PYTHON}" scripts/load.py
        """,
    )

    extract >> transform
    transform >> [load_rejected, load]

