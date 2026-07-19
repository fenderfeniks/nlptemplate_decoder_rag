"""
DAG: Batch Analytics (LLM-as-an-Analyst)
"""

import pendulum
from airflow import DAG
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s


# 1. ИНФРАСТРУКТУРА
# ИСПРАВЛЕНИЕ: Меняем тег на trainer-latest для батч-задач
IMAGE = Variable.get(
    "PROJECT_IMAGE", default_var="my-company/industrial_nlp_template:trainer-latest"
)
NAMESPACE = Variable.get("K8S_NAMESPACE", default_var="ml-pipelines")

# ИСПРАВЛЕНИЕ: Защитный словарь-заглушка от краша парсера Airflow
DEFAULT_CONFIG = {
    "schedule": "@daily",
    "default_args": {"owner": "mlops", "retries": 1, "retry_delay_minutes": 5},
    "resources": {
        "requests": {"cpu": "1", "memory": "4Gi"},
        "limits": {"cpu": "2", "memory": "8Gi"},  # Для батч-задач можно без GPU или с 1 GPU
    },
    "db_secret_name": "db-secrets",
}

# 2. БИЗНЕС-ЛОГИКА
# ИСПРАВЛЕНИЕ: Передаем default_var
CONFIG = Variable.get("analytics_config", default_var=DEFAULT_CONFIG, deserialize_json=True)

# 3. НАСТРОЙКИ ОТКАЗОУСТОЙЧИВОСТИ
default_args = {
    "owner": CONFIG["default_args"]["owner"],
    "depends_on_past": False,
    "start_date": pendulum.datetime(2026, 1, 1, tz="UTC"),
    "email_on_failure": True,
    "retries": CONFIG["default_args"]["retries"],
    "retry_delay": pendulum.duration(minutes=CONFIG["default_args"]["retry_delay_minutes"]),
}

with DAG(
    "batch_analytics_reporting",
    default_args=default_args,
    schedule_interval=CONFIG["schedule"],
    catchup=False,
    tags=["nlp", "analytics"],
) as dag:
    analyze_reviews = KubernetesPodOperator(
        task_id="run_batch_inference",
        name="analytics-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "src.jobs.batch_analytics"],
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources"]),
        # ИСПРАВЛЕНИЕ: Подключаем сервисный аккаунт с правами RBAC
        service_account_name="airflow-worker-sa",
        env_vars=[
            # БЕЗОПАСНАЯ ПЕРЕДАЧА СЕКРЕТА ИЗ K8S
            k8s.V1EnvVar(
                name="DB_CONN",
                value_from=k8s.V1EnvVarSource(
                    secret_key_ref=k8s.V1SecretKeySelector(
                        name=CONFIG["db_secret_name"], key="connection-string"
                    )
                ),
            )
        ],
        get_logs=True,
        is_delete_operator_pod=True,
    )
