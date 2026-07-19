"""
DAG: RAG Knowledge Base Update
"""

import pendulum
from airflow import DAG
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s


# 1. ИНФРАСТРУКТУРА
# ИСПРАВЛЕНИЕ: Меняем тег на trainer-latest для работы с библиотеками индексации
IMAGE = Variable.get(
    "PROJECT_IMAGE", default_var="my-company/industrial_nlp_template:trainer-latest"
)
NAMESPACE = Variable.get("K8S_NAMESPACE", default_var="ml-pipelines")

# ИСПРАВЛЕНИЕ: Защитный словарь-заглушка для парсера
DEFAULT_CONFIG = {
    "schedule": "@daily",
    "default_args": {"owner": "mlops", "retries": 1, "retry_delay_minutes": 5},
    "resources": {
        "requests": {"cpu": "1", "memory": "4Gi"},
        "limits": {"cpu": "2", "memory": "8Gi"},
    },
    "mount_path": "/app/data",
    "pvc_name": "pvc-data",
}

# 2. БИЗНЕС-ЛОГИКА
# ИСПРАВЛЕНИЕ: Передаем default_var для предотвращения ошибки отсутствия переменной
CONFIG = Variable.get("etl_config", default_var=DEFAULT_CONFIG, deserialize_json=True)

default_args = {
    "owner": CONFIG["default_args"]["owner"],
    "depends_on_past": False,
    "start_date": pendulum.datetime(2026, 1, 1, tz="UTC"),
    "email_on_failure": True,
    "retries": CONFIG["default_args"]["retries"],
    "retry_delay": pendulum.duration(minutes=CONFIG["default_args"]["retry_delay_minutes"]),
}

with DAG(
    "rag_knowledge_base_update",
    default_args=default_args,
    schedule_interval=CONFIG["schedule"],
    catchup=False,
    max_active_runs=1,
    tags=["nlp", "rag", "etl"],
) as dag:
    update_rag_task = KubernetesPodOperator(
        task_id="run_rag_update_job",
        name="rag-etl-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "src.jobs.rag_update_job"],
        # ИСПРАВЛЕНИЕ: Добавлен сервисный аккаунт
        service_account_name="airflow-worker-sa",
        arguments=[
            f"paths.data_dir={CONFIG['mount_path']}",
            f"rag.indexer.persist_dir={CONFIG['mount_path']}/processed/vector_db",
        ],
        env_vars=[
            k8s.V1EnvVar(
                name="HUGGINGFACE_TOKEN",
                value_from=k8s.V1EnvVarSource(
                    secret_key_ref=k8s.V1SecretKeySelector(
                        name="hf-secrets",
                        key="token",
                        optional=True,
                    )
                ),
            ),
        ],
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources"]),
        volume_mounts=[k8s.V1VolumeMount(name="data-volume", mount_path=CONFIG["mount_path"])],
        volumes=[
            k8s.V1Volume(
                name="data-volume",
                persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                    claim_name=CONFIG["pvc_name"]
                ),
            )
        ],
        get_logs=True,
        is_delete_operator_pod=True,
    )
