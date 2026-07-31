"""DAG: Сбор новых данных для базы знаний (Data Ingestion)."""

from typing import Any

from airflow import DAG
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

from dags.common import (
    COMMON_ENV_FROM,
    IMAGE,
    NAMESPACE,
    make_default_args,
    make_failure_slack_alert,
    make_pvc_volume,
)


CONFIG: dict[str, Any] = Variable.get(
    "rag_fetch_config",
    default_var={
        "schedule": "@hourly",
        "default_args": {"owner": "mlops", "retries": 1, "retry_delay_minutes": 5},
        "resources": {
            "requests": {"cpu": "1", "memory": "4Gi"},
            "limits": {"cpu": "2", "memory": "8Gi"},
        },
        "data_pvc_name": "raw-data-pvc",
        "data_mount_path": "/app/data",
    },
    deserialize_json=True,
)

with DAG(
    "rag_data_ingestion",
    default_args=make_default_args(**CONFIG["default_args"]),
    schedule=CONFIG["schedule"],
    catchup=False,
    tags=["nlp", "rag", "data_engineering"],
) as dag:
    data_vol, data_mount = make_pvc_volume(
        "raw-data", CONFIG["data_pvc_name"], CONFIG["data_mount_path"]
    )

    fetch_new_data = KubernetesPodOperator(
        task_id="fetch_data_from_sources",
        name="rag-fetch-data-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "src.tools.fetch_data", "pipeline_name=rag_pipeline"],
        service_account_name="airflow-worker-sa",
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources"]),
        env_from=COMMON_ENV_FROM,
        volume_mounts=[data_mount],
        volumes=[data_vol],
        get_logs=True,
        is_delete_operator_pod=True,
    )

    notify_failure = make_failure_slack_alert("notify_on_failure", "rag_data_ingestion")
    fetch_new_data >> notify_failure
