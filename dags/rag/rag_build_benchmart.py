"""DAG: Построение эталонного бенчмарка для RAG Pipeline (ручной запуск)."""

from typing import Any

from airflow import DAG
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
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
    "rag_build_benchmark_config",
    default_var={
        "default_args": {"owner": "mlops", "retries": 0},
        "resources": {
            "requests": {"cpu": "2", "memory": "8Gi"},
            "limits": {"cpu": "4", "memory": "16Gi", "nvidia.com/gpu": "1"},
        },
        # Данные для чанкинга берутся с PVC.
        # Веса генератора и NLI-судьи тянутся из S3 через StorageRouter.
        "data_pvc_name": "raw-data-pvc",
        "data_mount_path": "/app/data",
    },
    deserialize_json=True,
)

with DAG(
    "rag_build_benchmark",
    default_args=make_default_args(**CONFIG["default_args"]),
    schedule=None,  # Только ручной запуск
    catchup=False,
    tags=["nlp", "rag", "benchmark"],
) as dag:
    data_vol, data_mount = make_pvc_volume(
        "raw-data", CONFIG["data_pvc_name"], CONFIG["data_mount_path"]
    )

    build_benchmark = KubernetesPodOperator(
        task_id="build_rag_benchmark",
        name="rag-build-benchmark-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "scripts.rag_pipeline.build_benchmark"],
        service_account_name="airflow-worker-sa",
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources"]),
        env_from=COMMON_ENV_FROM,
        # Данные для чанкинга с PVC.
        # Веса NLI-судьи и генератора — из S3 через StorageRouter.
        volume_mounts=[data_mount],
        volumes=[data_vol],
        get_logs=True,
        is_delete_operator_pod=True,
    )

    notify_success = SlackWebhookOperator(
        task_id="notify_benchmark_ready",
        slack_webhook_conn_id="slack_conn",
        message="✅ RAG бенчмарк успешно построен и загружен в S3. Можно запускать e2e eval.",
        trigger_rule="all_success",
    )

    notify_failure = make_failure_slack_alert("notify_on_failure", "rag_build_benchmark")

    build_benchmark >> notify_success
    build_benchmark >> notify_failure
