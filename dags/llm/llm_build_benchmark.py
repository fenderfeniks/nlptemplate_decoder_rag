"""DAG: Построение эталонного SFT-бенчмарка для LLM Pipeline (ручной запуск)."""

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
)


CONFIG: dict[str, Any] = Variable.get(
    "llm_build_benchmark_config",
    default_var={
        "default_args": {"owner": "mlops", "retries": 0},
        "resources": {
            "requests": {"cpu": "1", "memory": "4Gi"},
            "limits": {"cpu": "2", "memory": "8Gi"},
        },
        # SFT-бенчмарк не требует GPU — только выборка и нормализация данных.
        # Данные берутся из HF Hub или внешнего источника через Fetcher,
        # поэтому PVC с сырыми данными не нужен.
    },
    deserialize_json=True,
)

with DAG(
    "llm_build_benchmark",
    default_args=make_default_args(**CONFIG["default_args"]),
    schedule=None,  # Только ручной запуск
    catchup=False,
    tags=["nlp", "llm", "benchmark"],
) as dag:
    build_benchmark = KubernetesPodOperator(
        task_id="build_llm_benchmark",
        name="llm-build-benchmark-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "scripts.decoder_pipeline.build_benchmark"],
        service_account_name="airflow-worker-sa",
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources"]),
        env_from=COMMON_ENV_FROM,
        # Данные тянутся из HF Hub или S3 через Fetcher — PVC не нужен.
        # Результат загружается в S3 и манифест обновляется через StorageRouter.
        get_logs=True,
        is_delete_operator_pod=True,
    )

    notify_success = SlackWebhookOperator(
        task_id="notify_benchmark_ready",
        slack_webhook_conn_id="slack_conn",
        message="✅ LLM SFT-бенчмарк успешно построен и загружен в S3.",
        trigger_rule="all_success",
    )

    notify_failure = make_failure_slack_alert("notify_on_failure", "llm_build_benchmark")

    build_benchmark >> notify_success
    build_benchmark >> notify_failure
