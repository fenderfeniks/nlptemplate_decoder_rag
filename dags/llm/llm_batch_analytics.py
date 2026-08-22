"""DAG: Batch Evaluation & Reporting (LLM Pipeline)."""

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
)


CONFIG: dict[str, Any] = Variable.get(
    "llm_analytics_config",
    default_var={
        "schedule": "@daily",
        "default_args": {"owner": "mlops", "retries": 1, "retry_delay_minutes": 5},
        "resources": {
            "requests": {"cpu": "1", "memory": "4Gi"},
            "limits": {"cpu": "2", "memory": "7Gi", "nvidia.com/gpu": "1"},
        },
    },
    deserialize_json=True,
)

with DAG(
    "llm_batch_analytics_reporting",
    default_args=make_default_args(**CONFIG["default_args"]),
    schedule=CONFIG["schedule"],
    catchup=False,
    tags=["nlp", "analytics", "llm"],
) as dag:
    # Используем eval.py вместо batch_analytics.py:
    # eval реализует полный прогон метрик по бенчмарку (ROUGE, BERTScore и др.)
    # и логирует результаты в MLflow — этого достаточно для ежедневного репорта.
    run_eval = KubernetesPodOperator(
        task_id="run_llm_eval",
        name="llm-analytics-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "scripts.decoder_pipeline.eval"],
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources"]),
        service_account_name="airflow-worker-sa",
        env_from=COMMON_ENV_FROM,
        get_logs=True,
        is_delete_operator_pod=True,
    )

    notify_failure = make_failure_slack_alert("notify_on_failure", "llm_batch_analytics_reporting")
    run_eval >> notify_failure
