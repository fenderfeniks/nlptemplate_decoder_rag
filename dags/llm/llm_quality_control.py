"""DAG: Model Evaluation & Quality Drift Detection (Generative LLM)."""

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
    "llm_evaluation_config",
    default_var={
        "schedule": "@weekly",
        "rouge1_threshold": 0.45,
        "resources": {
            "requests": {"cpu": "1", "memory": "4Gi"},
            "limits": {"cpu": "2", "memory": "7Gi", "nvidia.com/gpu": "1"},
        },
        "default_args": {"owner": "mlops", "retries": 1, "retry_delay_minutes": 5},
    },
    deserialize_json=True,
)

with DAG(
    "llm_quality_drift_detection",
    default_args=make_default_args(**CONFIG["default_args"]),  # [cite: 32]
    schedule=CONFIG["schedule"],
    catchup=False,
    tags=["nlp", "monitoring", "llm"],
) as dag:
    # ВАЖНО: Скрипт scripts.decoder_pipeline.eval ДОЛЖЕН падать (sys.exit(1)),
    # если метрика не проходит порог, чтобы Airflow зарегистрировал failure.
    evaluate_model = KubernetesPodOperator(
        task_id="evaluate_llm",
        name="evaluator-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "scripts.decoder_pipeline.eval"],
        arguments=[
            f"metric_thresholds.rouge1={CONFIG['rouge1_threshold']}",
            "model.builder.mlflow_alias=Staging",
        ],
        env_from=COMMON_ENV_FROM,
        service_account_name="airflow-worker-sa",
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources"]),
        get_logs=True,
        is_delete_operator_pod=True,
    )

    notify_drift = SlackWebhookOperator(
        task_id="alert_if_drift",
        slack_webhook_conn_id="slack_conn",
        message=f"Внимание! Качество генерации (ROUGE-1) упало ниже {int(CONFIG['rouge1_threshold'] * 100)}%.",
        trigger_rule="one_failed",
    )

    evaluate_model >> notify_drift
