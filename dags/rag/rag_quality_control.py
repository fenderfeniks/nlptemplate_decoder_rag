"""DAG: Model Evaluation & Quality Drift Detection (RAG Encoder)."""

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
    "rag_evaluation_config",
    default_var={
        "schedule": "@weekly",
        "mrr_threshold": 0.75,
        "resources": {
            "requests": {"cpu": "1", "memory": "4Gi"},
            "limits": {"cpu": "2", "memory": "7Gi", "nvidia.com/gpu": "1"},
        },
        "default_args": {"owner": "mlops", "retries": 1, "retry_delay_minutes": 5},
    },
    deserialize_json=True,
)

with DAG(
    "rag_quality_drift_detection",
    default_args=make_default_args(**CONFIG["default_args"]),
    schedule=CONFIG["schedule"],
    catchup=False,
    tags=["nlp", "monitoring", "rag"],
) as dag:
    evaluate_model = KubernetesPodOperator(
        task_id="evaluate_rag_encoder",
        name="rag-evaluator-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "scripts.rag_pipeline.eval"],
        arguments=[
            f"metric_thresholds.val_mrr={CONFIG['mrr_threshold']}",
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
        message=(
            f"Внимание! Качество поиска (MRR) упало ниже порога "
            f"{int(CONFIG['mrr_threshold'] * 100)}%. Требуется дообучение энкодера."
        ),
        trigger_rule="one_failed",
    )

    notify_failure = make_failure_slack_alert("notify_on_failure", "rag_quality_drift_detection")

    evaluate_model >> [notify_drift, notify_failure]
