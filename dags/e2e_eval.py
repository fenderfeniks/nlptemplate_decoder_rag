"""DAG: End-to-End оценка системы через API Gateway (ручной запуск)."""

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
    "e2e_eval_config",
    default_var={
        "default_args": {"owner": "mlops", "retries": 0},
        "resources": {
            "requests": {"cpu": "1", "memory": "2Gi"},
            "limits": {"cpu": "2", "memory": "4Gi"},
        },
        # URL живого Gateway внутри кластера.
        # Переопределяется через Airflow Variable или env в ConfigMap.
        "gateway_url": "http://nlp-template-api-gateway.ml-pipelines.svc.cluster.local:8000",
        # Пайплайн определяет какую секцию манифеста читать для бенчмарка.
        "pipeline_name": "rag_pipeline",
        # Пороги — при нарушении скрипт завершается с exit(1) и DAG падает.
        "rouge_threshold": 0.2,
        "latency_p95_max_s": 30.0,
        "max_samples": 50,
        "request_timeout_s": 60.0,
    },
    deserialize_json=True,
)

with DAG(
    "end2end_eval",
    default_args=make_default_args(**CONFIG["default_args"]),
    schedule=None,  # Только ручной запуск — после deploy или promote
    catchup=False,
    tags=["nlp", "e2e", "monitoring"],
) as dag:
    # Скрипт сам делает health check перед прогоном —
    # отдельный таск не нужен, это упрощает граф и уменьшает накладные расходы.
    run_e2e = KubernetesPodOperator(
        task_id="run_e2e_eval",
        name="e2e-eval-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "scripts.e2e_eval"],
        service_account_name="airflow-worker-sa",
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources"]),
        env_from=COMMON_ENV_FROM,
        env_vars=[
            # Gateway URL — сервис внутри кластера, не нужен внешний ingress
            k8s.V1EnvVar(name="GATEWAY_URL", value=CONFIG["gateway_url"]),
            k8s.V1EnvVar(name="PIPELINE_NAME", value=CONFIG["pipeline_name"]),
            k8s.V1EnvVar(name="ROUGE_THRESHOLD", value=str(CONFIG["rouge_threshold"])),
            k8s.V1EnvVar(name="LATENCY_P95_MAX_S", value=str(CONFIG["latency_p95_max_s"])),
            k8s.V1EnvVar(name="MAX_SAMPLES", value=str(CONFIG["max_samples"])),
            k8s.V1EnvVar(name="REQUEST_TIMEOUT_S", value=str(CONFIG["request_timeout_s"])),
        ],
        # Нет PVC — бенчмарк скачивается из S3, запросы идут в Gateway по HTTP.
        get_logs=True,
        is_delete_operator_pod=True,
    )

    notify_success = SlackWebhookOperator(
        task_id="notify_e2e_passed",
        slack_webhook_conn_id="slack_conn",
        message=(
            f"✅ E2E eval пройден. "
            f"ROUGE-1 ≥ {CONFIG['rouge_threshold']}, "
            f"latency p95 ≤ {CONFIG['latency_p95_max_s']}s. "
            f"Система готова к продакшену."
        ),
        trigger_rule="all_success",
    )

    notify_failure = make_failure_slack_alert("notify_on_failure", "end2end_eval")

    run_e2e >> notify_success
    run_e2e >> notify_failure
