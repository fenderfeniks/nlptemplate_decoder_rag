"""DAG: Batch Analytics (RAG Pipeline)."""

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
    "rag_analytics_config",
    default_var={
        "schedule": "@daily",
        "default_args": {"owner": "mlops", "retries": 1, "retry_delay_minutes": 5},
        "resources": {
            "requests": {"cpu": "1", "memory": "4Gi"},
            "limits": {"cpu": "2", "memory": "7Gi", "nvidia.com/gpu": "1"},
        },
        "db_secret_name": "db-secrets",
    },
    deserialize_json=True,
)

with DAG(
    "rag_batch_analytics_reporting",
    default_args=make_default_args(**CONFIG["default_args"]),
    schedule=CONFIG["schedule"],
    catchup=False,
    tags=["nlp", "analytics", "rag"],
) as dag:
    run_batch_inference = KubernetesPodOperator(
        task_id="run_batch_inference",
        name="rag-analytics-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "src.tools.batch_analytics", "pipeline_name=rag_pipeline"],
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources"]),
        service_account_name="airflow-worker-sa",
        env_from=COMMON_ENV_FROM,
        env_vars=[
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

    notify_failure = make_failure_slack_alert("notify_on_failure", "rag_batch_analytics_reporting")
    run_batch_inference >> notify_failure
