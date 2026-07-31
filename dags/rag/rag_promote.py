"""DAG: Model Promotion — RAG (Manual Approval Gate)."""

from typing import Any

from airflow import DAG
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator

from dags.common import (
    API_IMAGE,
    COMMON_ENV_FROM,
    NAMESPACE,
    make_default_args,
    make_failure_slack_alert,
)


CONFIG: dict[str, Any] = Variable.get(
    "rag_promotion_config",
    default_var={
        "default_args": {"owner": "mlops", "retries": 0},
        "deployment_name": "nlp-template-rag-api",
        "configmap_name": "nlp-template-api-config",
    },
    deserialize_json=True,
)

with DAG(
    "promote_rag_to_prod",
    default_args=make_default_args(**CONFIG["default_args"]),
    schedule=None,  # Только ручной запуск
    catchup=False,
    tags=["nlp", "production", "rag"],
) as dag:
    promote_model = KubernetesPodOperator(
        task_id="promote_staging_to_prod",
        name="rag-promote-model-pod",
        namespace=NAMESPACE,
        image=API_IMAGE,
        cmds=["python", "-m", "src.tools.promote", "pipeline_name=rag_pipeline"],
        service_account_name="airflow-worker-sa",
        env_from=COMMON_ENV_FROM,
        get_logs=True,
        is_delete_operator_pod=True,
    )

    restart_api = KubernetesPodOperator(
        task_id="restart_rag_api_deployment",
        name="restart-rag-api-pod",
        namespace=NAMESPACE,
        image="bitnami/kubectl:1.29",
        # Исправлено #5: Ожидание завершения перезапуска
        cmds=["sh", "-c"],
        arguments=[
            f"kubectl rollout restart deployment/{CONFIG['deployment_name']} -n {NAMESPACE} "
            f"&& kubectl rollout status deployment/{CONFIG['deployment_name']} -n {NAMESPACE} --timeout=300s"
        ],
        get_logs=True,
        is_delete_operator_pod=True,
        service_account_name="airflow-worker-sa",
    )

    notify_failure = make_failure_slack_alert("notify_on_failure", "promote_rag_to_prod")

    promote_model >> restart_api >> notify_failure
