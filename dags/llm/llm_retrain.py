"""DAG: Регулярное дообучение LLM (SFT) и слияние LoRA-адаптеров."""

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
    "llm_training_config",
    default_var={
        "schedule": "@weekly",
        "resources_gpu": {
            "requests": {"cpu": "2", "memory": "4Gi"},
            "limits": {"cpu": "4", "memory": "7Gi", "nvidia.com/gpu": "1"},
        },
        "resources_cpu": {
            "requests": {"cpu": "2", "memory": "4Gi"},
            "limits": {"cpu": "4", "memory": "7Gi"},
        },
        "default_args": {"owner": "mlops", "retries": 1, "retry_delay_minutes": 5},
    },
    deserialize_json=True,
)

with DAG(
    "llm_weekly_finetuning",
    default_args=make_default_args(**CONFIG["default_args"]),
    schedule=CONFIG["schedule"],
    catchup=False,
    tags=["nlp", "llm", "training"],
) as dag:
    # Веса модели тянутся из S3 через StorageRouter — PVC не нужен.

    train_model_task = KubernetesPodOperator(
        task_id="run_lora_finetuning",
        name="llm-training-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "scripts.decoder_pipeline.train"],
        service_account_name="airflow-worker-sa",
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources_gpu"]),
        env_from=COMMON_ENV_FROM,
        get_logs=True,
        is_delete_operator_pod=True,
    )

    # promote: продвигает лучший адаптер из MLflow Staging → Production
    # и загружает его в S3, обновляет манифест (load_type: lora).
    promote_task = KubernetesPodOperator(
        task_id="promote_adapter_to_storage",
        name="llm-promote-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "src.tools.promote", "pipeline_name=decoder_pipeline"],
        service_account_name="airflow-worker-sa",
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources_cpu"]),
        env_from=COMMON_ENV_FROM,
        get_logs=True,
        is_delete_operator_pod=True,
    )

    # merge_lora: мерджит Production LoRA + базу → монолит, кладёт в S3,
    # обновляет манифест (load_type: full_model).
    merge_weights_task = KubernetesPodOperator(
        task_id="merge_lora_weights",
        name="llm-merge-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "src.tools.merge_lora", "pipeline_name=decoder_pipeline"],
        service_account_name="airflow-worker-sa",
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources_cpu"]),
        env_from=COMMON_ENV_FROM,
        get_logs=True,
        is_delete_operator_pod=True,
    )

    evaluate_staging_task = KubernetesPodOperator(
        task_id="evaluate_staging_model",
        name="llm-eval-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        service_account_name="airflow-worker-sa",
        cmds=["python", "-m", "scripts.decoder_pipeline.eval"],
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources_gpu"]),
        env_from=COMMON_ENV_FROM,
        get_logs=True,
        is_delete_operator_pod=True,
    )

    request_approval = SlackWebhookOperator(
        task_id="request_manual_approval",
        slack_webhook_conn_id="slack_conn",
        message="✅ Обучение LLM завершено. Монолитная модель ждёт в Staging. Проверьте MLflow.",
    )

    notify_failure = make_failure_slack_alert("notify_on_failure", "llm_weekly_finetuning")

    (
        train_model_task
        >> promote_task
        >> merge_weights_task
        >> evaluate_staging_task
        >> request_approval
    )
    evaluate_staging_task >> notify_failure
