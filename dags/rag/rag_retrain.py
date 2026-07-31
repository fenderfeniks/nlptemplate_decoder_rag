"""DAG: Обучение RAG-энкодера (Contrastive Learning)."""

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
    "rag_training_config",
    default_var={
        "schedule": "@weekly",
        "pvc_name": "model-weights-pvc",
        "mount_path": "/app/models",
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
    "rag_encoder_finetuning",
    default_args=make_default_args(**CONFIG["default_args"]),
    schedule=CONFIG["schedule"],
    catchup=False,
    tags=["nlp", "rag", "training"],
) as dag:
    model_vol, model_mount = make_pvc_volume(
        "model-weights", CONFIG["pvc_name"], CONFIG["mount_path"]
    )

    train_model_task = KubernetesPodOperator(
        task_id="run_contrastive_learning",
        name="rag-training-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "scripts.rag_pipeline.train"],
        service_account_name="airflow-worker-sa",
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources_gpu"]),
        env_from=COMMON_ENV_FROM,
        volume_mounts=[model_mount],
        volumes=[model_vol],
        get_logs=True,
        is_delete_operator_pod=True,
    )

    merge_weights_task = KubernetesPodOperator(
        task_id="merge_rag_lora",
        name="rag-merge-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "src.tools.merge_lora", "pipeline_name=rag_pipeline"],
        service_account_name="airflow-worker-sa",
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources_cpu"]),
        env_from=COMMON_ENV_FROM,
        volume_mounts=[model_mount],
        volumes=[model_vol],
        get_logs=True,
        is_delete_operator_pod=True,
    )

    evaluate_staging_task = KubernetesPodOperator(
        task_id="evaluate_retrieval",
        name="rag-eval-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        service_account_name="airflow-worker-sa",
        cmds=["python", "-m", "scripts.rag_pipeline.eval"],
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources_gpu"]),
        env_from=COMMON_ENV_FROM,
        volume_mounts=[model_mount],
        volumes=[model_vol],
        get_logs=True,
        is_delete_operator_pod=True,
    )

    request_approval = SlackWebhookOperator(
        task_id="request_manual_approval",
        slack_webhook_conn_id="slack_conn",
        message="✅ Обучение RAG-энкодера завершено. Модель ждёт в Staging. Проверьте метрики MRR.",
    )

    notify_failure = make_failure_slack_alert("notify_on_failure", "rag_encoder_finetuning")

    train_model_task >> merge_weights_task >> evaluate_staging_task >> request_approval
    evaluate_staging_task >> notify_failure
