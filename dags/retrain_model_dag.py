# dags/retrain_model_dag.py
"""
DAG: Еженедельное переобучение модели (Continuous Finetuning).
"""

import pendulum
from airflow import DAG
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from kubernetes.client import models as k8s


# ИНФРАСТРУКТУРА
IMAGE = Variable.get(
    "PROJECT_IMAGE", default_var="my-company/industrial_nlp_template:trainer-latest"
)
NAMESPACE = Variable.get("K8S_NAMESPACE", default_var="ml-pipelines")

# Защитный словарь-заглушка
DEFAULT_CONFIG = {
    "schedule": "@weekly",
    "default_args": {"owner": "mlops", "retries": 1, "retry_delay_minutes": 5},
    "resources": {
        "requests": {"cpu": "2", "memory": "8Gi"},
        "limits": {"cpu": "4", "memory": "16Gi", "nvidia.com/gpu": "1"},
    },
    "mount_path": "/app/models",
    "pvc_name": "model-weights-pvc",
}

CONFIG = Variable.get("training_config", default_var=DEFAULT_CONFIG, deserialize_json=True)

default_args = {
    "owner": CONFIG["default_args"]["owner"],
    "depends_on_past": False,
    "start_date": pendulum.datetime(2026, 1, 1, tz="UTC"),
    "email_on_failure": True,
    "retries": CONFIG["default_args"]["retries"],
    "retry_delay": pendulum.duration(minutes=CONFIG["default_args"]["retry_delay_minutes"]),
}

with DAG(
    "weekly_llm_finetuning",
    default_args=default_args,
    schedule_interval=CONFIG["schedule"],
    catchup=False,
    tags=["nlp", "training"],
) as dag:
    train_model_task = KubernetesPodOperator(
        task_id="run_lora_finetuning",
        name="llm-trainer-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "src.train"],
        # ИСПРАВЛЕНИЕ: Добавлен service_account_name
        service_account_name="airflow-worker-sa",
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources"]),
        env_vars=[
            k8s.V1EnvVar(
                name="HUGGINGFACE_TOKEN",
                value_from=k8s.V1EnvVarSource(
                    secret_key_ref=k8s.V1SecretKeySelector(
                        name="hf-secrets", key="token", optional=True
                    )
                ),
            ),
            k8s.V1EnvVar(
                name="WANDB_API_KEY",
                value_from=k8s.V1EnvVarSource(
                    secret_key_ref=k8s.V1SecretKeySelector(
                        name="wandb-secrets", key="api-key", optional=True
                    )
                ),
            ),
        ],
        volume_mounts=[k8s.V1VolumeMount(name="model-weights", mount_path=CONFIG["mount_path"])],
        volumes=[
            k8s.V1Volume(
                name="model-weights",
                persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                    claim_name=CONFIG["pvc_name"]
                ),
            )
        ],
        get_logs=True,
        is_delete_operator_pod=True,
    )

    evaluate_staging_task = KubernetesPodOperator(
        task_id="evaluate_staging_model",
        name="llm-eval-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        # ИСПРАВЛЕНИЕ: Добавлен service_account_name
        service_account_name="airflow-worker-sa",
        cmds=["python", "-m", "src.eval"],
        arguments=[f"ckpt_path={CONFIG['mount_path']}/staging/best.ckpt"],
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources"]),
        volume_mounts=[k8s.V1VolumeMount(name="model-weights", mount_path=CONFIG["mount_path"])],
        volumes=[
            k8s.V1Volume(
                name="model-weights",
                persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                    claim_name=CONFIG["pvc_name"]
                ),
            )
        ],
        get_logs=True,
        is_delete_operator_pod=True,
    )

    request_approval = SlackWebhookOperator(
        task_id="request_manual_approval",
        slack_webhook_conn_id="slack_conn",
        message="✅ Обучение завершено. Метрики посчитаны. Чекпоинт ждет в папке Staging.\n"
        "👉 Проверьте MLflow. Если качество устраивает, запустите DAG `promote_to_prod`.",
    )

    train_model_task >> evaluate_staging_task >> request_approval
