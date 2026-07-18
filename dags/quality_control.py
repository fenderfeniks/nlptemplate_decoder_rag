"""
DAG: Model Evaluation & Drift Detection
"""

import pendulum
from airflow import DAG
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from kubernetes.client import models as k8s


# 1. ИНФРАСТРУКТУРА
IMAGE = Variable.get("PROJECT_IMAGE", default_var="my-company/industrial_nlp_template:latest")
NAMESPACE = Variable.get("K8S_NAMESPACE", default_var="ml-pipelines")

# 2. БИЗНЕС-ЛОГИКА
CONFIG = Variable.get("evaluation_config", deserialize_json=True)

# 3. НАСТРОЙКИ ОТКАЗОУСТОЙЧИВОСТИ
default_args = {
    "owner": CONFIG["default_args"]["owner"],
    "depends_on_past": False,
    "start_date": pendulum.datetime(2026, 1, 1, tz="UTC"),
    "email_on_failure": True,
    "retries": CONFIG["default_args"]["retries"],
    "retry_delay": pendulum.duration(minutes=CONFIG["default_args"]["retry_delay_minutes"]),
}

with DAG(
    "model_drift_detection",
    default_args=default_args,
    schedule_interval=CONFIG["schedule"],
    catchup=False,
    tags=["nlp", "monitoring"],
) as dag:
    # 1. Запуск оценки
    evaluate_model = KubernetesPodOperator(
        task_id="evaluate_model",
        name="evaluator-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "src.evaluate"],
        arguments=[f"ckpt_path={CONFIG['mount_path']}/best.ckpt"],
        # Подключаем GPU и память из JSON
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources"]),
        # Монтируем диск с моделями, чтобы скрипт увидел /app/models/best.ckpt
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

    # 2. Динамический алерт
    # Переводим порог из долей в проценты (0.9 -> 90%)
    threshold_percent = int(CONFIG["drift_threshold"] * 100)

    notify_slack = SlackWebhookOperator(
        task_id="alert_if_drift",
        slack_webhook_conn_id="slack_conn",
        message=f"⚠️ Внимание! Качество модели упало ниже порога {threshold_percent}%. Нужен ретрейн.",
        trigger_rule="one_failed",
    )

    evaluate_model >> notify_slack
