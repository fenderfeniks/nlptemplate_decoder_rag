# dags/promote_to_prod.py
"""
DAG: Model Promotion (Manual Approval Gate).
Переводит модель из Staging в Production и перезапускает API.
"""

import pendulum
from airflow import DAG
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s


# ИСПРАВЛЕНИЕ: Меняем дефолтный тег на api-latest
IMAGE = Variable.get("PROJECT_IMAGE", default_var="my-company/industrial_nlp_template:api-latest")
NAMESPACE = Variable.get("K8S_NAMESPACE", default_var="ml-pipelines")

# ИСПРАВЛЕНИЕ: Защитный словарь-заглушка от краша парсера Airflow
DEFAULT_CONFIG = {
    "default_args": {
        "owner": "mlops",
    },
    "mount_path": "/app/models",
    "pvc_name": "model-weights-pvc",
}

# ИСПРАВЛЕНИЕ: Передаем default_var
CONFIG = Variable.get("training_config", default_var=DEFAULT_CONFIG, deserialize_json=True)

default_args = {
    "owner": CONFIG["default_args"]["owner"],
    "depends_on_past": False,
    "start_date": pendulum.datetime(2026, 1, 1, tz="UTC"),
}

with DAG(
    "promote_to_prod",
    default_args=default_args,
    schedule_interval=None,  # СТРОГО РУЧНОЙ ЗАПУСК
    catchup=False,
    tags=["nlp", "mlops", "production"],
) as dag:
    promote_weights = KubernetesPodOperator(
        task_id="copy_weights_to_prod",
        name="promote-weights-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["bash", "-c"],
        # ИСПРАВЛЕНИЕ: Безопасное копирование с созданием директории prod
        arguments=[
            f"mkdir -p {CONFIG['mount_path']}/prod && cp {CONFIG['mount_path']}/staging/best.ckpt {CONFIG['mount_path']}/prod/best.ckpt && echo 'Weights promoted!'"
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
        # ИСПРАВЛЕНИЕ: Подключаем сервисный аккаунт
        service_account_name="airflow-worker-sa",
    )

    restart_api = KubernetesPodOperator(
        task_id="restart_api_deployment",
        name="restart-api-pod",
        namespace=NAMESPACE,
        image="bitnami/kubectl:latest",
        # ИСПРАВЛЕНИЕ: Исправлено имя деплоймента на то, которое задано в K8s манифестах
        cmds=["kubectl", "rollout", "restart", "deployment/industrial-nlp-api", "-n", NAMESPACE],
        get_logs=True,
        is_delete_operator_pod=True,
        # ИСПРАВЛЕНИЕ: Подключаем сервисный аккаунт (критично для kubectl)
        service_account_name="airflow-worker-sa",
    )

    promote_weights >> restart_api
