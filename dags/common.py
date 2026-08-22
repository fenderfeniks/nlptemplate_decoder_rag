# dags/common.py
"""Общие утилиты и фабрики для Airflow DAGов проекта.

Выносим повторяющиеся конструкции (volumes, default_args, callbacks) сюда,
чтобы не дублировать их в каждом DAG-файле.

Note:
    Variable.get() обёрнуто в try/except — при недоступности БД Airflow
    во время парсинга DAGов используются дефолтные значения, чтобы все DAGи
    оставались доступны в UI.
"""

from __future__ import annotations

from typing import Any

import pendulum
from airflow.models import Variable
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from kubernetes.client import models as k8s


# ---------------------------------------------------------------------------
# Глобальные переменные — читаем один раз при парсинге DAG
# ---------------------------------------------------------------------------


def _get_variable(key: str, default: str) -> str:
    """Безопасное чтение Airflow Variable с fallback при недоступной БД."""
    try:
        return Variable.get(key, default_var=default)
    except Exception:
        return default


IMAGE: str = _get_variable("PROJECT_IMAGE", "my-company/nlp_template:training-latest")
API_IMAGE: str = _get_variable("PROJECT_API_IMAGE", "my-company/nlp_template:api-latest")
NAMESPACE: str = _get_variable("K8S_NAMESPACE", "ml-pipelines")

# Общий ConfigMap для всех подов
COMMON_ENV_FROM = [
    k8s.V1EnvFromSource(config_map_ref=k8s.V1ConfigMapEnvSource(name="nlp-template-api-config"))
]


# ---------------------------------------------------------------------------
# Фабрика default_args
# ---------------------------------------------------------------------------


def make_default_args(
    owner: str = "mlops",
    retries: int = 1,
    retry_delay_minutes: int = 5,
    on_failure_callback=None,
) -> dict[str, Any]:
    """Строит стандартный словарь default_args для DAG.

    Args:
        owner: Владелец DAG.
        retries: Количество повторов при падении.
        retry_delay_minutes: Задержка между повторами в минутах.
        on_failure_callback: Callback при падении (опционально).
    """
    args: dict[str, Any] = {
        "owner": owner,
        "depends_on_past": False,
        "start_date": pendulum.datetime(2026, 1, 1, tz="UTC"),
        "retries": retries,
        "retry_delay": pendulum.duration(minutes=retry_delay_minutes),
    }
    if on_failure_callback is not None:
        args["on_failure_callback"] = on_failure_callback
    return args


# ---------------------------------------------------------------------------
# Фабрика Volume + VolumeMount
# ---------------------------------------------------------------------------


def make_pvc_volume(
    volume_name: str,
    claim_name: str,
    mount_path: str,
) -> tuple[k8s.V1Volume, k8s.V1VolumeMount]:
    """Возвращает пару (Volume, VolumeMount) для PVC.

    Args:
        volume_name: Имя тома в Pod-спеке.
        claim_name: Имя PersistentVolumeClaim.
        mount_path: Путь монтирования внутри контейнера.
    """
    volume = k8s.V1Volume(
        name=volume_name,
        persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name=claim_name),
    )
    mount = k8s.V1VolumeMount(name=volume_name, mount_path=mount_path)
    return volume, mount


# ---------------------------------------------------------------------------
# Slack-нотификатор при падении
# ---------------------------------------------------------------------------


def make_failure_slack_alert(task_id: str, dag_id: str) -> SlackWebhookOperator:
    """Создаёт SlackWebhookOperator, который срабатывает при падении любого таска.

    Args:
        task_id: ID таска-нотификатора.
        dag_id: ID DAG для включения в сообщение.
    """
    return SlackWebhookOperator(
        task_id=task_id,
        slack_webhook_conn_id="slack_conn",
        message=f"🔴 DAG `{dag_id}` завершился с ошибкой. Проверьте логи Airflow.",
        trigger_rule="one_failed",
    )
