"""DAG: System Maintenance (Cleanup)."""

from typing import Any

from airflow import DAG
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

from dags.common import (
    IMAGE,
    NAMESPACE,
    make_default_args,
    make_failure_slack_alert,
    make_pvc_volume,
)


CONFIG: dict[str, Any] = Variable.get(
    "maintenance_config",
    default_var={
        "schedule": "0 3 * * 0",
        "retention_days": 30,
        "logs_pvc_name": "logs-pvc",
        "logs_mount_path": "/app/logs",
        "resources": {"requests": {"cpu": "0.5", "memory": "1Gi"}},
        "default_args": {"owner": "mlops", "retries": 1, "retry_delay_minutes": 5},
    },
    deserialize_json=True,
)

with DAG(
    "system_maintenance",
    default_args=make_default_args(**CONFIG["default_args"]),  # [cite: 32]
    schedule=CONFIG["schedule"],
    catchup=False,
    tags=["maintenance", "nlp"],
) as dag:
    # Используем общую фабрику[cite: 32]
    logs_vol, logs_mount = make_pvc_volume(
        "logs-data", CONFIG["logs_pvc_name"], CONFIG["logs_mount_path"]
    )

    cleanup_logs = KubernetesPodOperator(
        task_id="cleanup_logs_and_mlruns",
        name="cleanup-logs-pod",
        namespace=NAMESPACE,  # [cite: 32]
        image=IMAGE,  # [cite: 32]
        service_account_name="airflow-worker-sa",
        cmds=[
            "python",
            "-m",
            "src.tools.maintenance",
            "--action",
            "cleanup",
            "--days",
            str(CONFIG["retention_days"]),
        ],
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources"]),
        env_vars=[
            # Исправлен баг №2: Теперь передаем корректный MLRUNS_DIR
            k8s.V1EnvVar(name="MLRUNS_DIR", value=CONFIG["logs_mount_path"]),
        ],
        volume_mounts=[logs_mount],
        volumes=[logs_vol],
        get_logs=True,
        is_delete_operator_pod=True,
    )

    notify_failure = make_failure_slack_alert(
        "notify_on_failure", "system_maintenance"
    )  # [cite: 32]
    cleanup_logs >> notify_failure
