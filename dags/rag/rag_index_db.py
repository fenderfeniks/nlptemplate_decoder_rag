"""DAG: Инкрементальная индексация базы знаний (FAISS Indexing)."""

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
    "rag_index_config",
    default_var={
        "schedule": "0 2 * * *",
        "default_args": {"owner": "mlops", "retries": 1, "retry_delay_minutes": 5},
        "resources_gpu": {
            "requests": {"cpu": "2", "memory": "4Gi"},
            "limits": {"cpu": "4", "memory": "8Gi", "nvidia.com/gpu": "1"},
        },
        "db_pvc_name": "vector-db-pvc",
        "db_mount_path": "/app/vector_db",
        "data_pvc_name": "raw-data-pvc",
        "data_mount_path": "/app/data",
    },
    deserialize_json=True,
)

with DAG(
    "rag_incremental_indexing",
    default_args=make_default_args(**CONFIG["default_args"]),  # [cite: 32]
    schedule=CONFIG["schedule"],
    catchup=False,
    tags=["nlp", "rag", "indexing"],
) as dag:
    # Вызовы фабрики томов[cite: 32]
    data_vol, data_mount = make_pvc_volume(
        "raw-data", CONFIG["data_pvc_name"], CONFIG["data_mount_path"]
    )
    db_vol, db_mount = make_pvc_volume("vector-db", CONFIG["db_pvc_name"], CONFIG["db_mount_path"])

    reindex_vector_db = KubernetesPodOperator(
        task_id="incremental_reindex",
        name="rag-reindex-pod",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "scripts.rag_pipeline.index_db"],
        service_account_name="airflow-worker-sa",
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources_gpu"]),
        env_from=COMMON_ENV_FROM,
        volume_mounts=[data_mount, db_mount],
        volumes=[data_vol, db_vol],
        get_logs=True,
        is_delete_operator_pod=True,
    )

    notify_success = SlackWebhookOperator(
        task_id="notify_update_success",
        slack_webhook_conn_id="slack_conn",
        message="✅ Инкрементальная индексация завершена. База знаний FAISS обновлена.",
        trigger_rule="all_success",
    )

    notify_failure = make_failure_slack_alert(
        "notify_on_failure", "rag_incremental_indexing"
    )  # [cite: 32]

    reindex_vector_db >> notify_success
    reindex_vector_db >> notify_failure
