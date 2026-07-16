"""
DAG: System Maintenance (Backups & Cleanup)
"""
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.models import Variable
import pendulum

CONFIG = Variable.get("maintenance_config", deserialize_json=True)

with DAG(
    'system_maintenance',
    schedule_interval=CONFIG["schedule"],
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=['maintenance'],
) as dag:

    # Бекап Qdrant (через скрипт)
    backup_qdrant = KubernetesPodOperator(
        task_id='backup_qdrant',
        namespace=Variable.get("K8S_NAMESPACE"),
        image=Variable.get("PROJECT_IMAGE"),
        cmds=["python", "-m", "src.jobs.maintenance", "--action", "backup"],
        get_logs=True,
        is_delete_operator_pod=True,
    )

    # Очистка старых логов
    cleanup_logs = KubernetesPodOperator(
        task_id='cleanup_mlruns',
        namespace=Variable.get("K8S_NAMESPACE"),
        image=Variable.get("PROJECT_IMAGE"),
        cmds=["bash", "-c", "find /app/mlruns -mtime +30 -delete"],
        get_logs=True,
        is_delete_operator_pod=True,
    )

    backup_qdrant >> cleanup_logs