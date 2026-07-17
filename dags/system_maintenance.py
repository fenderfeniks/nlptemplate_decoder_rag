"""
DAG: System Maintenance (Backups & Cleanup)
"""
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.models import Variable
from kubernetes.client import models as k8s
import pendulum

# 1. ИНФРАСТРУКТУРА
IMAGE = Variable.get("PROJECT_IMAGE", default_var="my-company/industrial_nlp_template:latest")
NAMESPACE = Variable.get("K8S_NAMESPACE", default_var="ml-pipelines")

# 2. БИЗНЕС-ЛОГИКА
CONFIG = Variable.get("maintenance_config", deserialize_json=True)

default_args = {
    'owner': CONFIG["default_args"]["owner"],
    'depends_on_past': False,
    'start_date': pendulum.datetime(2026, 1, 1, tz="UTC"),
    'email_on_failure': True,
    'retries': CONFIG["default_args"]["retries"],
    'retry_delay': pendulum.duration(minutes=CONFIG["default_args"]["retry_delay_minutes"]),
}

with DAG(
    'system_maintenance',
    default_args=default_args,
    schedule_interval=CONFIG["schedule"],
    catchup=False,
    tags=['maintenance', 'nlp'],
) as dag:

    # Задача 1: Бекап Qdrant
    backup_qdrant = KubernetesPodOperator(
        task_id='backup_qdrant',
        name='qdrant-backup-pod',
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "src.jobs.maintenance", "--action", "backup"],
        
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources"]),
        env_vars=[
            k8s.V1EnvVar(name="QDRANT_URL", value=CONFIG["qdrant_url"]),
            k8s.V1EnvVar(
                name="QDRANT_API_KEY", 
                value_from=k8s.V1EnvVarSource(
                    secret_key_ref=k8s.V1SecretKeySelector(
                        name=CONFIG["qdrant_secret_name"], 
                        key="api-key"
                    )
                )
            )
        ],
        get_logs=True,
        is_delete_operator_pod=True,
    )

    # Задача 2: Очистка старых логов (MLflow / чекпоинты)
    cleanup_logs = KubernetesPodOperator(
        task_id='cleanup_mlruns',
        name='cleanup-logs-pod',
        namespace=NAMESPACE,
        image=IMAGE,
        # Используем наш Python-скрипт и передаем дни динамически из JSON
        cmds=[
            "python", "-m", "src.jobs.maintenance", 
            "--action", "cleanup", 
            "--days", str(CONFIG["retention_days"])
        ],
        
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources"]),
        
        # МАГИЯ ЗДЕСЬ: Монтируем реальный диск, чтобы было что удалять
        volume_mounts=[
            k8s.V1VolumeMount(name='mlruns-data', mount_path=CONFIG["mlruns_mount_path"])
        ],
        volumes=[
            k8s.V1Volume(
                name='mlruns-data', 
                persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name=CONFIG["mlruns_pvc_name"])
            )
        ],
        get_logs=True,
        is_delete_operator_pod=True,
    )

    backup_qdrant >> cleanup_logs