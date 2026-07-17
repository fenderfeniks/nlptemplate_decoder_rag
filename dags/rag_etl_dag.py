from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.models import Variable
from kubernetes.client import models as k8s
import pendulum

# 1. ИНФРАСТРУКТУРА
IMAGE = Variable.get("PROJECT_IMAGE", default_var="my-company/industrial_nlp_template:latest")
NAMESPACE = Variable.get("K8S_NAMESPACE", default_var="ml-pipelines")

# 2. БИЗНЕС-ЛОГИКА
# ИСПРАВЛЕНИЕ: Имя переменной теперь точно совпадает с ключом в variables.json
CONFIG = Variable.get("etl_config", deserialize_json=True)

default_args = {
    'owner': CONFIG["default_args"]["owner"],
    'depends_on_past': False,
    'start_date': pendulum.datetime(2026, 1, 1, tz="UTC"),
    'email_on_failure': True,
    'retries': CONFIG["default_args"]["retries"],
    'retry_delay': pendulum.duration(minutes=CONFIG["default_args"]["retry_delay_minutes"]),
}

with DAG(
    'rag_knowledge_base_update',
    default_args=default_args,
    schedule_interval=CONFIG["schedule"],
    catchup=False,
    max_active_runs=1,
    tags=['nlp', 'rag', 'etl'],
) as dag:

    update_rag_task = KubernetesPodOperator(
        task_id='run_rag_update_job',
        name='rag-etl-pod',
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "src.jobs.rag_update_job"], 
        env_vars=[
            k8s.V1EnvVar(name="QDRANT_URL", value=CONFIG["qdrant_url"]),
            # Секрет для Qdrant
            k8s.V1EnvVar(
                name="QDRANT_API_KEY", 
                value_from=k8s.V1EnvVarSource(
                    secret_key_ref=k8s.V1SecretKeySelector(
                        name=CONFIG["qdrant_secret_name"], 
                        key="api-key"
                    )
                )
            ),
            # Секрет для загрузки модели эмбеддингов (если используется HF)
            k8s.V1EnvVar(
                name="HUGGINGFACE_TOKEN", 
                value_from=k8s.V1EnvVarSource(
                    secret_key_ref=k8s.V1SecretKeySelector(
                        name="hf-secrets", # Предполагаем, что такой секрет есть в K8s
                        key="token",
                        optional=True
                    )
                )
            ),
            # 2. OpenAI
            k8s.V1EnvVar(
                name="OPENAI_API_KEY", 
                value_from=k8s.V1EnvVarSource(
                    secret_key_ref=k8s.V1SecretKeySelector(
                        name="openai-secrets", 
                        key="api-key",
                        optional=True  # <--- И ЗДЕСЬ
                    )
                )
            )
        ],
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources"]),
        volume_mounts=[
            k8s.V1VolumeMount(name='raw-data', mount_path=CONFIG["mount_path"])
        ],
        volumes=[
            k8s.V1Volume(
                name='raw-data', 
                persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name=CONFIG["pvc_name"])
            )
        ],
        get_logs=True,
        is_delete_operator_pod=True,
    )