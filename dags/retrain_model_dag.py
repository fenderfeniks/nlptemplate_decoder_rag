"""
DAG: Еженедельное переобучение модели (Continuous Finetuning).
Архитектура: Разделение инфраструктуры и бизнес-логики.
"""
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.models import Variable
from kubernetes.client import models as k8s
import pendulum

# 1. ИНФРАСТРУКТУРА (Глобальные переменные)
IMAGE = Variable.get("PROJECT_IMAGE", default_var="my-company/industrial_nlp_template:latest")
NAMESPACE = Variable.get("K8S_NAMESPACE", default_var="ml-pipelines")

# 2. БИЗНЕС-ЛОГИКА (Конфиг из Variables без хардкода по умолчанию, так как он уже есть в БД Airflow)
CONFIG = Variable.get("training_config", deserialize_json=True)

default_args = {
    'owner': CONFIG["default_args"]["owner"],
    'depends_on_past': False,
    'start_date': pendulum.datetime(2026, 1, 1, tz="UTC"),
    'email_on_failure': True,
    'retries': CONFIG["default_args"]["retries"],
    'retry_delay': pendulum.duration(minutes=CONFIG["default_args"]["retry_delay_minutes"]),
}

with DAG(
    'weekly_llm_finetuning',
    default_args=default_args,
    schedule_interval=CONFIG["schedule"],
    catchup=False,
    tags=['nlp', 'training'],
) as dag:

    train_model_task = KubernetesPodOperator(
        task_id='run_lora_finetuning',
        name='llm-trainer-pod',
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "src.train"],
        
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources"]),
        
        # --- ДОБАВЛЕНО: Проброс секретов для обучения ---
        env_vars=[
            # Токен HuggingFace для скачивания базовой модели (опционально)
            k8s.V1EnvVar(
                name="HUGGINGFACE_TOKEN", 
                value_from=k8s.V1EnvVarSource(
                    secret_key_ref=k8s.V1SecretKeySelector(
                        name="hf-secrets", 
                        key="token",
                        optional=True 
                    )
                )
            ),
            # Ключ Weights & Biases для логирования экспериментов (опционально)
            k8s.V1EnvVar(
                name="WANDB_API_KEY", 
                value_from=k8s.V1EnvVarSource(
                    secret_key_ref=k8s.V1SecretKeySelector(
                        name="wandb-secrets", 
                        key="api-key",
                        optional=True 
                    )
                )
            )
        ],
        
        volume_mounts=[
            k8s.V1VolumeMount(name='model-weights', mount_path=CONFIG["mount_path"])
        ],
        volumes=[
            k8s.V1Volume(
                name='model-weights', 
                persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name=CONFIG["pvc_name"])
            )
        ],
        
        get_logs=True,
        is_delete_operator_pod=True,
    )