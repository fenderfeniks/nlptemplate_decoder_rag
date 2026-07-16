"""
DAG: Batch Analytics (LLM-as-an-Analyst)
"""
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.models import Variable
from kubernetes.client import models as k8s
import pendulum

IMAGE = Variable.get("PROJECT_IMAGE")
NAMESPACE = Variable.get("K8S_NAMESPACE")
CONFIG = Variable.get("analytics_config", deserialize_json=True)

with DAG(
    'batch_analytics_reporting',
    schedule_interval=CONFIG["schedule"],
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=['nlp', 'analytics'],
) as dag:

    analyze_reviews = KubernetesPodOperator(
        task_id='run_batch_inference',
        name='analytics-pod',
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", "src.jobs.batch_analytics"], # Новый модуль
        container_resources=k8s.V1ResourceRequirements(**CONFIG["resources"]),
        env_vars=[k8s.V1EnvVar(name="DB_CONN", value=CONFIG["db_connection"])],
        get_logs=True,
        is_delete_operator_pod=True,
    )