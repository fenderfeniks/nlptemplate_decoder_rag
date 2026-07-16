"""
DAG: Model Evaluation & Drift Detection
"""
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.models import Variable
import pendulum

CONFIG = Variable.get("evaluation_config", deserialize_json=True)

with DAG(
    'model_drift_detection',
    schedule_interval=CONFIG["schedule"],
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=['nlp', 'monitoring'],
) as dag:

    # 1. Запуск оценки (используем наш готовый src.evaluate)
    evaluate_model = KubernetesPodOperator(
        task_id='evaluate_model',
        namespace=Variable.get("K8S_NAMESPACE"),
        image=Variable.get("PROJECT_IMAGE"),
        cmds=["python", "-m", "src.evaluate"], # Тот самый скрипт, что мы правили
        arguments=["ckpt_path=/app/models/best.ckpt"], # Передаем аргументы
        get_logs=True,
        is_delete_operator_pod=True,
    )

    # 2. Условный алерт (упрощенно)
    notify_slack = SlackWebhookOperator(
        task_id='alert_if_drift',
        slack_webhook_conn_id='slack_conn',
        message="⚠️ Внимание! Качество модели упало ниже порога 90%. Нужен ретрейн.",
        trigger_rule='one_failed' # Сработает, если предыдущий таск "упал" по метрикам
    )

    evaluate_model >> notify_slack