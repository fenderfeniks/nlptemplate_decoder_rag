# tests/dags/test_dag_contracts.py
"""Тесты контрактов DAGов: секреты, образы, монтирования, аргументы скриптов."""


def test_batch_analytics_db_conn_from_secret(dagbag):
    """DB_CONN должна пробрасываться из K8s Secret, а не задаваться явно."""
    for dag_id in ["llm_batch_analytics_reporting", "rag_batch_analytics_reporting"]:
        dag = dagbag.get_dag(dag_id)
        assert dag is not None, f"DAG {dag_id} не найден"
        task = dag.get_task("run_batch_inference")

        db_env = next((env for env in task.env_vars if env.name == "DB_CONN"), None)
        assert db_env is not None, f"Переменная DB_CONN не передана в {dag_id}"
        assert db_env.value_from.secret_key_ref.name == "db-secrets", (
            f"Неверный секрет для DB_CONN в {dag_id}"
        )


def test_promote_kubectl_image_is_pinned(dagbag):
    """kubectl-образ должен быть зафиксирован на конкретном теге (не latest)."""
    for dag_id, task_id in [
        ("promote_llm_to_prod", "restart_api_deployment"),
        ("promote_rag_to_prod", "restart_rag_api_deployment"),
    ]:
        dag = dagbag.get_dag(dag_id)
        assert dag is not None, f"DAG {dag_id} не найден"
        task = dag.get_task(task_id)
        assert task.image == "bitnami/kubectl:1.29", (
            f"Незафиксированный образ kubectl в {dag_id}.{task_id}: {task.image}"
        )


def test_rag_indexing_mounts_multiple_volumes(dagbag):
    """Задача индексации должна монтировать и raw-data, и vector-db PVC."""
    dag = dagbag.get_dag("rag_incremental_indexing")
    assert dag is not None, "DAG rag_incremental_indexing не найден"
    task = dag.get_task("incremental_reindex")

    assert len(task.volumes) >= 2, f"Ожидается ≥2 Volumes, получено {len(task.volumes)}"
    assert len(task.volume_mounts) >= 2, (
        f"Ожидается ≥2 VolumeMounts, получено {len(task.volume_mounts)}"
    )

    mount_names = [m.name for m in task.volume_mounts]
    assert "raw-data" in mount_names, f"Нет raw-data в {mount_names}"
    assert "vector-db" in mount_names, f"Нет vector-db в {mount_names}"


def test_rag_ingestion_pipeline_name(dagbag):
    """fetch_data должен запускаться с pipeline_name=rag_pipeline."""
    dag = dagbag.get_dag("rag_data_ingestion")
    assert dag is not None, "DAG rag_data_ingestion не найден"
    task = dag.get_task("fetch_data_from_sources")
    assert any("rag_pipeline" in cmd for cmd in task.cmds), (
        f"pipeline_name=rag_pipeline не найден в cmds: {task.cmds}"
    )


def test_quality_control_slack_trigger_rule(dagbag):
    """alert_if_drift должен срабатывать только при падении (one_failed)."""
    # Импорт внутри теста — избегаем circular import при коллекции на Windows
    from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

    for dag_id in ["llm_quality_drift_detection", "rag_quality_drift_detection"]:
        dag = dagbag.get_dag(dag_id)
        assert dag is not None, f"DAG {dag_id} не найден"
        task = dag.get_task("alert_if_drift")
        assert isinstance(task, SlackWebhookOperator), (
            f"alert_if_drift в {dag_id} не является SlackWebhookOperator"
        )
        assert task.trigger_rule == "one_failed", (
            f"Неверный trigger_rule в {dag_id}: {task.trigger_rule}"
        )
