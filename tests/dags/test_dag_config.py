# tests/dags/test_dag_config.py
"""Тесты конфигурации DAGов: расписания, пороги метрик."""


def test_maintenance_schedule(dagbag):
    dag = dagbag.get_dag("system_maintenance")
    assert dag is not None, "DAG system_maintenance не найден"
    assert dag.schedule == "0 3 * * 0"


def test_rag_indexing_schedule(dagbag):
    dag = dagbag.get_dag("rag_incremental_indexing")
    assert dag is not None, "DAG rag_incremental_indexing не найден"
    assert dag.schedule == "0 2 * * *"


def test_llm_quality_control_threshold(dagbag):
    """Порог ROUGE-1 корректно передаётся в eval-скрипт."""
    dag = dagbag.get_dag("llm_quality_drift_detection")
    assert dag is not None, "DAG llm_quality_drift_detection не найден"
    task = dag.get_task("evaluate_llm")
    assert any("rouge1=0.45" in arg for arg in task.arguments), (
        f"Порог rouge1=0.45 не найден в аргументах: {task.arguments}"
    )


def test_rag_quality_control_threshold(dagbag):
    """Порог MRR корректно передаётся в eval-скрипт энкодера."""
    dag = dagbag.get_dag("rag_quality_drift_detection")
    assert dag is not None, "DAG rag_quality_drift_detection не найден"
    task = dag.get_task("evaluate_rag_encoder")
    assert any("val_mrr=0.75" in arg for arg in task.arguments), (
        f"Порог val_mrr=0.75 не найден в аргументах: {task.arguments}"
    )
