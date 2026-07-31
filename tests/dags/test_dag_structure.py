# tests/dags/test_dag_structure.py
"""Тесты структуры DAGов: наличие, отсутствие ошибок импорта, расписание промоута."""

EXPECTED_DAGS = [
    # LLM
    "system_maintenance",
    "llm_batch_analytics_reporting",
    "promote_llm_to_prod",
    "llm_quality_drift_detection",
    "llm_weekly_finetuning",
    # RAG
    "rag_batch_analytics_reporting",
    "rag_data_ingestion",
    "rag_incremental_indexing",
    "promote_rag_to_prod",
    "rag_quality_drift_detection",
    "rag_encoder_finetuning",
]


def test_no_import_errors(dagbag):
    """Ни один DAG не должен падать при импорте."""
    assert not dagbag.import_errors, "Ошибки импорта DAG:\n" + "\n".join(
        f"  {path}: {err}" for path, err in dagbag.import_errors.items()
    )


def test_dag_structure_exists(dagbag):
    """Все ожидаемые DAGи должны быть найдены в DagBag."""
    missing = [dag_id for dag_id in EXPECTED_DAGS if dag_id not in dagbag.dags]
    assert not missing, f"Не найдены DAGи: {missing}"


def test_promote_is_manual_trigger_only(dagbag):
    """Промоут моделей в прод должен запускаться только вручную (schedule=None)."""
    for dag_id in ["promote_llm_to_prod", "promote_rag_to_prod"]:
        dag = dagbag.get_dag(dag_id)
        assert dag is not None, f"DAG {dag_id} не найден"
        assert dag.schedule is None, (
            f"DAG {dag_id} должен быть manual-only, но schedule={dag.schedule}"
        )
