# tests/dags/test_dag_k8s.py
"""Тесты соответствия K8s-операторов инфраструктурным требованиям."""


def test_all_k8s_tasks_compliant(dagbag):
    """Все KubernetesPodOperator должны соответствовать инфраструктурным требованиям:
    - неймспейс ml-pipelines
    - service account airflow-worker-sa
    - pod удаляется после завершения
    """
    # Импорт внутри теста — избегаем circular import при коллекции на Windows
    from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator

    violations = []
    for dag_id, dag in dagbag.dags.items():
        for task in dag.tasks:
            if not isinstance(task, KubernetesPodOperator):
                continue
            tid = f"{dag_id}.{task.task_id}"

            if task.namespace != "ml-pipelines":
                violations.append(f"{tid}: namespace={task.namespace!r} (ожидается 'ml-pipelines')")
            if task.service_account_name != "airflow-worker-sa":
                violations.append(f"{tid}: service_account={task.service_account_name!r}")
            if task._is_delete_operator_pod is not True:
                violations.append(f"{tid}: is_delete_operator_pod={task._is_delete_operator_pod}")

    assert not violations, "Нарушения K8s-политики:\n" + "\n".join(violations)
