# tests/dags/conftest.py
"""Конфигурация тестовой сессии для Airflow DAGов.

Порядок инициализации критичен — нарушение порядка приводит к circular import
и OperationalError на Windows:
  1. Unix-моки               — до любых импортов airflow.*
  2. Env-переменные Airflow  — до первого импорта airflow.*
  3. Патч Variable.get       — до создания DagBag
  4. Инициализация схемы БД  — до создания DagBag
  5. DagBag                  — один на всю сессию

Все импорты airflow.* вынесены внутрь фикстур (не на уровень модуля),
чтобы избежать circular import при pytest-коллекции на Windows.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. Unix-фикс — ОБЯЗАТЕЛЬНО до любых импортов airflow.*
#    Выполняется при импорте conftest.py, то есть раньше коллекции тестов.
# ---------------------------------------------------------------------------
if os.name == "nt":
    _unix_mocks: dict = {
        "fcntl": MagicMock(**{"ioctl.return_value": b"\x00" * 8}),
        "pwd": MagicMock(),
        "grp": MagicMock(),
        "posix": MagicMock(),
        "termios": MagicMock(),
    }
    for _mod_name, _mock in _unix_mocks.items():
        sys.modules.setdefault(_mod_name, _mock)


# ---------------------------------------------------------------------------
# 2. Env-переменные Airflow — до первого импорта airflow.*
#
#    sqlite:///:memory: отклоняется Airflow (требует абсолютный путь).
#    Используем временный файл — он удаляется после сессии.
# ---------------------------------------------------------------------------
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp_db.close()
_DB_PATH = Path(_tmp_db.name).as_posix()  # /tmp/... или C:/Users/.../tmp/...

os.environ.update(
    {
        "AIRFLOW__CORE__UNIT_TEST_MODE": "True",
        "AIRFLOW__CORE__LOAD_EXAMPLES": "False",
        "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN": f"sqlite:///{_DB_PATH}",
        # SequentialExecutor не требует внешних зависимостей
        "AIRFLOW__CORE__EXECUTOR": "SequentialExecutor",
        # Отключаем структурированный логгер Airflow — он вызывает circular import
        # при резолве airflow.utils.log.* на Windows во время коллекции тестов
        "AIRFLOW__LOGGING__LOGGING_CONFIG_CLASS": "",
        "AIRFLOW__LOGGING__FAB_LOGGING_LEVEL": "WARNING",
    }
)


# ---------------------------------------------------------------------------
# Вспомогательная функция — заглушка для Variable.get
# ---------------------------------------------------------------------------
def _variable_get(key, default_var=None, deserialize_json=False, **kwargs):
    """Возвращает default_var без обращения к БД."""
    return default_var


# ---------------------------------------------------------------------------
# 3. Патч Variable.get — autouse, scope=session
#    Активируется до создания DagBag (dagbag зависит от этой фикстуры явно).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def mock_airflow_variable():
    with patch("airflow.models.Variable.get", side_effect=_variable_get):
        yield


# ---------------------------------------------------------------------------
# 4. Инициализация БД + единый DagBag
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def dagbag(mock_airflow_variable):
    """Единый DagBag на всю тестовую сессию."""
    from airflow.models import DagBag
    from airflow.utils import db

    # Исправление для Windows: патчим и DB-таймаут, и PythonImporter-таймаут
    if os.name == "nt":
        import contextlib

        try:
            from airflow.utils import db as db_module

            @contextlib.contextmanager
            def _dummy_timeout(seconds, message="Operation timed out"):
                yield

            db_module.timeout_with_traceback = _dummy_timeout
        except Exception:
            pass

        try:
            from airflow.dag_processing.importers import python_importer

            @contextlib.contextmanager
            def _dummy_importer_timeout(seconds, error_message=None):
                yield

            python_importer._timeout = _dummy_importer_timeout
        except Exception:
            pass

    db.upgradedb()

    root_dir = Path(__file__).resolve().parent.parent.parent
    dags_path = root_dir / "dags"

    bag = DagBag(dag_folder=str(dags_path), include_examples=False, safe_mode=False)
    return bag


# ---------------------------------------------------------------------------
# 5. Мок выполнения операторов
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def mock_operators():
    """Предотвращает реальный запуск K8s и Slack операторов."""
    # Импорты внутри фикстуры — безопасно после env-инициализации
    with (
        patch(
            "airflow.providers.cncf.kubernetes.operators.pod.KubernetesPodOperator.execute",
            return_value=None,
        ),
        patch(
            "airflow.providers.slack.operators.slack_webhook.SlackWebhookOperator.execute",
            return_value=None,
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# 6. Cleanup временного DB-файла после сессии
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def cleanup_tmp_db():
    yield
    try:
        Path(_tmp_db.name).unlink(missing_ok=True)
    except Exception:
        pass
