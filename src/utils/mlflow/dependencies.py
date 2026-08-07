# src/utils/mlflow/dependencies.py
"""Извлечение pip-зависимостей из pyproject.toml для MLflow model logging.

Отдельный модуль потому что логика работает с pyproject.toml,
а не с MLflow API — общее только то, что результат используется
при регистрации моделей.
"""

import logging
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logger = logging.getLogger(__name__)

_INFERENCE_GROUP: str = "inference-core"


def _strip_version_specifier(requirement: str) -> str:
    """Отрезает версии и extras, оставляя чистое имя пакета.

    Пример: "torch[cuda]>=2.0" → "torch"
    """
    name = re.split(r"[<>=!~\[;]", requirement, maxsplit=1)[0].strip()
    return name


def get_inference_pip_requirements(pyproject_path: str | Path) -> list[str]:
    """Читает группу inference-core из pyproject.toml и пинит установленные версии.

    Args:
        pyproject_path: Путь к pyproject.toml проекта.

    Returns:
        Список строк вида ``["torch==2.3.0", "transformers==4.40.0", ...]``.
        Пакеты, которые не установлены в окружении, пропускаются с предупреждением.
    """
    pyproject_path = Path(pyproject_path)
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    try:
        declared = data["project"]["optional-dependencies"][_INFERENCE_GROUP]
    except KeyError:
        logger.warning(
            "Группа [project.optional-dependencies.%s] не найдена в %s.",
            _INFERENCE_GROUP,
            pyproject_path,
        )
        return []

    pinned: list[str] = []
    for requirement in declared:
        pkg_name = _strip_version_specifier(requirement)
        try:
            installed_version = version(pkg_name)
            pinned.append(f"{pkg_name}=={installed_version}")
        except PackageNotFoundError:
            logger.warning("Пакет '%s' не установлен — пропускаю.", pkg_name)

    return pinned
