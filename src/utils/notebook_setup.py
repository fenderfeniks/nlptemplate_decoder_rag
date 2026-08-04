import logging
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


logger = logging.getLogger(__name__)


def _find_project_root() -> Path:
    """Ищет корень проекта по наличию ``pyproject.toml``."""
    for directory in [Path.cwd(), *Path.cwd().parents]:
        if (directory / "pyproject.toml").exists():
            return directory

    raise FileNotFoundError(
        "Could not locate project root: 'pyproject.toml' not found "
        f"in '{Path.cwd()}' or any of its parents."
    )


def _ensure_on_path(directory: Path) -> None:
    """Добавляет директорию в начало ``sys.path``, если её там нет."""
    path_str = str(directory)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        logger.debug("Added '%s' to sys.path.", path_str)


def _init_hydra(config_dir: str, config_name: str) -> DictConfig:
    """Инициализирует Hydra по абсолютному пути и возвращает собранный конфиг."""
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=config_dir, version_base="1.3")
    return compose(config_name=config_name)


def setup_notebook(config_name: str = "main") -> DictConfig:
    """Инициализирует окружение для запуска ноутбука."""
    project_root = _find_project_root()
    _ensure_on_path(project_root)

    # Вычисляем абсолютный путь к директории с конфигами
    config_path = str(project_root / "configs")
    cfg = _init_hydra(
        config_dir=config_path,
        config_name=config_name,
    )

    logging.basicConfig(level=logging.INFO, force=True)
    logger.info("NLP Environment ready. Root: %s", project_root)

    return cfg
