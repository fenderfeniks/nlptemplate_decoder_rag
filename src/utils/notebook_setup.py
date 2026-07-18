import logging
import os
import sys
from pathlib import Path

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


def init_nlp_notebook(config_name: str = "main"):
    """
    Инициализирует окружение для NLP ноутбука.
    Настраивает пути, логирование и загружает конфиг Hydra.
    """
    # 1. Находим корень проекта (путь до файла -> родитель -> родитель -> корень)
    project_root = Path(__file__).resolve().parents[2]

    # 2. Устанавливаем рабочую директорию
    if os.getcwd() != str(project_root):
        os.chdir(project_root)
        print(f"Working directory set to: {os.getcwd()}")

    # 3. Добавляем корень в sys.path, чтобы работали импорты 'from src...'
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    # 4. Инициализация Hydra
    GlobalHydra.instance().clear()
    initialize(config_path="../../configs", version_base="1.3")
    cfg = compose(config_name=config_name)

    # Настройка логгера для ноутбука (чтобы видеть логи библиотек)
    logging.basicConfig(level=logging.INFO)

    print(f"NLP Environment ready. Config loaded: {config_name}")
    return cfg
