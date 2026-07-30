# src/utils/hydra_utils.py
import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from src.utils.config_schema import ConfigSchema


logger = logging.getLogger(__name__)


def _force_utf8_console_encoding() -> None:
    """Принудительно устанавливает кодировку UTF-8 для вывода в консоль.

    Полезно для корректного отображения спецсимволов и кириллицы
    в логах, особенно в средах Windows или нестандартных терминалах.
    """
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            stream = getattr(handler, "stream", None)
            if stream is not None and hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="backslashreplace")
                except (ValueError, OSError):
                    pass


def _resolve_trainer_callbacks(trainer_cfg: DictConfig) -> DictConfig:
    """Конвертирует словарь коллбэков в список для pytorch_lightning.Trainer.

    В Hydra-конфигах коллбэки часто задаются как словарь для удобства
    точечного переопределения параметров, но Trainer ожидает список.
    Функция выполняет эту трансформацию.

    Поддерживает три варианта того, что может прийти в trainer_cfg.callbacks:
    - DictConfig (словарь name -> callback_cfg) — берём .values()
    - ListConfig (уже список) — просто конвертируем в list
    - None / отсутствует — ставим пустой список

    ВАЖНО: если элемент в callbacks — строка, а не DictConfig, значит
    Hydra не смог смёржить файл коллбэка (неверный @package или путь).
    В этом случае бросаем понятную ошибку, а не даём PL падать с
    "AttributeError: 'str' object has no attribute 'log'".

    Args:
        trainer_cfg: Конфигурация тренера (извлеченная секция ``trainer``).

    Returns:
        Обновленная конфигурация тренера со списком коллбэков.
    """
    OmegaConf.set_struct(trainer_cfg, False)
    callbacks_node = trainer_cfg.get("callbacks")

    logger.debug(
        "callbacks_node type=%s, value=\n%s",
        type(callbacks_node).__name__,
        OmegaConf.to_yaml(callbacks_node)
        if isinstance(callbacks_node, (DictConfig, ListConfig))
        else repr(callbacks_node),
    )

    if not callbacks_node:
        trainer_cfg.callbacks = []
        return trainer_cfg

    if isinstance(callbacks_node, DictConfig):
        callbacks_list = list(callbacks_node.values())
    elif isinstance(callbacks_node, ListConfig):
        callbacks_list = list(callbacks_node)
    else:
        raise TypeError(
            f"trainer.callbacks должен быть DictConfig или ListConfig, "
            f"получен {type(callbacks_node).__name__}: {callbacks_node!r}\n"
            "Возможная причина: неверный синтаксис defaults в trainer/default.yaml "
            "или отсутствие '# @package _group_' в yaml-файле коллбэка."
        )

    # Проверяем, что внутри DictConfig/ListConfig нет голых строк
    # (признак того, что Hydra смёржил имена файлов вместо их содержимого)
    bad = [item for item in callbacks_list if isinstance(item, str)]
    if bad:
        raise ValueError(
            f"В trainer.callbacks обнаружены строки вместо конфигов коллбэков: {bad}\n"
            "Возможные причины:\n"
            "  1. Неверный синтаксис defaults в trainer/default.yaml.\n"
            "     Используйте:\n"
            "       defaults:\n"
            "         - callbacks/checkpoint\n"
            "         - callbacks/lr_monitor\n"
            "     А НЕ:\n"
            "       defaults:\n"
            "         - callbacks:\n"
            "             - checkpoint\n"
            "  2. Отсутствует '# @package _group_' в yaml-файле коллбэка.\n"
            "  3. Неверный путь к файлу коллбэка в defaults."
        )

    trainer_cfg.callbacks = callbacks_list
    return trainer_cfg


def setup_config(cfg: DictConfig) -> DictConfig:
    """Валидирует конфиг, разрешает ссылки и выставляет пути проекта.

    Выполняет следующие шаги:
    - принудительно включает UTF-8 для консоли;
    - выставляет корень проекта в ``cfg.paths.root_dir``;
    - разрешает интерполяции OmegaConf по всему дереву (включая trainer);
    - временно извлекает секцию ``decoder_pipeline.trainer`` для обхода
      строгой валидации схемой (trainer слишком динамичный);
    - валидирует структуру конфигурации через ``ConfigSchema``;
    - трансформирует словари коллбэков в списки;
    - возвращает структуру к строгой форме (set_struct=True).

    Args:
        cfg: Исходный Hydra-конфиг (DictConfig).

    Returns:
        Валидированный и подготовленный к инжекции в пайплайн DictConfig.
    """
    _force_utf8_console_encoding()

    OmegaConf.set_struct(cfg, False)

    # 1. Выставляем корень проекта до резолва, чтобы ${paths.root_dir} работал
    project_root = str(Path(__file__).resolve().parents[2])
    cfg.paths.root_dir = project_root

    # 2. Резолвим всё дерево целиком, пока trainer ещё на месте —
    #    иначе интерполяции вида ${decoder_pipeline.trainer...} не найдутся.
    OmegaConf.resolve(cfg)

    # 3. Вынимаем trainer после резолва — схема ConfigSchema его не знает.
    trainer_cfg = cfg.decoder_pipeline.pop("trainer")

    # 4. Валидируем всё остальное строгой схемой.
    schema = OmegaConf.structured(ConfigSchema)
    validated_cfg = OmegaConf.merge(schema, cfg)

    OmegaConf.set_struct(validated_cfg, False)

    # 5. Конвертируем callbacks dict -> list и возвращаем trainer на место.
    trainer_cfg = _resolve_trainer_callbacks(trainer_cfg)
    validated_cfg.decoder_pipeline.trainer = trainer_cfg

    OmegaConf.set_struct(validated_cfg, True)

    logger.debug("Финальная конфигурация эксперимента:\n%s", OmegaConf.to_yaml(validated_cfg))

    return validated_cfg
