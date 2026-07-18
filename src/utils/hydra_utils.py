# src/core/utils/hydra_utils.py
import logging
import os

from omegaconf import DictConfig, OmegaConf

from src.utils.config_schema import ConfigSchema


# Опционально импортируем python-json-logger для продакшена
try:
    from pythonjsonlogger import jsonlogger

    HAS_JSON_LOGGER = True
except ImportError:
    HAS_JSON_LOGGER = False

logger = logging.getLogger(__name__)


def setup_config(cfg: DictConfig) -> DictConfig:
    """
    Валидирует конфиг, разрешает ссылки и выводит его в логи.
    Возвращает валидированный конфиг.
    """

    # ИСПРАВЛЕНИЕ: Автоматический перевод логов в JSON формат (для ELK/Loki)
    # Срабатывает только если установлен пакет python-json-logger и мы в PROD
    if HAS_JSON_LOGGER and os.getenv("ENVIRONMENT") == "prod":
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            formatter = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
            handler.setFormatter(formatter)

    # 1. Разрешаем все ссылки вида ${paths.log_dir}
    OmegaConf.resolve(cfg)

    # 2. Строгая валидация (сверяем YAML с нашим датаклассом Schema)
    schema = OmegaConf.structured(ConfigSchema)

    # Сохраняем результат слияния схемы и пользовательского конфига
    validated_cfg = OmegaConf.merge(schema, cfg)

    # 3. Красиво выводим финальный конфиг в логи
    logger.info(f"Финальная конфигурация эксперимента:\n{OmegaConf.to_yaml(validated_cfg)}")

    return validated_cfg
