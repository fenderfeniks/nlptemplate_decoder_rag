# src/core/utils/hydra_utils.py
import logging

from omegaconf import DictConfig, OmegaConf

from src.utils.config_schema import ConfigSchema


logger = logging.getLogger(__name__)


def setup_config(cfg: DictConfig) -> None:
    """
    Валидирует конфиг, разрешает ссылки и выводит его в логи.
    """
    # 1. Разрешаем все ссылки вида ${paths.log_dir}
    OmegaConf.resolve(cfg)

    # 2. Строгая валидация (сверяем YAML с нашим датаклассом Schema)
    # Если в YAML будет опечатка, скрипт упадет прямо здесь с понятной ошибкой
    schema = OmegaConf.structured(ConfigSchema)
    OmegaConf.merge(schema, cfg)

    # 3. Красиво выводим финальный конфиг в логи
    logger.info(f"Финальная конфигурация эксперимента:\n{OmegaConf.to_yaml(cfg)}")
