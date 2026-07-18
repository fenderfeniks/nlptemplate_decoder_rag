"""
Главный скрипт запуска обучения (Orchestrator).
Инициализирует Hydra, настраивает воспроизводимость, собирает компоненты
и запускает тренировочный цикл PyTorch Lightning.
"""

import logging

import hydra
import pytorch_lightning as pl
from dotenv import load_dotenv
from omegaconf import DictConfig


load_dotenv()
from src.utils.hydra_utils import setup_config  # noqa: E402


# Настройка логгера для текущего файла
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def train(cfg: DictConfig) -> None:
    """
    Основная функция запуска эксперимента.

    Args:
        cfg (DictConfig): Разрешенная конфигурация Hydra.
    """

    # ИСПРАВЛЕНИЕ: Перехватываем возвращаемый валидированный конфиг
    cfg = setup_config(cfg)

    # 2. Обеспечение воспроизводимости
    if "seed" in cfg:
        pl.seed_everything(cfg.seed, workers=True)
        logger.info(f"Зафиксирован глобальный seed: {cfg.seed}")

    # 3. Сборка токенизатора через нашу фабрику
    logger.info("Инициализация токенизатора...")
    tokenizer_builder = hydra.utils.instantiate(cfg.model.tokenizer)
    tokenizer = tokenizer_builder.build()

    # 4. Сборка базовой модели
    logger.info("Загрузка и сборка архитектуры модели...")
    model_builder = hydra.utils.instantiate(cfg.model.builder, tokenizer=tokenizer)
    base_model = model_builder.build()

    # 5. Инициализация LightningModule
    logger.info("Инициализация PyTorch Lightning Module...")
    model_module = hydra.utils.instantiate(cfg.model_module, model=base_model)

    # 6. Инициализация DataModule
    logger.info("Инициализация DataModule...")
    datamodule = hydra.utils.instantiate(cfg.datamodule, tokenizer=tokenizer)

    # 7. Сборка PyTorch Lightning Trainer
    logger.info("Инициализация PyTorch Lightning Trainer...")
    trainer = hydra.utils.instantiate(cfg.trainer)

    # 8. Запуск обучения!
    logger.info("Старт тренировочного цикла...")
    try:
        trainer.fit(model=model_module, datamodule=datamodule)
        logger.info("Обучение успешно завершено!")
    except Exception as e:
        logger.exception("Произошла критическая ошибка во время обучения:")
        raise e


if __name__ == "__main__":
    train()
