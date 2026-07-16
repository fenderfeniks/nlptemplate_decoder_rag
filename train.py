"""
Главный скрипт запуска обучения (Orchestrator).
Инициализирует Hydra, настраивает воспроизводимость, собирает компоненты
и запускает тренировочный цикл PyTorch Lightning.
"""

import os
import logging
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from src.core.utils.hydra_utils import setup_config

# Настройка логгера для текущего файла
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def train(cfg: DictConfig) -> None:
    """
    Основная функция запуска эксперимента.
    
    Args:
        cfg (DictConfig): Разрешенная конфигурация Hydra.
    """
    
    setup_config(cfg)

    # 2. Обеспечение воспроизводимости
    # pl.seed_everything фиксирует сиды для random, numpy, torch (CPU/GPU)
    if "seed" in cfg:
        pl.seed_everything(cfg.seed, workers=True)
        logger.info(f"Зафиксирован глобальный seed: {cfg.seed}")

    # 3. Сборка токенизатора через нашу фабрику
    logger.info("Инициализация токенизатора...")
    tokenizer_builder = hydra.utils.instantiate(cfg.model.tokenizer)
    tokenizer = tokenizer_builder.build()

    # 4. Сборка базовой модели (с квантизацией и LoRA, если они настроены)
    logger.info("Загрузка и сборка архитектуры модели...")
    model_builder = hydra.utils.instantiate(cfg.model.builder, tokenizer=tokenizer)
    base_model = model_builder.build()

    # 5. Инициализация LightningModule (оркестратора логики шагов обучения)
    logger.info("Инициализация PyTorch Lightning Module...")
    # Передаем готовую модель внутрь LightningModule
    model_module = hydra.utils.instantiate(
        cfg.model_module,
        model=base_model
    )

    # 6. Инициализация DataModule
    logger.info("Инициализация DataModule...")
    # Прокидываем токенизатор, который создали на шаге 3
    datamodule = hydra.utils.instantiate(
        cfg.datamodule,
        tokenizer=tokenizer
    )

    # 7. Сборка PyTorch Lightning Trainer (включая MLflow логгер и коллбэки)
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