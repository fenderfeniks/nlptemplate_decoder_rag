"""
Скрипт оценки (Evaluation) обученной модели.
Загружает лучшие веса из чекпоинта и запускает тестирование на валидационной или тестовой выборке.
"""

import logging
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from src.utils.hydra_utils import setup_config

logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def evaluate(cfg: DictConfig) -> None:
    # 1. Валидация и подготовка конфига
    setup_config(cfg)
    
    # 2. Инициализация (Токенизатор, Модель, Датамодуль)
    # Аналогично тому, как мы делали в train.py
    tokenizer = hydra.utils.instantiate(cfg.model.tokenizer).build()
    model_builder = hydra.utils.instantiate(cfg.model.builder, tokenizer=tokenizer)
    base_model = model_builder.build()
    
    model_module = hydra.utils.instantiate(cfg.model_module, model=base_model)
    datamodule = hydra.utils.instantiate(cfg.datamodule, tokenizer=tokenizer)
    
    # 3. Инициализация Тренера (без GPU, если нужно)
    trainer = hydra.utils.instantiate(cfg.trainer)
    
    # 4. ЗАГРУЗКА ЛУЧШИХ ВЕСОВ
    # Мы ожидаем, что путь к чекпоинту передается через командную строку
    # Пример: python eval.py ckpt_path=/path/to/checkpoint.ckpt
    ckpt_path = cfg.get("ckpt_path")
    
    if ckpt_path:
        logger.info(f"Загрузка весов из: {ckpt_path}")
    else:
        logger.warning("Путь к чекпоинту не передан. Запуск оценки на случайных весах.")

    # 5. Запуск оценки
    logger.info("Старт процесса оценки...")
    trainer.test(model=model_module, datamodule=datamodule, ckpt_path=ckpt_path)
    logger.info("Оценка завершена.")

if __name__ == "__main__":
    evaluate()