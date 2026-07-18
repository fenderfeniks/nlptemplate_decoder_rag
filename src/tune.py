# src/tune.py
import logging

from dotenv import load_dotenv


# Загружаем окружение до Гидры!
load_dotenv()

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

# Импортируем наш новый класс
from src.training.tuner import NLPOptunaTuner  # noqa: E402
from src.utils.hydra_utils import setup_config  # noqa: E402


logger = logging.getLogger(__name__)


# ИСПРАВЛЕНИЕ: Изменен config_path с "configs" на "../configs"
@hydra.main(config_path="../configs", config_name="tune", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Валидация конфига (твоя кастомная функция)
    setup_config(cfg)

    if not hasattr(cfg, "optuna") or cfg.optuna is None:
        raise ValueError("Секция 'optuna' отсутствует в конфигурации! Проверьте configs/tune.yaml.")

    logger.info("Инициализация Optuna Tuner...")
    tuner = NLPOptunaTuner(cfg=cfg)

    logger.info(f"Запуск подбора параметров (Trials: {cfg.optuna.n_trials})...")
    tuner.optimize()


if __name__ == "__main__":
    main()
