import logging

from dotenv import load_dotenv


# Загружаем окружение до Гидры!
load_dotenv()

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from src.utils.hydra_utils import setup_config  # noqa: E402


logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def evaluate(cfg: DictConfig) -> None:
    setup_config(cfg)

    logger.info("Инициализация компонентов для оценки...")
    tokenizer = hydra.utils.instantiate(cfg.model.tokenizer).build()
    model_builder = hydra.utils.instantiate(cfg.model.builder, tokenizer=tokenizer)
    base_model = model_builder.build()

    model_module = hydra.utils.instantiate(cfg.model_module, model=base_model)
    datamodule = hydra.utils.instantiate(cfg.datamodule, tokenizer=tokenizer)

    trainer = hydra.utils.instantiate(cfg.trainer)

    # Путь к PyTorch Lightning чекпоинту (.ckpt)
    ckpt_path = cfg.get("ckpt_path")

    if ckpt_path:
        logger.info(f"Загрузка PL чекпоинта из: {ckpt_path}")
    else:
        logger.warning("Путь к чекпоинту не передан. Запуск оценки на случайных весах.")

    logger.info("Старт процесса оценки...")
    # Метод test() сам загрузит стейты из .ckpt и подменит веса в model_module
    trainer.test(model=model_module, datamodule=datamodule, ckpt_path=ckpt_path)
    logger.info("Оценка завершена.")


if __name__ == "__main__":
    evaluate()
