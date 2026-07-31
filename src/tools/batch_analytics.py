# src/tools/batch_analytics.py
import logging
import os
import sys

import hydra
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.utils.hydra_utils import setup_config
from src.utils.logger import setup_logging


load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Батч-аналитика с динамической инициализацией инференс-пайплайна."""
    cfg = setup_config(cfg)

    db_url = os.getenv("DB_CONN")
    if not db_url:
        logger.warning("DB_CONN не задан — используем моковые данные.")

    pipeline_name = cfg.pipeline_name
    pipeline_cfg = getattr(cfg, pipeline_name)
    logger.info("Запуск батч-аналитики для пайплайна: %s", pipeline_name)

    # ── 1. Инициализация пайплайна ────────────────────────────────────────
    try:
        logger.info("Сборка инференс-пайплайна через Hydra...")
        pipeline = hydra.utils.instantiate(pipeline_cfg.inference, cfg=cfg)
    except Exception as e:
        logger.exception("Не удалось инициализировать пайплайн инференса: %s", e)
        sys.exit(1)

    # ── 2. Подготовка данных ──────────────────────────────────────────────
    input_column = "text" if "rag" in pipeline_name else "prompt"

    df = pd.DataFrame(
        {
            "id": [1, 2],
            input_column: [
                "Напиши краткое саммари для новости о снижении ключевой ставки."
                if "decoder" in pipeline_name
                else "Текст статьи про машинное обучение для RAG базы.",
                "Объясни, что такое градиентный спуск."
                if "decoder" in pipeline_name
                else "Описание процесса векторизации данных.",
            ],
        }
    )

    logger.info("Запуск батч-обработки (%d записей)...", len(df))

    # ── 3. Инференс ───────────────────────────────────────────────────────
    results = pipeline(df[input_column].tolist())

    output_column = "embedding" if "rag" in pipeline_name else "generated_text"

    if isinstance(results[0], dict) and output_column in results[0]:
        df[output_column] = [res[output_column] for res in results]
    else:
        df[output_column] = results

    logger.info("Пример результатов:\n%s", df.head())
    logger.info("Батч-аналитика успешно завершена.")


if __name__ == "__main__":
    main()
