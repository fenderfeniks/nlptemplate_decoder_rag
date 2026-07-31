# src/tools/fetch_data.py
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.utils.hydra_utils import setup_config
from src.utils.logger import setup_logging


setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def fetch_data(cfg: DictConfig) -> None:
    """Универсальный скрипт для сбора новых данных из внешних источников."""
    cfg = setup_config(cfg)
    pipeline_name = cfg.pipeline_name
    pipeline_cfg = getattr(cfg, pipeline_name)

    logger.info("Старт сбора данных для пайплайна: %s", pipeline_name)

    # Директория, куда мы будем складывать сырые дампы (json/csv/parquet)
    # RAGDataModule потом прочитает их отсюда на этапе prepare_data()
    raw_data_dir = Path(pipeline_cfg.data.paths.raw_data_dir)
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # TODO: Реализовать логику получения данных
    # =========================================================================
    # 1. Сходить в базу данных (ClickHouse, PostgreSQL) или внешнее API (Confluence, Jira).
    # 2. Вытащить только инкремент (например, где updated_at > last_sync).
    # 3. Привести к единому формату (title, text, url, metadata).
    # 4. Сохранить в raw_data_dir (например, new_docs_20260731.json).
    # =========================================================================

    logger.info("TODO: Логика выгрузки данных еще не реализована.")

    # Пример заглушки:
    # output_file = raw_data_dir / "latest_dump.json"
    # with open(output_file, "w") as f:
    #     json.dump([{"text": "Пример нового документа", "title": "Test"}], f)
    # logger.info("Данные успешно сохранены в %s", output_file)


if __name__ == "__main__":
    fetch_data()
