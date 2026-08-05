import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.tools.storage.resolver import ArtifactResolver
from src.utils.hydra_utils import setup_config
from src.utils.logger import setup_logging


setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def download_all(cfg: DictConfig) -> None:
    """Скачивает веса для Decoder и RAG перед запуском основных сервисов."""
    cfg = setup_config(cfg)
    logger.info("Старт предварительной загрузки артефактов (Init Container)...")

    router = hydra.utils.instantiate(cfg.storage_router)

    # --- 1. Загрузка артефактов Decoder ---
    logger.info("Синхронизация Decoder...")
    decoder_cache = Path(cfg.paths.model_dir) / "decoder_cache"
    decoder_resolver = ArtifactResolver(router=router, cache_base_dir=decoder_cache)
    decoder_manifest = os.getenv(
        "DECODER_MANIFEST_URI", "local://./prod_storage/manifests/decoder_pipeline_manifest.json"
    )
    decoder_resolver.resolve_and_patch(cfg, decoder_manifest, pipeline_name="decoder_pipeline")

    # --- 2. Загрузка артефактов RAG ---
    logger.info("Синхронизация RAG (Энкодер + БД)...")
    rag_cache = Path(cfg.paths.model_dir) / "rag_cache"
    rag_resolver = ArtifactResolver(router=router, cache_base_dir=rag_cache)
    rag_manifest = os.getenv(
        "RAG_MANIFEST_URI", "local://./prod_storage/manifests/rag_pipeline_manifest.json"
    )
    rag_resolver.resolve_and_patch(cfg, rag_manifest, pipeline_name="rag_pipeline")

    logger.info("Все артефакты успешно загружены. Инфраструктура готова к запуску.")


if __name__ == "__main__":
    download_all()
