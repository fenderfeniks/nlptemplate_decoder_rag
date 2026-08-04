import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.tools.storage.router import StorageRouter


logger = logging.getLogger(__name__)


class ArtifactResolver:
    """Оркестратор для скачивания артефактов по манифесту и патчинга конфигов Hydra."""

    def __init__(self, router: StorageRouter, cache_base_dir: str | Path) -> None:
        self.router = router
        self.cache_base = Path(cache_base_dir)

    def resolve_and_patch(
        self, cfg: DictConfig, manifest_uri: str, pipeline_name: str = "rag_pipeline"
    ) -> Path | None:
        """Скачивает артефакты и мутирует конфиг.

        Args:
            cfg: Корневой конфиг Hydra.
            manifest_uri: URI манифеста (s3://, local://, hf://).
            pipeline_name: Имя пайплайна (rag_pipeline или decoder_pipeline).

        Returns:
            Path | None: Путь к скачанной векторной БД (только для RAG), иначе None.
        """
        logger.info("Скачивание манифеста развертывания: %s", manifest_uri)
        manifest = self.router.download_manifest(manifest_uri, self.cache_base / "manifests")

        db_dir = None

        # 1. Специфика RAG: Векторная база
        if "vector_db_uri" in manifest:
            db_dir = self.router.download_from_uri(
                manifest["vector_db_uri"], self.cache_base / "vector_db"
            )

        # 2. Логика загрузки весов (общая для RAG и Decoder)
        builder_config_path = f"{pipeline_name}.model.builder"

        if manifest["load_type"] == "lora":
            logger.info("Режим: Загрузка базы + LoRA адаптера.")
            base_path = self.router.download_from_uri(
                manifest["base_model_uri"], self.cache_base / "base_model"
            )
            lora_path = self.router.download_from_uri(
                manifest["lora_uri"], self.cache_base / "adapter"
            )

            OmegaConf.update(
                cfg, f"{builder_config_path}.model_name_or_path", str(base_path), force_add=True
            )
            OmegaConf.update(
                cfg, f"{builder_config_path}.lora_resume_path", str(lora_path), force_add=True
            )

        elif manifest["load_type"] == "full_model":
            logger.info("Режим: Загрузка монолитной модели.")
            model_path = self.router.download_from_uri(
                manifest["model_uri"], self.cache_base / "merged_model"
            )

            OmegaConf.update(
                cfg, f"{builder_config_path}.model_name_or_path", str(model_path), force_add=True
            )
            OmegaConf.update(cfg, f"{builder_config_path}.lora_resume_path", None, force_add=True)

        else:
            raise ValueError(f"Неизвестный тип загрузки в манифесте: {manifest['load_type']}")

        return db_dir
