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

        Патчит напрямую в cfg до instantiate():
        - builder.model_name_or_path    — локальный путь к весам
        - builder.lora_resume_path      — путь к адаптеру (lora-режим) или None
        - modifiers.finetuning.skip_peft — True при full_model, False при lora

        Args:
            cfg: Корневой конфиг Hydra.
            manifest_uri: URI манифеста (s3://, local://, hf://).
            pipeline_name: Имя пайплайна (rag_pipeline или decoder_pipeline).

        Returns:
            db_dir: Путь к скачанной векторной БД (только для RAG), иначе None.
        """
        logger.info("Скачивание манифеста развертывания: %s", manifest_uri)
        manifest = self.router.download_manifest(manifest_uri, self.cache_base / "manifests")

        db_dir = None
        builder_cfg_path = f"{pipeline_name}.model.builder"
        modifier_cfg_path = f"{pipeline_name}.model.modifiers.finetuning"

        # 1. Специфика RAG: Векторная база
        if "vector_db_uri" in manifest:
            db_dir = self.router.download_from_uri(
                manifest["vector_db_uri"], self.cache_base / "vector_db"
            )

        # 2. Логика загрузки весов (общая для RAG и Decoder)
        if manifest["load_type"] == "lora":
            logger.info("Режим: Загрузка базы + LoRA адаптера.")

            base_model_uri = manifest.get("base_model_uri", "")
            if base_model_uri.startswith("hf://"):
                # Убираем схему — оставляем чистый HF id: "hf-internal-testing/tiny-random-BertModel"
                model_name_or_path = base_model_uri[len("hf://") :]
            elif base_model_uri.startswith("local://"):
                model_name_or_path = str(
                    self.router.download_from_uri(base_model_uri, self.cache_base / "base_model")
                )
            else:
                model_name_or_path = base_model_uri

            OmegaConf.update(
                cfg, f"{builder_cfg_path}.model_name_or_path", model_name_or_path, force_add=True
            )

            # LoRA адаптер — качаем через Storage
            lora_path = self.router.download_from_uri(
                manifest["lora_uri"], self.cache_base / "adapter"
            )

        elif manifest["load_type"] == "full_model":
            logger.info("Режим: Загрузка монолитной модели.")
            model_path = self.router.download_from_uri(
                manifest["model_uri"], self.cache_base / "merged_model"
            )
            OmegaConf.update(
                cfg, f"{builder_cfg_path}.model_name_or_path", str(model_path), force_add=True
            )
            OmegaConf.update(cfg, f"{modifier_cfg_path}.skip_peft", True, force_add=True)
            lora_path = None

        return db_dir, lora_path
