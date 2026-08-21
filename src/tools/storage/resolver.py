import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.tools.storage.router import StorageRouter


logger = logging.getLogger(__name__)

# Схемы URI, для которых не нужно скачивать файлы из storage.
# Данные уже доступны на сервере — передаём URI напрямую в connect().
_SERVER_PERSISTENT_SCHEMES = ("qdrant://", "qdrant+memory://")


def _is_server_persistent_uri(uri: str) -> bool:
    """True если URI указывает на серверную БД (не файловый storage)."""
    return any(uri.startswith(scheme) for scheme in _SERVER_PERSISTENT_SCHEMES)


class ArtifactResolver:
    """Оркестратор для скачивания артефактов по манифесту и патчинга конфигов Hydra."""

    def __init__(self, router: StorageRouter, cache_base_dir: str | Path) -> None:
        self.router = router
        self.cache_base = Path(cache_base_dir)

    def get_model_name(self, manifest_uri: str, pipeline_name: str) -> str:
        """Читает mlflow_model_name из манифеста без скачивания артефактов.

        Используется когда нужно только имя модели — например для декодера,
        где модель уже запущена в llama.cpp/vLLM и патчить конфиг не нужно.

        Args:
            manifest_uri:   URI манифеста в storage.
            pipeline_name:  Ключ секции в манифесте (например "decoder_pipeline").

        Returns:
            Значение mlflow_model_name или "unknown" если поле отсутствует.

        Raises:
            KeyError: Если pipeline_name не найден в манифесте.
        """
        full_manifest = self.router.download_manifest(manifest_uri, self.cache_base)
        if pipeline_name not in full_manifest:
            raise KeyError(f"Пайплайн '{pipeline_name}' не найден в манифесте {manifest_uri}")
        model_name = full_manifest[pipeline_name].get("mlflow_model_name", "unknown")
        logger.info("mlflow_model_name для '%s': %s", pipeline_name, model_name)
        return model_name

    def _patch_model_path(self, cfg: DictConfig, model_name_or_path: str) -> None:
        """Патчит все пути к модели и токенизатору в плоском конфиге."""
        patches = [
            "model.builder.model_name_or_path",
            "model.tokenizer.tokenizer_name",
            "model.architecture.model_name_or_path",
        ]
        for path in patches:
            try:
                OmegaConf.update(cfg, path, model_name_or_path, force_add=True)
                logger.debug("Пропатчен путь '%s' -> %s", path, model_name_or_path)
            except Exception as e:
                logger.warning("Не удалось пропатчить '%s': %s", path, e)

    def _resolve_base_model_uri(self, base_model_uri: str, cache_subdir: str) -> str:
        """Разрешает URI базовой модели в локальный путь или HF-идентификатор."""
        if base_model_uri.startswith("hf://"):
            return base_model_uri[len("hf://"):]
        elif base_model_uri.startswith("local://"):
            model_name = base_model_uri.rstrip("/").split("/")[-1]
            local_path = self.router.download_from_uri(
                base_model_uri, self.cache_base / cache_subdir
            )
            return str(local_path)
        else:
            return base_model_uri

    def resolve_and_patch(
        self,
        cfg: DictConfig,
        manifest_uri: str,
        pipeline_name: str = "rag_pipeline",
        is_training: bool = False,
    ) -> tuple[Path | str | None, Path | None, Path | None]:
        """Скачивает артефакты и мутирует конфиг.

        Args:
            cfg: Корневой конфиг Hydra (плоская структура: cfg.model, cfg.data, cfg.training).
            manifest_uri: URI манифеста в storage.
            pipeline_name: Ключ секции в манифесте. Не влияет на пути в конфиге.
            is_training: True — PEFT остаётся активным; False — skip_peft=True.

        Returns:
            (db_dir, lora_path, benchmark_dir) — None если артефакт отсутствует в манифесте.

            ``db_dir`` может быть:
            - ``Path`` — локальная директория (FAISS, скачано из storage).
            - ``str``  — URI серверной БД вида ``qdrant://<url>/<collection>``
              (Qdrant и другие серверные бэкенды). Передаётся напрямую в
              ``hydra.utils.instantiate(cfg.vector_db.loader, directory=db_dir)``,
              где ``QdrantVectorStore.connect()`` распарсит его.
            - ``None`` — vector_db_uri отсутствует в манифесте.
        """
        logger.info("Скачивание реестра манифестов: %s", manifest_uri)
        full_manifest = self.router.download_manifest(manifest_uri, self.cache_base)

        if pipeline_name not in full_manifest:
            raise KeyError(f"Пайплайн '{pipeline_name}' не найден в манифесте {manifest_uri}")

        manifest = full_manifest[pipeline_name]
        logger.debug("Манифест для '%s': %s", pipeline_name, manifest)

        db_dir: Path | str | None = None
        lora_path = None
        benchmark_dir = None

        # --- Векторная база ---
        if "vector_db_uri" in manifest:
            vector_db_uri: str = manifest["vector_db_uri"]

            if _is_server_persistent_uri(vector_db_uri):
                # Qdrant и другие серверные БД: данные уже на сервере.
                # Не скачиваем ничего — передаём URI как есть в connect().
                db_dir = vector_db_uri
                logger.info(
                    "Vector DB серверная (%s) — скачивание пропущено, "
                    "URI будет передан в connect(): %s",
                    vector_db_uri.split("://")[0],
                    vector_db_uri,
                )
            else:
                # FAISS и другие файловые бэкенды: скачиваем из storage.
                db_dir = self.router.download_from_uri(
                    vector_db_uri, self.cache_base / "vector_db"
                )
                logger.info("Vector DB загружена из storage: %s", db_dir)

        # --- BM25 ---
        if "bm25_uri" in manifest:
            bm25_dir = self.router.download_from_uri(
                manifest["bm25_uri"], self.cache_base / "bm25"
            )
            bm25_index_path = str(bm25_dir / "bm25_index.pkl")
            OmegaConf.update(cfg, "inference.bm25.index_path", bm25_index_path, force_add=True)
            logger.info("BM25 загружен. Путь пропатчен: %s", bm25_index_path)

        # --- Бенчмарк ---
        if "benchmark_uri" in manifest:
            benchmark_dir = self.router.download_from_uri(
                manifest["benchmark_uri"], self.cache_base / "benchmark"
            )
            logger.info("Бенчмарк загружен из storage: %s", benchmark_dir)

        # --- Веса модели ---
        load_type = manifest.get("load_type")

        if load_type == "lora":
            logger.info("Режим: Загрузка базы + LoRA адаптера.")
            base_model_uri = manifest.get("base_model_uri", "")
            model_name = base_model_uri.rstrip("/").split("/")[-1]
            model_name_or_path = self._resolve_base_model_uri(
                base_model_uri, cache_subdir=f"base_{model_name}"
            )
            self._patch_model_path(cfg, model_name_or_path)
            lora_path = self.router.download_from_uri(
                manifest["lora_uri"], self.cache_base / "adapter"
            )
            logger.info("LoRA адаптер загружен: %s", lora_path)

        elif load_type == "full_model":
            logger.info("Режим: Загрузка монолитной модели (базовые веса).")
            model_name = manifest["model_uri"].rstrip("/").split("/")[-1]
            model_path = self.router.download_from_uri(
                manifest["model_uri"], self.cache_base / f"merged_{model_name}"
            )
            self._patch_model_path(cfg, str(model_path))

            if not is_training:
                logger.info("Инференс: принудительно отключаем PEFT (skip_peft=True).")
                OmegaConf.update(cfg, "model.modifiers.finetuning.skip_peft", True, force_add=True)
            else:
                logger.info("Обучение: PEFT оставлен активным.")

        else:
            logger.warning(
                "Ключ 'load_type' отсутствует или неизвестен ('%s'). Веса не пропатчены.",
                load_type,
            )

        return db_dir, lora_path, benchmark_dir