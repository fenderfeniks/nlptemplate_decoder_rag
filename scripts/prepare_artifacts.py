# scripts/prepare_artifacts.py
"""
Подготовка артефактов перед стартом обучения или eval (smoke-тест).

Что делает:
  1. Читает pipeline_name из main.yaml (или CLI override)
  2. Берёт model_name_or_path из architecture-конфига выбранного пайплайна
  3. Скачивает базовую модель:
       - hf://  → snapshot_download с HuggingFace Hub → кладёт в storage через storage_client
       - storage URI (local://, s3://) → storage_client.download() в локальный кэш
  4. Собирает манифест с load_type=full_model (без LoRA, без PEFT)
  5. Загружает манифест в storage через storage_client.upload()

Принцип: один storage_client из конфига — и для весов, и для манифеста.
  - storage: source/local → всё в локальной папке prod_storage/
  - storage: source/s3    → весы и манифест идут в S3-бакет

После этого eval.py / smoke-тест стартуют без ошибок:
  - router.download_manifest() найдёт манифест через тот же storage_client
  - ArtifactResolver.resolve_and_patch() войдёт в ветку full_model
  - LoRA/PEFT не активируются

Использование:
  # Дефолтный пайплайн из main.yaml
  python scripts/prepare_artifacts.py

  # Явно указать пайплайн
  python scripts/prepare_artifacts.py pipeline_name=rag_pipeline

  # Другая архитектура без смены пайплайна
  python scripts/prepare_artifacts.py \\
      decoder_pipeline/model/architecture=Qwen3-4B-Instruct-2507

  # Форсировать перезапись даже если весы уже есть в storage
  python scripts/prepare_artifacts.py force_download=true
"""

import json
import logging
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.utils.hydra_utils import setup_config
from src.utils.logger import setup_logging


load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_default_pipeline_name() -> str:
    """Читает pipeline_name из configs/main.yaml без инициализации Hydra."""
    main_yaml = Path(__file__).parents[1] / "configs" / "main.yaml"
    match = re.search(r"^pipeline_name:\s*['\"]?(\w+)['\"]?", main_yaml.read_text(), re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Не удалось определить pipeline_name из main.yaml")


def _get_pipeline_cfg(cfg: DictConfig) -> DictConfig:
    """Возвращает конфиг пайплайна по имени из cfg.pipeline_name."""
    pipeline_name = cfg.pipeline_name
    if not hasattr(cfg, pipeline_name):
        raise ValueError(
            f"Пайплайн '{pipeline_name}' не найден в конфиге. Доступные ключи: {list(cfg.keys())}"
        )
    return getattr(cfg, pipeline_name)


def _is_hf_id(model_name_or_path: str) -> bool:
    """
    True если строка — HuggingFace Hub id вида 'org/model-name'.
    False если это локальный путь или storage URI.
    """
    # Storage URI: local://, s3://, gs://
    if "://" in model_name_or_path:
        return False
    # Абсолютный или относительный путь файловой системы
    if model_name_or_path.startswith(("/", ".", "~")):
        return False
    # HF id содержит ровно один слеш и не существует на ФС
    if "/" in model_name_or_path and not Path(model_name_or_path).exists():
        return True
    return False


def _download_from_hub(
    repo_id: str,
    local_dir: Path,
    force_download: bool = False,
) -> Path:
    """
    Скачивает модель с HuggingFace Hub в local_dir.
    Возвращает путь к скачанным весам.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub не установлен. Выполните: pip install huggingface-hub"
        ) from None

    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "HuggingFace Hub: скачивание '%s' → %s (force=%s)",
        repo_id,
        local_dir,
        force_download,
    )
    local_path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        force_download=force_download,
        ignore_patterns=["*.msgpack", "*.h5", "rust_model.ot", "tf_model.h5"],
    )
    logger.info("Модель скачана в: %s", local_path)
    return Path(local_path)


def _upload_manifest(
    storage_client,
    manifest_data: dict,
    pipeline_name: str,
    manifest_remote_dir: str,
) -> None:
    """
    Сериализует манифест во временную директорию и загружает через storage_client.upload().
    storage_client.upload() принимает (local_dir, remote_path) — поэтому нужна tmpdir.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest_file = Path(tmp_dir) / f"{pipeline_name}_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=4, ensure_ascii=False)

        storage_client.upload(local_dir=tmp_dir, remote_path=manifest_remote_dir)
        logger.info(
            "Манифест загружен в storage: %s/%s_manifest.json",
            manifest_remote_dir,
            pipeline_name,
        )


# ---------------------------------------------------------------------------
# Определение remote-пути для весов в storage
# ---------------------------------------------------------------------------


def _model_remote_path(mlflow_model_name: str) -> str:
    """Путь внутри storage где будут лежать веса базовой модели."""
    return f"base_models/{mlflow_model_name}"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def prepare(cfg: DictConfig) -> None:
    cfg = setup_config(cfg)

    pipeline_name: str = cfg.pipeline_name
    pipeline_cfg = _get_pipeline_cfg(cfg)
    force_download: bool = bool(cfg.get("force_download", False))

    # Единственный storage_client из конфига — он же для весов и для манифеста
    storage_client = hydra.utils.instantiate(cfg.storage)
    uri_prefix: str = cfg.storage.uri_prefix  # "local://" или "s3://"

    arch_cfg = pipeline_cfg.model.architecture
    model_name_or_path: str = arch_cfg.model_name_or_path
    mlflow_model_name: str = arch_cfg.mlflow_model_name

    model_remote_path = _model_remote_path(mlflow_model_name)

    logger.info(
        "=== prepare_artifacts | pipeline: %s | storage: %s ===",
        pipeline_name,
        uri_prefix,
    )
    logger.info("model_name_or_path из конфига: %s", model_name_or_path)

    # ------------------------------------------------------------------
    # 1. Определяем источник и кладём веса в storage
    # ------------------------------------------------------------------

    if _is_hf_id(model_name_or_path):
        # --- Источник: HuggingFace Hub ---
        if storage_client.exists(model_remote_path) and not force_download:
            logger.info(
                "Веса уже есть в storage (%s%s). Скачивание пропущено. "
                "Используйте force_download=true для принудительного обновления.",
                uri_prefix,
                model_remote_path,
            )
        else:
            # Скачиваем во временную папку, затем пушим в storage
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_weights = _download_from_hub(
                    repo_id=model_name_or_path,
                    local_dir=Path(tmp_dir) / mlflow_model_name,
                    force_download=force_download,
                )
                logger.info(
                    "Загрузка весов в storage: %s%s",
                    uri_prefix,
                    model_remote_path,
                )
                storage_client.upload(
                    local_dir=local_weights,
                    remote_path=model_remote_path,
                )

        # URI для манифеста — всегда через тот же storage_client
        model_uri = f"{uri_prefix}{model_remote_path}"

    elif "://" in model_name_or_path:
        # --- Источник: уже storage URI (local:// или s3://) ---
        # Проверяем что путь существует через router чтобы не молча упасть в eval
        router = hydra.utils.instantiate(cfg.storage_router)
        cache_dir = Path(cfg.paths.data_dir) / "weights" / mlflow_model_name
        logger.info("Проверка доступности артефакта: %s", model_name_or_path)
        router.download_from_uri(model_name_or_path, cache_dir)
        model_uri = model_name_or_path

    else:
        # --- Источник: локальный путь на файловой системе ---
        local_path = Path(model_name_or_path)
        if not local_path.exists():
            raise FileNotFoundError(
                f"Локальная модель не найдена: {local_path}\n"
                f"Проверьте model_name_or_path в конфиге архитектуры."
            )

        if storage_client.exists(model_remote_path) and not force_download:
            logger.info(
                "Веса уже есть в storage (%s%s). Загрузка пропущена.",
                uri_prefix,
                model_remote_path,
            )
        else:
            logger.info(
                "Загрузка локальных весов (%s) в storage: %s%s",
                local_path,
                uri_prefix,
                model_remote_path,
            )
            storage_client.upload(
                local_dir=local_path,
                remote_path=model_remote_path,
            )

        model_uri = f"{uri_prefix}{model_remote_path}"

    # ------------------------------------------------------------------
    # 2. Собираем манифест
    # ------------------------------------------------------------------

    # Путь к директории манифеста берём из cfg.manifest.uri
    # Пример: "local://manifests/decoder_pipeline_manifest.json"
    #      → remote_dir = "manifests"
    manifest_uri: str = cfg.manifest.uri
    # Убираем схему и имя файла — нам нужна только remote-директория
    manifest_uri_no_scheme = re.sub(r"^[a-z][a-z0-9+\-.]*://", "", manifest_uri)
    manifest_remote_dir = str(Path(manifest_uri_no_scheme).parent)

    manifest: dict = {
        "load_type": "full_model",
        "model_uri": model_uri,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_name": pipeline_name,
        "generated_by": "prepare_artifacts.py",
        "mlflow_model_name": mlflow_model_name,
    }
    # Чистим LoRA-ключи на случай если переиспользуем существующий манифест
    manifest.pop("base_model_uri", None)
    manifest.pop("lora_uri", None)

    # ------------------------------------------------------------------
    # 3. Загружаем манифест в storage (тот же клиент что и для весов)
    # ------------------------------------------------------------------
    _upload_manifest(
        storage_client=storage_client,
        manifest_data=manifest,
        pipeline_name=pipeline_name,
        manifest_remote_dir=manifest_remote_dir,
    )

    # ------------------------------------------------------------------
    # 4. Итоговый отчёт
    # ------------------------------------------------------------------
    logger.info(
        "=== Готово ===\n"
        "  pipeline    : %s\n"
        "  storage     : %s\n"
        "  model_uri   : %s\n"
        "  manifest    : %s%s/%s_manifest.json\n"
        "  load_type   : full_model (LoRA не используется)\n"
        "\nДальнейший шаг: запустите eval.py или smoke-тест.",
        pipeline_name,
        uri_prefix,
        model_uri,
        uri_prefix,
        manifest_remote_dir,
        pipeline_name,
    )


def _pipeline_has_finetuning_group(pipeline_name: str) -> bool:
    """Проверяет что configs/<pipeline_name>/model/modifiers/finetuning/ существует.

    Только для таких пайплайнов безопасно добавлять override finetuning=full.
    Для внешних пайплайнов (nli_pipeline и др.) эта группа отсутствует —
    попытка override вызовет ошибку Hydra 'Key is not in struct'.
    """
    configs_root = Path(__file__).parents[1] / "configs"
    finetuning_dir = configs_root / pipeline_name / "model" / "modifiers" / "finetuning"
    return finetuning_dir.is_dir()


if __name__ == "__main__":
    pipeline_name = next(
        (arg.split("=")[1] for arg in sys.argv if arg.startswith("pipeline_name=")),
        None,
    )

    if pipeline_name is None:
        pipeline_name = _read_default_pipeline_name()
        sys.argv.append(f"pipeline_name={pipeline_name}")

    # Отключаем LoRA/PEFT модификаторы только если пайплайн поддерживает эту группу.
    # Для NLI, RAG-энкодера и других пайплайнов без modifiers/finetuning — пропускаем.
    finetuning_key = f"{pipeline_name}/model/modifiers/finetuning="
    if _pipeline_has_finetuning_group(pipeline_name) and not any(
        arg.startswith(finetuning_key) for arg in sys.argv
    ):
        sys.argv.append(f"{finetuning_key}full")

    prepare()
