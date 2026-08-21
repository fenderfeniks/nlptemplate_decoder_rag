"""
Скачивает модели с HF Hub и оформляет единый манифест.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _is_hf_id(model_name_or_path: str) -> bool:
    if "://" in model_name_or_path:
        return False
    if model_name_or_path.startswith(("/", ".", "~")):
        return False
    if "/" in model_name_or_path and not Path(model_name_or_path).exists():
        return True
    return False


def _download_from_hub(repo_id: str, local_dir: Path, force: bool = False) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("pip install huggingface-hub")

    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info("HF Hub: скачивание '%s' -> %s", repo_id, local_dir)
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        force_download=force,
        ignore_patterns=["*.msgpack", "*.h5", "rust_model.ot", "tf_model.h5"],
    )
    return Path(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prepare_artifacts.yaml")
    parser.add_argument("--pipeline", default=None, help="Подготовить только один пайплайн")
    parser.add_argument("--force", action="store_true", help="Перескачать даже если уже есть")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    storage_root = Path(cfg["storage"]["root"])
    uri_prefix = cfg["storage"]["uri_prefix"]
    manifest_filename = cfg.get("manifest_filename", "manifest.json")
    
    artifacts = cfg["artifacts"]
    if args.pipeline:
        artifacts = [a for a in artifacts if a["pipeline_name"] == args.pipeline]
        if not artifacts:
            logger.error("pipeline_name '%s' не найден в конфиге", args.pipeline)
            sys.exit(1)

    # Загружаем существующий манифест, чтобы не затереть данные других пайплайнов
    manifest_path = storage_root / manifest_filename
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            global_manifest = json.load(f)
    else:
        global_manifest = {}

    storage_root.mkdir(parents=True, exist_ok=True)

    for artifact in artifacts:
        pipeline_name = artifact["pipeline_name"]
        model_name_or_path = artifact["model_name_or_path"]
        mlflow_model_name = artifact["mlflow_model_name"]

        # Папка теперь строится по имени пайплайна
        model_local_dir = storage_root / pipeline_name / mlflow_model_name
        logger.info("=== %s | %s ===", pipeline_name, model_name_or_path)

        if model_local_dir.exists() and not args.force:
            logger.info("Веса уже есть: %s. Пропускаем. (--force чтобы перескачать)", model_local_dir)
        elif _is_hf_id(model_name_or_path):
            _download_from_hub(model_name_or_path, model_local_dir, force=args.force)
        else:
            local_path = Path(model_name_or_path)
            if not local_path.exists():
                raise FileNotFoundError(f"Модель не найдена: {local_path}")
            logger.info("Локальный путь — копирование не требуется: %s", local_path)

        # Обновляем секцию манифеста для данного пайплайна
        if pipeline_name not in global_manifest:
            global_manifest[pipeline_name] = {}
            
        global_manifest[pipeline_name].update({
            "load_type": "full_model",
            "model_uri": f"{uri_prefix}{pipeline_name}/{mlflow_model_name}",
            "mlflow_model_name": mlflow_model_name,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "prepare_artifacts.py",
        })

    # Сохраняем обновленный реестр
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(global_manifest, f, indent=4, ensure_ascii=False)
    
    logger.info("Манифест обновлен: %s", manifest_path)
    logger.info("=== Готово. Подготовлено артефактов: %d ===", len(artifacts))


if __name__ == "__main__":
    main()