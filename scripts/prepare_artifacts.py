"""
Скачивает модели с HF Hub и оформляет единый манифест.

Storage backend выбирается через uri_prefix в prepare_artifacts.yaml (local:// или s3://).
Router инициализируется напрямую из configs/system/storage/router.yaml без Hydra compose.

Запуск:
    python -m scripts.prepare_artifacts
    python -m scripts.prepare_artifacts --pipeline decoder_pipeline
    python -m scripts.prepare_artifacts --force
"""

import argparse
import json
import logging
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import OmegaConf
import yaml

from src.tools.storage.router import StorageRouter
from src.utils.logger import setup_logging

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_oc_env(obj):
    """Рекурсивно резолвит ${oc.env:VAR, default} в строках через os.environ.

    Это единственный паттерн который используется в router.yaml.
    "null" и None конвертируем в None — boto3 принимает endpoint_url=None.
    """
    if isinstance(obj, str):
        def replace(m):
            parts = m.group(1).split(",", 1)
            var = parts[0].strip()
            default = parts[1].strip() if len(parts) > 1 else None
            val = os.environ.get(var, default)
            if val in (None, "null", " null"):
                return "__NONE__"
            return val.strip()
        result = re.sub(r"\$\{oc\.env:([^}]+)\}", replace, obj)
        return None if result == "__NONE__" else result
    if isinstance(obj, dict):
        return {k: _resolve_oc_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_oc_env(i) for i in obj]
    return obj


def _build_router(config_dir: Path) -> StorageRouter:
    """Читает router.yaml, резолвит env-переменные, инстанциирует StorageRouter."""
    router_yaml_path = config_dir / "system" / "storage" / "router.yaml"
    if not router_yaml_path.exists():
        logger.error("router.yaml не найден: %s", router_yaml_path)
        sys.exit(1)

    raw = yaml.safe_load(router_yaml_path.read_text(encoding="utf-8"))
    resolved = _resolve_oc_env(raw)
    cfg = OmegaConf.create(resolved)
    router = hydra.utils.instantiate(cfg)
    if not isinstance(router, StorageRouter):
        logger.error(
            "Ожидался StorageRouter, получен %s. Проверь _target_ в router.yaml.",
            type(router),
        )
        sys.exit(1)
    return router


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


def _artifact_exists(router: StorageRouter, uri: str) -> bool:
    try:
        client, remote_path = router._get_client_and_path(uri)
        return client.exists(remote_path)
    except Exception as e:
        logger.warning("Не удалось проверить существование '%s': %s", uri, e)
        return False


def _load_manifest(router: StorageRouter, manifest_uri: str, tmp_dir: Path) -> dict:
    try:
        result = router.download_manifest(manifest_uri, tmp_dir)
        logger.info("Загружен существующий манифест из: %s", manifest_uri)
        return result
    except FileNotFoundError:
        logger.info("Манифест не найден — будет создан новый.")
        return {}
    except Exception as e:
        logger.warning("Не удалось загрузить манифест (%s) — начинаем с пустого.", e)
        return {}


def _save_manifest(
    router: StorageRouter,
    manifest: dict,
    manifest_uri: str,
    tmp_dir: Path,
) -> None:
    filename = manifest_uri.rstrip("/").split("/")[-1]
    local_path = tmp_dir / filename
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)
    router.upload_file_to_uri(local_path, manifest_uri)
    logger.info("Манифест сохранён в storage: %s", manifest_uri)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-config", default="configs/prepare_artifacts.yaml")
    parser.add_argument("--pipeline", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config-dir", default=None, help="Путь к папке configs/")
    args = parser.parse_args()

    artifacts_cfg = yaml.safe_load(Path(args.artifacts_config).read_text(encoding="utf-8"))
    artifacts = artifacts_cfg["artifacts"]
    manifest_filename = artifacts_cfg.get("manifest_filename", "manifest.json")
    uri_prefix: str = artifacts_cfg.get("storage", {}).get("uri_prefix", "local://")
    manifest_uri = f"{uri_prefix}{manifest_filename}"

    if args.pipeline:
        artifacts = [a for a in artifacts if a["pipeline_name"] == args.pipeline]
        if not artifacts:
            logger.error("pipeline_name '%s' не найден в конфиге", args.pipeline)
            sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    config_dir = Path(args.config_dir or (script_dir.parent / "configs"))

    router = _build_router(config_dir)

    logger.info("Storage: uri_prefix=%s", uri_prefix)
    logger.info("Manifest URI: %s", manifest_uri)

    tmp_root = Path(tempfile.mkdtemp(prefix="prepare_artifacts_"))
    manifest_tmp = tmp_root / "manifest"
    manifest_tmp.mkdir(parents=True, exist_ok=True)

    global_manifest = _load_manifest(router, manifest_uri, manifest_tmp)

    prepared_count = 0
    for artifact in artifacts:
        pipeline_name = artifact["pipeline_name"]
        model_name_or_path = artifact["model_name_or_path"]
        mlflow_model_name = artifact["mlflow_model_name"]
        artifact_uri = f"{uri_prefix.rstrip('/')}/{pipeline_name}/{mlflow_model_name}"

        logger.info("=== %s | %s ===", pipeline_name, model_name_or_path)

        if not args.force and _artifact_exists(router, artifact_uri):
            logger.info("Уже есть в storage: %s — пропускаем.", artifact_uri)
        else:
            local_download_dir = tmp_root / pipeline_name / mlflow_model_name

            if _is_hf_id(model_name_or_path):
                _download_from_hub(model_name_or_path, local_download_dir, force=args.force)
            else:
                local_path = Path(model_name_or_path)
                if not local_path.exists():
                    logger.error("Модель не найдена: %s", local_path)
                    sys.exit(1)
                local_download_dir = local_path

            logger.info("Загрузка в storage: %s -> %s", local_download_dir, artifact_uri)
            router.upload_dir_to_uri(local_download_dir, artifact_uri)
            logger.info("Загружено: %s", artifact_uri)

        global_manifest.setdefault(pipeline_name, {}).update({
            "load_type": "full_model",
            "model_uri": artifact_uri,
            "mlflow_model_name": mlflow_model_name,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "prepare_artifacts.py",
        })
        prepared_count += 1

    _save_manifest(router, global_manifest, manifest_uri, manifest_tmp)
    logger.info("=== Готово. Артефактов: %d ===", prepared_count)


if __name__ == "__main__":
    main()