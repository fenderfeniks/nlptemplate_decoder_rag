import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.hydra_utils import setup_config
from src.utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="promote", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = setup_config(cfg)

    # Инициализация абстрактного логгера через Hydra
    experiment_logger = hydra.utils.instantiate(cfg.system.logger.experiment_logger)

    mlflow_model_name = cfg.model.architecture.get("mlflow_model_name")
    if not mlflow_model_name:
        raise ValueError(
            f"mlflow_model_name не задан в model.architecture для пайплайна '{cfg.pipeline_name}'"
        )
    reg_model_name = f"{mlflow_model_name}_LoRA"
    pipeline_name = cfg.pipeline_name

    # 1. Продвигаем модель через логгер
    try:
        experiment_logger.promote_model(
            reg_model_name=reg_model_name,
            staging_alias="Staging",
            production_alias="Production",
            metric_tag="val_loss"
        )
    except Exception as e:
        logger.error("Ошибка при продвижении модели: %s", e)
        sys.exit(1)

    # 2. Инициализируем хранилище и роутер
    storage_client = hydra.utils.instantiate(cfg.system.storage)
    router = hydra.utils.instantiate(cfg.system.storage_router)
    uri_prefix = cfg.system.storage.uri_prefix
    manifest_uri = cfg.system.manifest.uri

    # 3. Загружаем текущий манифест — он источник истины о базовой модели,
    # а не yaml конфиг который служит только для экспериментов
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        old_manifest_dir = tmp_path / "old_manifest"

        try:
            manifest = router.download_manifest(manifest_uri, cache_dir=old_manifest_dir)
            logger.info("Найден существующий манифест. Обновляем секцию '%s'.", pipeline_name)
        except Exception:
            logger.warning("Существующий манифест не найден. Будет создан новый.")
            manifest = {}

        # 4. Определяем base_model_uri из манифеста, не из конфига
        # Манифест фиксирует откуда реально грузилась базовая модель при обучении
        current = manifest.get(pipeline_name, {})
        base_model_uri = (
            current.get("base_model_uri")       # уже была lora — берём оттуда
            or current.get("model_uri")          # был full_model — это и есть база
            or cfg.model.architecture.get(       # крайний fallback — конфиг
                "base_model_uri",
                f"hf://{cfg.model.architecture.model_name_or_path}"
            )
        )
        logger.info("base_model_uri для манифеста: %s", base_model_uri)

        # 5. Достаем Production-адаптер через логгер
        lora_cfg = OmegaConf.create(
            {
                "enabled": True,
                "model_name": reg_model_name,
                "alias": "Production",
                "artifact_path": cfg.system.logger.registry.artifact_path,
            }
        )
        lora_local_path = experiment_logger.load_adapter(lora_cfg)
        if not lora_local_path:
            logger.error("Не удалось скачать Production LoRA адаптер.")
            sys.exit(1)

        # 6. Загружаем адаптер в Production Storage (S3 / Local)
        remote_adapter_dir = f"adapters/{mlflow_model_name}_prod"
        logger.info("Загрузка адаптера в хранилище: %s", remote_adapter_dir)
        storage_client.upload(local_dir=lora_local_path, remote_path=remote_adapter_dir)

        # 7. Обновляем секцию пайплайна в манифесте
        if pipeline_name not in manifest:
            manifest[pipeline_name] = {}

        manifest[pipeline_name].update({
            "load_type": "lora",
            "base_model_uri": base_model_uri,
            "lora_uri": f"{uri_prefix}adapters/{mlflow_model_name}_prod",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

        # Удаляем ключи от монолитной сборки если были
        manifest[pipeline_name].pop("model_uri", None)

        # 8. upload_file — точечная замена одного файла, не трогает остальное в storage
        manifest_file = tmp_path / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)

        storage_client.upload_file(local_path=manifest_file, remote_path="manifest.json")

    logger.info(
        "Манифест обновлен для пайплайна '%s'. Инференс будет использовать LoRA.",
        pipeline_name,
    )


if __name__ == "__main__":
    main()