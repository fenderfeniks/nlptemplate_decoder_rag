import gc
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel

from src.utils.hydra_utils import setup_config
from src.utils.logger import setup_logging
from src.utils.mlflow import resolve_lora_resume_path


load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def merge_and_export(cfg: DictConfig) -> None:
    """Сливает LoRA адаптер с базовой моделью и экспортирует монолит в хранилище."""
    cfg = setup_config(cfg)
    pipeline_cfg = getattr(cfg, cfg.pipeline_name)
    tracking_uri = cfg.logger.pylightning.tracking_uri

    storage_client = hydra.utils.instantiate(cfg.storage)
    uri_prefix = cfg.storage.uri_prefix.rstrip("/")

    # 1. Токенизатор и Базовая модель
    logger.info("Сборка базовой модели...")
    tokenizer = hydra.utils.instantiate(pipeline_cfg.model.tokenizer).build()

    builder = hydra.utils.instantiate(pipeline_cfg.model.builder)
    builder.lora_resume_path = None
    base_model = builder.build(tokenizer=tokenizer)

    # 2. Поиск Production LoRA в MLflow
    mlflow_model_name = pipeline_cfg.model.architecture.mlflow_model_name
    lora_cfg = OmegaConf.create(
        {
            "enabled": True,
            "model_name": f"{mlflow_model_name}_LoRA",
            "alias": "Production",
            "artifact_path": cfg.logger.registry.artifact_path,
        }
    )

    lora_path = resolve_lora_resume_path(lora_cfg, tracking_uri=tracking_uri)
    if not lora_path:
        raise FileNotFoundError(f"Не найден LoRA адаптер (Production) для {lora_cfg.model_name}")

    # 3. Навешивание и слияние
    logger.info("Слияние весов (Merge and Unload)...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    merged_model = model.merge_and_unload()

    if hasattr(merged_model, "generation_config") and getattr(
        merged_model.generation_config, "pad_token_id", None
    ) in (None, -1):
        merged_model.generation_config.pad_token_id = (
            tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    # 4. Сохранение локально перед выгрузкой
    output_path = Path(cfg.paths.model_dir) / f"merged_{mlflow_model_name}"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Локальное сохранение монолитной модели в: %s", output_path)
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # 5. Загрузка монолита в Storage
    remote_merged_dir = f"merged_models/{mlflow_model_name}_prod"
    logger.info("Выгрузка монолита в Storage: %s", remote_merged_dir)
    storage_client.upload(local_dir=output_path, remote_path=remote_merged_dir)

    # 6. Очистка GPU
    del model, merged_model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 7. Формирование и выгрузка Манифеста
    manifest = {
        "load_type": "full_model",
        "model_uri": f"{uri_prefix}/{remote_merged_dir}",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    manifest_remote_path = f"manifests/{cfg.pipeline_name}_manifest.json"

    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest_file = Path(tmp_dir) / f"{cfg.pipeline_name}_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)

        storage_client.upload(local_dir=tmp_dir, remote_path="manifests")

    logger.info(
        "Манифест обновлен. Инференс будет использовать полную модель. Путь: %s/%s",
        uri_prefix,
        manifest_remote_path,
    )


if __name__ == "__main__":
    merge_and_export()
