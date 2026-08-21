import gc
import json
import logging
import sys
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
from src.tools.storage.resolver import ArtifactResolver

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="promote", version_base="1.3")
def merge_and_export(cfg: DictConfig) -> None:
    """Сливает LoRA адаптер с базовой моделью и экспортирует монолит в хранилище."""
    cfg = setup_config(cfg)
    
    # Инициализация абстрактного логгера через Hydra
    experiment_logger = hydra.utils.instantiate(cfg.system.logger.experiment_logger)

    # Инициализируем хранилище и роутер
    storage_client = hydra.utils.instantiate(cfg.system.storage)
    router = hydra.utils.instantiate(cfg.system.storage_router)
    uri_prefix = cfg.system.storage.uri_prefix
    manifest_uri = cfg.system.manifest.uri

    mlflow_model_name = cfg.model.architecture.get("mlflow_model_name")
    if not mlflow_model_name:
        raise ValueError(
            f"mlflow_model_name не задан в model.architecture для пайплайна '{cfg.pipeline_name}'"
        )
    reg_model_name = f"{mlflow_model_name}_LoRA"

    # 1. Получаем версию текущей Production модели через логгер
    try:
        prod_version = experiment_logger.get_production_version(reg_model_name, "Production")
        logger.info("Текущая Production модель: %s (версия %s)", reg_model_name, prod_version)
    except Exception as e:
        logger.error(
            "Алиас 'Production' не найден для модели '%s' или произошла ошибка: %s. Слияние отменено.", 
            reg_model_name, 
            e
        )
        sys.exit(1)

    # 2. Формируем путь с учетом версии для предотвращения перезаписи
    remote_merged_dir = f"merged_models/{mlflow_model_name}_prod_v{prod_version}"

    # 3. Проверяем, существует ли уже эта версия монолита в хранилище
    if storage_client.exists(remote_merged_dir):
        logger.info(
            "Монолитная модель версии v%s уже существует в %s. Процесс слияния и выгрузки пропущен.",
            prod_version,
            remote_merged_dir,
        )
    else:
        logger.info("Монолит версии v%s не найден. Начинаем сборку и слияние...", prod_version)

        # 3.1 Резолвинг базовой модели и адаптера из storage через манифест
        cache_base = Path(cfg.system.paths.model_dir) / f"{cfg.pipeline_name}_cache"
        resolver = ArtifactResolver(router=router, cache_base_dir=cache_base)
        try:
            _, lora_path, _ = resolver.resolve_and_patch(
                cfg, manifest_uri, pipeline_name=cfg.pipeline_name, is_training=False
            )
        except Exception as e:
            logger.critical("Сбой резолвинга артефактов: %s", e)
            sys.exit(1)

        # 3.2 Токенизатор и Базовая модель (пути уже пропатчены резолвером)
        tokenizer = hydra.utils.instantiate(cfg.model.tokenizer).build()

        # Модификаторы отключаем — нужна чистая базовая модель без LoRA для merge
        OmegaConf.update(cfg, "model.builder.modifiers", None, force_add=True)
        builder = hydra.utils.instantiate(cfg.model.builder)
        base_model = builder.build(tokenizer=tokenizer)

        # 3.3 Адаптер уже скачан резолвером из storage — используем его напрямую
        # MLflow здесь не нужен, адаптер берётся из prod_storage/adapters/
        if not lora_path:
            raise FileNotFoundError(
                f"Резолвер не вернул lora_path — проверь 'lora_uri' в манифесте для '{cfg.pipeline_name}'"
            )
        logger.info("LoRA адаптер из storage: %s", lora_path)

        # 3.4 Навешивание и слияние
        logger.info("Слияние весов (Merge and Unload)...")
        model = PeftModel.from_pretrained(base_model, lora_path)
        merged_model = model.merge_and_unload()

        if hasattr(merged_model, "generation_config") and getattr(
            merged_model.generation_config, "pad_token_id", None
        ) in (None, -1):
            merged_model.generation_config.pad_token_id = (
                tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        # 3.5 Сохранение локально перед выгрузкой
        output_path = Path(cfg.system.paths.model_dir) / f"merged_{mlflow_model_name}_v{prod_version}"
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Локальное сохранение монолитной модели в: %s", output_path)
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        # 3.6 Загрузка монолита в Storage
        logger.info("Выгрузка монолита в Storage: %s", remote_merged_dir)
        storage_client.upload(local_dir=output_path, remote_path=remote_merged_dir)

        # 3.7 Очистка памяти GPU
        del model, merged_model, base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Безопасное обновление Манифеста
    pipeline_name = cfg.pipeline_name
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # old_manifest — только для чтения, не попадает в upload
        old_manifest_dir = tmp_path / "old_manifest"

        try:
            manifest = router.download_manifest(manifest_uri, cache_dir=old_manifest_dir)
            logger.info("Найден существующий манифест. Обновляем секцию '%s'.", pipeline_name)
        except Exception:
            logger.warning("Существующий манифест не найден. Будет создан новый.")
            manifest = {}

        # Обновляем только секцию нужного пайплайна, не корень
        if pipeline_name not in manifest:
            manifest[pipeline_name] = {}

        manifest[pipeline_name].update({
            "load_type": "full_model",
            "model_uri": f"{uri_prefix}merged_models/{mlflow_model_name}_prod_v{prod_version}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

        # Удаляем ключи от раздельной сборки если были
        manifest[pipeline_name].pop("base_model_uri", None)
        manifest[pipeline_name].pop("lora_uri", None)

        # upload_file — точечная замена одного файла, не трогает остальное в storage
        manifest_file = tmp_path / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)

        storage_client.upload_file(local_path=manifest_file, remote_path="manifest.json")

    logger.info(
        "Манифест обновлен для пайплайна '%s'. Инференс будет использовать полную модель.",
        pipeline_name,
    )


if __name__ == "__main__":
    merge_and_export()