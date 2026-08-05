import gc
import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import hydra
import mlflow
import torch
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
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

    # Инициализируем хранилище и роутер
    storage_client = hydra.utils.instantiate(cfg.storage)
    router = hydra.utils.instantiate(cfg.storage_router)
    uri_prefix = cfg.storage.uri_prefix.rstrip("/")
    uri_prefix = cfg.storage.uri_prefix
    manifest_uri = cfg.manifest.uri

    mlflow_model_name = pipeline_cfg.model.architecture.mlflow_model_name
    reg_model_name = f"{mlflow_model_name}_LoRA"

    # 1. Получаем версию текущей Production модели из MLflow
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    try:
        prod_mv = client.get_model_version_by_alias(reg_model_name, "Production")
        prod_version = prod_mv.version
        logger.info("Текущая Production модель: %s (версия %s)", reg_model_name, prod_version)
    except MlflowException:
        logger.error(
            "Алиас 'Production' не найден для модели '%s'. Слияние отменено.", reg_model_name
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

        # 3.1 Токенизатор и Базовая модель
        tokenizer = hydra.utils.instantiate(pipeline_cfg.model.tokenizer).build()
        OmegaConf.update(cfg, f"{cfg.pipeline_name}.model.builder.modifiers", None, force_add=True)
        builder = hydra.utils.instantiate(pipeline_cfg.model.builder)
        base_model = builder.build(tokenizer=tokenizer)

        # 3.2 Подготовка конфига для поиска адаптера
        lora_cfg = OmegaConf.create(
            {
                "enabled": True,
                "model_name": reg_model_name,
                "alias": "Production",
                "artifact_path": cfg.logger.registry.artifact_path,
            }
        )

        lora_path = resolve_lora_resume_path(lora_cfg, tracking_uri=tracking_uri)
        if not lora_path:
            raise FileNotFoundError(
                f"Не найден LoRA адаптер (Production) для {lora_cfg.model_name}"
            )

        # 3.3 Навешивание и слияние
        logger.info("Слияние весов (Merge and Unload)...")
        model = PeftModel.from_pretrained(base_model, lora_path)
        merged_model = model.merge_and_unload()

        if hasattr(merged_model, "generation_config") and getattr(
            merged_model.generation_config, "pad_token_id", None
        ) in (None, -1):
            merged_model.generation_config.pad_token_id = (
                tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        # 3.4 Сохранение локально перед выгрузкой
        output_path = Path(cfg.paths.model_dir) / f"merged_{mlflow_model_name}_v{prod_version}"
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Локальное сохранение монолитной модели в: %s", output_path)
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        # 3.5 Загрузка монолита в Storage
        logger.info("Выгрузка монолита в Storage: %s", remote_merged_dir)
        storage_client.upload(local_dir=output_path, remote_path=remote_merged_dir)

        # 3.6 Очистка памяти GPU
        del model, merged_model, base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Безопасное обновление Манифеста (выполняется всегда, чтобы гарантировать консистентность)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        try:
            manifest = router.download_manifest(manifest_uri, cache_dir=tmp_path / "old_manifest")
            logger.info("Найден существующий манифест. Обновляем ключи модели.")
        except Exception:
            logger.warning("Существующий манифест не найден. Будет создан новый.")
            manifest = {}

        manifest["load_type"] = "full_model"
        manifest["model_uri"] = (
            f"{uri_prefix}merged_models/{mlflow_model_name}_prod_v{prod_version}"
        )
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Удаляем ключи от раздельной сборки
        manifest.pop("base_model_uri", None)
        manifest.pop("lora_uri", None)

        manifest_file = tmp_path / f"{cfg.pipeline_name}_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)

        storage_client.upload(local_dir=tmp_dir, remote_path="manifests")

    logger.info(
        "Манифест обновлен. Инференс будет использовать полную модель. Путь: %s/%s",
        uri_prefix,
        f"manifests/{cfg.pipeline_name}_manifest.json",
    )


if __name__ == "__main__":
    import re
    from pathlib import Path

    # 1. Пробуем взять из CLI
    pipeline_name = next(
        (arg.split("=")[1] for arg in sys.argv if arg.startswith("pipeline_name=")),
        None,
    )

    # 2. Если не передан — читаем дефолт из main.yaml напрямую
    if pipeline_name is None:
        main_yaml = Path(__file__).parents[2] / "configs" / "main.yaml"
        match = re.search(r"^pipeline_name:\s*['\"]?(\w+)['\"]?", main_yaml.read_text(), re.M)
        if match:
            pipeline_name = match.group(1)
        else:
            raise RuntimeError("Не удалось определить pipeline_name из main.yaml")
        sys.argv.append(f"pipeline_name={pipeline_name}")

    # 3. Переопределяем finetuning на full для найденного пайплайна
    finetuning_key = f"{pipeline_name}/model/modifiers/finetuning="
    if not any(arg.startswith(finetuning_key) for arg in sys.argv):
        sys.argv.append(f"{finetuning_key}full")

    merge_and_export()
