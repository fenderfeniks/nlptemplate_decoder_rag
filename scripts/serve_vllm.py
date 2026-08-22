# scripts/serve_vllm.py
"""Запуск vLLM inference server с автоматическим резолвингом модели из манифеста.

Что делает:
    1. Читает манифест через StorageRouter.
    2. Определяет load_type (full_model / lora) из секции decoder_pipeline.
    3. Скачивает модель (и LoRA адаптер если нужно) через ArtifactResolver.
    4. Читает параметры квантизации из манифеста (quantization.method).
    5. Запускает `vllm serve` с нужными флагами.

load_type = full_model:
    → vllm serve <model_path> [--quantization awq/gptq/...]

load_type = lora:
    → vllm serve <base_model_path>
          --enable-lora
          --lora-modules <model_name>=<lora_path>

Запуск:
    python scripts/serve_vllm.py

Env-переменные:
    MANIFEST_URI         — переопределить URI манифеста из конфига
    PIPELINE_NAME        — секция манифеста (по умолчанию decoder_pipeline)
    VLLM_PORT            — порт сервера (по умолчанию 8081)
    VLLM_HOST            — хост сервера (по умолчанию 0.0.0.0)
    VLLM_CTX_SIZE        — размер контекста (по умолчанию 4096)
    VLLM_TENSOR_PARALLEL — tensor_parallel_size, кол-во GPU (по умолчанию 1)
    VLLM_DTYPE           — тип данных: auto/float16/bfloat16 (по умолчанию auto)
    VLLM_GPU_MEMORY_UTIL — доля VRAM под KV-cache, 0.0–1.0 (по умолчанию 0.90)
    VLLM_MAX_MODEL_LEN   — максимальная длина последовательности (опционально)
    VLLM_QUANTIZATION    — переопределить метод квантизации из манифеста
    HYDRA_CONFIG_DIR     — путь к папке configs/
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from src.tools.storage.resolver import ArtifactResolver

load_dotenv()

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG_DIR = str(_SCRIPT_DIR.parent / "configs")
_PIPELINE_NAME = os.getenv("PIPELINE_NAME", "decoder_pipeline")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config():
    config_dir = os.getenv("HYDRA_CONFIG_DIR", _DEFAULT_CONFIG_DIR)
    try:
        GlobalHydra.instance().clear()
    except Exception:
        pass
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = hydra.compose(config_name="decoder_api")
        OmegaConf.resolve(cfg)
    return cfg


def _resolve_local_uri(uri: str, storage_root: Path) -> Path:
    """Резолвит local://path/to/dir в абсолютный путь на диске."""
    if not uri.startswith("local://"):
        logger.error("Поддерживается только local:// URI, получен: %s", uri)
        sys.exit(1)
    relative = uri[len("local://"):]
    return storage_root / relative


def _resolve_model(
    cfg,
    manifest: dict,
    pipeline_name: str,
    storage_root: Path,
    router,
) -> tuple[Path, Path | None, str]:
    """Резолвит модель из манифеста в зависимости от load_type.

    Returns:
        (model_path, lora_path, model_name)
        lora_path — None если load_type=full_model.
    """
    load_type = manifest.get("load_type")
    model_name = manifest.get("mlflow_model_name", pipeline_name)

    cache_base = Path(
        OmegaConf.select(cfg, "system.paths.model_dir", default="/tmp/nlp_cache")
    ) / f"{pipeline_name}_cache"

    resolver = ArtifactResolver(router=router, cache_base_dir=cache_base)

    if load_type == "full_model":
        logger.info("load_type=full_model — резолвим монолитную модель.")
        model_uri = manifest.get("model_uri")
        if not model_uri:
            logger.error("Поле 'model_uri' не найдено в манифесте секции '%s'.", pipeline_name)
            sys.exit(1)

        # Используем router напрямую — нам нужен только путь, без патча конфига.
        # ArtifactResolver.resolve_and_patch патчит cfg.model.builder — это лишнее
        # для serve-скрипта, где конфиг Hydra не используется после старта сервера.
        model_path = router.download_from_uri(
            model_uri,
            cache_base / f"model_{model_name}",
        )
        logger.info("Модель скачана: %s", model_path)
        return model_path, None, model_name

    elif load_type == "lora":
        logger.info("load_type=lora — резолвим базовую модель + LoRA адаптер.")
        base_model_uri = manifest.get("base_model_uri")
        lora_uri = manifest.get("lora_uri")

        if not base_model_uri:
            logger.error("Поле 'base_model_uri' не найдено в манифесте (load_type=lora).")
            sys.exit(1)
        if not lora_uri:
            logger.error("Поле 'lora_uri' не найдено в манифесте (load_type=lora).")
            sys.exit(1)

        # Базовая модель: hf://, local://, s3://
        if base_model_uri.startswith("hf://"):
            # HuggingFace — передаём идентификатор напрямую в vLLM
            base_model_path = Path(base_model_uri[len("hf://"):])
            logger.info("Базовая модель из HuggingFace Hub: %s", base_model_path)
        else:
            base_model_name = base_model_uri.rstrip("/").split("/")[-1]
            base_model_path = router.download_from_uri(
                base_model_uri,
                cache_base / f"base_{base_model_name}",
            )
            logger.info("Базовая модель скачана: %s", base_model_path)

        lora_path = router.download_from_uri(
            lora_uri,
            cache_base / "adapter",
        )
        logger.info("LoRA адаптер скачан: %s", lora_path)
        return base_model_path, lora_path, model_name

    else:
        logger.error(
            "Неизвестный load_type='%s' в манифесте. Поддерживаются: full_model, lora.",
            load_type,
        )
        sys.exit(1)


def _resolve_quantization(manifest: dict) -> str | None:
    """Определяет метод квантизации.

    Приоритет:
        1. VLLM_QUANTIZATION env (явное переопределение)
        2. manifest[pipeline].quantization.method
        3. None (нет квантизации, fp16/bf16)
    """
    # Явное переопределение через env — всегда приоритет
    env_quant = os.getenv("VLLM_QUANTIZATION")
    if env_quant:
        logger.info("Квантизация из env VLLM_QUANTIZATION: %s", env_quant)
        return env_quant

    quant_info = manifest.get("quantization")
    if quant_info and isinstance(quant_info, dict):
        method = quant_info.get("method")
        if method:
            w_bit = quant_info.get("w_bit", "?")
            q_group_size = quant_info.get("q_group_size", "?")
            logger.info(
                "Квантизация из манифеста: method=%s, w_bit=%s, q_group_size=%s",
                method, w_bit, q_group_size,
            )
            return method

    logger.info("Квантизация не задана — vLLM загрузит в dtype из VLLM_DTYPE (default: auto).")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=== serve_vllm.py: старт ===")

    cfg = _load_config()

    manifest_uri: str = os.getenv(
        "MANIFEST_URI",
        OmegaConf.select(cfg, "system.manifest.uri", default=None),
    )
    if not manifest_uri:
        logger.error(
            "manifest_uri не задан ни в конфиге (system.manifest.uri) "
            "ни в env MANIFEST_URI."
        )
        sys.exit(1)

    storage_root = Path(
        OmegaConf.select(cfg, "system.paths.storage_root", default="prod_storage")
    )
    cache_base = Path(
        OmegaConf.select(cfg, "system.paths.model_dir", default="/tmp/nlp_cache")
    )

    router = hydra.utils.instantiate(cfg.system.storage_router)

    # --- Читаем манифест ---
    logger.info("Читаем манифест: %s", manifest_uri)
    full_manifest = router.download_manifest(manifest_uri, cache_base / "manifest")

    if _PIPELINE_NAME not in full_manifest:
        logger.error(
            "Секция '%s' не найдена в манифесте. Доступные: %s",
            _PIPELINE_NAME, list(full_manifest.keys()),
        )
        sys.exit(1)

    manifest = full_manifest[_PIPELINE_NAME]
    logger.info(
        "Манифест прочитан. load_type=%s, model=%s",
        manifest.get("load_type"), manifest.get("mlflow_model_name"),
    )

    # --- Резолвинг модели ---
    model_path, lora_path, model_name = _resolve_model(
        cfg=cfg,
        manifest=manifest,
        pipeline_name=_PIPELINE_NAME,
        storage_root=storage_root,
        router=router,
    )

    # --- Параметры сервера из env ---
    port = int(os.getenv("VLLM_PORT", "8081"))
    host = os.getenv("VLLM_HOST", "0.0.0.0")
    ctx_size = int(os.getenv("VLLM_CTX_SIZE", "4096"))
    tensor_parallel = int(os.getenv("VLLM_TENSOR_PARALLEL", "1"))
    dtype = os.getenv("VLLM_DTYPE", "auto")
    gpu_memory_util = float(os.getenv("VLLM_GPU_MEMORY_UTIL", "0.90"))
    max_model_len = os.getenv("VLLM_MAX_MODEL_LEN")  # опционально, None если не задан

    quantization = _resolve_quantization(manifest)

    # --- Сборка команды ---
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_path),
        "--served-model-name", model_name,
        "--host", host,
        "--port", str(port),
        "--max-model-len", str(ctx_size),
        "--tensor-parallel-size", str(tensor_parallel),
        "--dtype", dtype,
        "--gpu-memory-utilization", str(gpu_memory_util),
    ]

    if quantization:
        cmd += ["--quantization", quantization]

    if max_model_len:
        # VLLM_MAX_MODEL_LEN переопределяет ctx_size если задан явно
        # Полезно когда модель поддерживает длинный контекст но памяти мало
        cmd += ["--max-model-len", max_model_len]

    if lora_path:
        # LoRA в vLLM: --enable-lora + --lora-modules <name>=<path>
        # name используется в запросах через поле model= в OpenAI API
        cmd += [
            "--enable-lora",
            "--lora-modules", f"{model_name}={lora_path}",
        ]
        logger.info("LoRA режим: адаптер будет доступен как модель '%s'", model_name)

    logger.info("Запускаем vLLM:")
    logger.info("$ %s", " ".join(str(c) for c in cmd))
    logger.info("Сервер будет доступен на http://%s:%d", host, port)
    if lora_path:
        logger.info("LoRA адаптер: %s", lora_path)
    if quantization:
        logger.info("Квантизация: %s", quantization)
    if tensor_parallel > 1:
        logger.info("Tensor parallel: %d GPU", tensor_parallel)

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Сервер остановлен.")
    except subprocess.CalledProcessError as e:
        logger.error("vLLM завершился с ошибкой: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()