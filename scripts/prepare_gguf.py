# scripts/decoder/prepare_gguf.py
"""Конвертация decoder-модели в GGUF и обновление манифеста.

Что делает:
    1. Читает манифест и резолвит decoder_pipeline секцию.
    2. Если load_type == "lora" — мержит адаптер в базовую модель и сохраняет merged.
    3. Конвертирует HF-модель в GGUF (float16) через llama.cpp convert_hf_to_gguf.py.
    4. Квантизует GGUF через llama.cpp quantize.
    5. Кладёт GGUF в storage через router.upload_file_to_uri.
    6. Обновляет манифест: добавляет gguf_uri / gguf_quant / gguf_updated_at.
    7. Загружает обновлённый манифест обратно через router.upload_file_to_uri.

Переменная QUANT_TYPE в начале файла управляет квантизацией.

Запуск:
    python scripts/decoder/prepare_gguf.py

Env-переменные:
    HYDRA_CONFIG_DIR   — путь к папке configs/ (по умолчанию ../../configs от скрипта)
    LLAMACPP_DIR       — путь к собранному репозиторию llama.cpp (по умолчанию ~/llama.cpp)
    MANIFEST_URI       — переопределить URI манифеста из конфига
    PIPELINE_NAME      — секция манифеста (по умолчанию decoder_pipeline)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

load_dotenv()

# ---------------------------------------------------------------------------
# Квантизация: Q2_K | Q3_K_M | Q4_0 | Q4_K_M | Q5_K_M | Q6_K | Q8_0
# ---------------------------------------------------------------------------
QUANT_TYPE: str = "f16"

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG_DIR = str(_SCRIPT_DIR.parents[0] / "configs")
_DEFAULT_LLAMACPP_DIR = Path.home() / "llama.cpp"
_PIPELINE_NAME = os.getenv("PIPELINE_NAME", "decoder_pipeline")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# llama.cpp helpers
# ---------------------------------------------------------------------------

def _find_llamacpp_dir() -> Path:
    """Ищет директорию llama.cpp: LLAMACPP_DIR env → ~/llama.cpp → sys.exit."""
    env_val = os.getenv("LLAMACPP_DIR")
    if env_val:
        p = Path(env_val)
        if p.exists():
            return p
        logger.error("LLAMACPP_DIR указывает на несуществующий путь: %s", p)
        sys.exit(1)

    if _DEFAULT_LLAMACPP_DIR.exists():
        return _DEFAULT_LLAMACPP_DIR

    logger.error(
        "Не найден репозиторий llama.cpp.\n"
        "Клонируйте и соберите:\n"
        "  git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp\n"
        "  cd ~/llama.cpp && make -j\n"
        "Или укажите путь: LLAMACPP_DIR=/path/to/llama.cpp"
    )
    sys.exit(1)


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    """Запускает subprocess. При ненулевом returncode — sys.exit."""
    logger.info("$ %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        logger.error("Команда завершилась с кодом %d", result.returncode)
        sys.exit(result.returncode)


def _find_convert_script(llamacpp_dir: Path) -> Path:
    """Ищет скрипт конвертации HF → GGUF (название менялось между версиями llama.cpp)."""
    for name in ("convert_hf_to_gguf.py", "convert.py"):
        p = llamacpp_dir / name
        if p.exists():
            return p
    logger.error(
        "Не найден скрипт конвертации в %s.\n"
        "Ожидается convert_hf_to_gguf.py (актуальные версии) или convert.py (старые).",
        llamacpp_dir,
    )
    sys.exit(1)


def _find_quantize_binary(llamacpp_dir: Path) -> Path:
    for name in ("llama-quantize", "quantize", "llama-quantize.exe", "quantize.exe"):
        # корень (Linux/Mac)
        p = llamacpp_dir / name
        if p.exists():
            return p
        # Windows CMake build
        p = llamacpp_dir / "build" / "bin" / "Release" / name
        if p.exists():
            return p
    import shutil as _shutil
    found = _shutil.which("llama-quantize")
    if found:
        return Path(found)
    logger.error(
        "Не найден бинарник квантизации в %s.\n"
        "Пересоберите llama.cpp: cd %s && make -j",
        llamacpp_dir, llamacpp_dir,
    )
    sys.exit(1)


def _convert_to_gguf_f16(llamacpp_dir: Path, hf_model_dir: Path, output_path: Path) -> None:
    """HF-модель → GGUF float16."""
    script = _find_convert_script(llamacpp_dir)
    _run([
        sys.executable, str(script),
        str(hf_model_dir),
        "--outfile", str(output_path),
        "--outtype", "f16",
    ])


def _quantize_gguf(
    llamacpp_dir: Path,
    input_gguf: Path,
    output_gguf: Path,
    quant_type: str,
) -> None:
    """GGUF float16 → квантизованный GGUF."""
    binary = _find_quantize_binary(llamacpp_dir)
    _run([str(binary), str(input_gguf), str(output_gguf), quant_type], cwd=llamacpp_dir)


# ---------------------------------------------------------------------------
# LoRA merge helper
# ---------------------------------------------------------------------------

def _merge_lora(
    base_model_path: str | Path,
    lora_path: Path,
    merge_output_dir: Path,
) -> Path:
    """Мержит LoRA адаптер в базовую модель, сохраняет merged HF-модель.

    llama.cpp convert_hf_to_gguf.py принимает только monolithic HF-модель —
    PEFT адаптеры он не понимает, поэтому merge обязателен перед конвертацией.
    """
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        logger.error(
            "Для merge LoRA нужны пакеты: pip install peft transformers torch\n%s", e
        )
        sys.exit(1)

    logger.info("Загружаем базовую модель: %s", base_model_path)
    base = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    logger.info("Загружаем LoRA адаптер: %s", lora_path)
    model = PeftModel.from_pretrained(base, str(lora_path))

    logger.info("Мержим и выгружаем в CPU...")
    merged = model.merge_and_unload()

    merge_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Сохраняем merged модель → %s", merge_output_dir)
    merged.save_pretrained(str(merge_output_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(base_model_path)).save_pretrained(str(merge_output_dir))

    logger.info("LoRA merge завершён.")
    return merge_output_dir


# ---------------------------------------------------------------------------
# Манифест helpers
# ---------------------------------------------------------------------------

def _patch_manifest(
    manifest: dict,
    pipeline_name: str,
    gguf_uri: str,
    quant_type: str,
) -> dict:
    """Добавляет GGUF-поля в секцию pipeline_name. Существующие поля не трогает."""
    manifest[pipeline_name].update({
        "gguf_uri": gguf_uri,
        "gguf_quant": quant_type,
        "gguf_updated_at": datetime.now(timezone.utc).isoformat(),
    })
    logger.info("Манифест обновлён: gguf_uri=%s, gguf_quant=%s", gguf_uri, quant_type)
    return manifest


# ---------------------------------------------------------------------------
# Hydra config
# ---------------------------------------------------------------------------

def _load_config():
    config_dir = os.getenv("HYDRA_CONFIG_DIR", _DEFAULT_CONFIG_DIR)
    try:
        GlobalHydra.instance().clear()
    except Exception:
        pass
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = hydra.compose(
            config_name="decoder_api",
            overrides=["inference/generator=local"],
        )
        OmegaConf.resolve(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Основной сценарий
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=== prepare_gguf.py: старт ===")
    logger.info("Квантизация: %s", QUANT_TYPE)

    llamacpp_dir = _find_llamacpp_dir()
    logger.info("llama.cpp: %s", llamacpp_dir)

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

    pipeline_manifest = full_manifest[_PIPELINE_NAME]
    load_type = pipeline_manifest.get("load_type")
    model_name = pipeline_manifest.get("mlflow_model_name", "model")
    logger.info("load_type: %s, model: %s", load_type, model_name)

    with tempfile.TemporaryDirectory(prefix="prepare_gguf_") as tmp_str:
        tmp_dir = Path(tmp_str)
        merge_dir = tmp_dir / "merged"
        gguf_f16_path = tmp_dir / "model_f16.gguf"
        gguf_quant_name = f"{model_name}_{QUANT_TYPE.lower()}.gguf"
        gguf_quant_path = tmp_dir / gguf_quant_name

        # --- Резолвим HF-модель на диске ---
        if load_type == "lora":
            base_model_uri = pipeline_manifest.get("base_model_uri", "")
            lora_uri = pipeline_manifest.get("lora_uri", "")

            if not base_model_uri:
                logger.error("load_type=lora, но base_model_uri отсутствует в манифесте.")
                sys.exit(1)
            if not lora_uri:
                logger.error("load_type=lora, но lora_uri отсутствует в манифесте.")
                sys.exit(1)

            if base_model_uri.startswith("hf://"):
                base_model_path: str | Path = base_model_uri[len("hf://"):]
                logger.info("Базовая модель: HuggingFace Hub (%s)", base_model_path)
            else:
                base_name = base_model_uri.rstrip("/").split("/")[-1]
                base_model_path = router.download_from_uri(
                    base_model_uri, cache_base / f"base_{base_name}"
                )
                logger.info("Базовая модель скачана: %s", base_model_path)

            lora_path = router.download_from_uri(lora_uri, cache_base / "adapter")
            logger.info("LoRA адаптер скачан: %s", lora_path)

            hf_model_dir = _merge_lora(
                base_model_path=base_model_path,
                lora_path=lora_path,
                merge_output_dir=merge_dir,
            )

        elif load_type == "full_model":
            model_uri = pipeline_manifest.get("model_uri", "")
            if not model_uri:
                logger.error("load_type=full_model, но model_uri отсутствует в манифесте.")
                sys.exit(1)

            if model_uri.startswith("hf://"):
                hf_id = model_uri[len("hf://"):]
                logger.info("Модель: HuggingFace Hub (%s)", hf_id)
                try:
                    from huggingface_hub import snapshot_download
                except ImportError:
                    logger.error("pip install huggingface_hub")
                    sys.exit(1)
                local_hf = cache_base / f"hf_{model_name}"
                hf_model_dir = Path(snapshot_download(
                    repo_id=hf_id,
                    local_dir=str(local_hf),
                    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
                ))
                logger.info("Скачано с HF Hub → %s", hf_model_dir)
            else:
                dl_name = model_uri.rstrip("/").split("/")[-1]
                hf_model_dir = router.download_from_uri(
                    model_uri, cache_base / f"model_{dl_name}"
                )
                logger.info("Модель скачана из storage: %s", hf_model_dir)

        else:
            logger.error(
                "Неизвестный load_type='%s'. Поддерживаются: lora, full_model.", load_type
            )
            sys.exit(1)

        # --- HF → GGUF float16 ---
        logger.info("Конвертируем %s → %s", hf_model_dir, gguf_f16_path)
        _convert_to_gguf_f16(llamacpp_dir, hf_model_dir, gguf_f16_path)

        if not gguf_f16_path.exists():
            logger.error("GGUF float16 не создан — конвертация не удалась.")
            sys.exit(1)
        logger.info("GGUF float16: %.1f MB", gguf_f16_path.stat().st_size / 1e6)

        # --- Квантизация ---
        logger.info("Квантизуем → %s (%s)", gguf_quant_path, QUANT_TYPE)
        _quantize_gguf(llamacpp_dir, gguf_f16_path, gguf_quant_path, QUANT_TYPE)

        if not gguf_quant_path.exists():
            logger.error("Квантизованный GGUF не создан.")
            sys.exit(1)
        logger.info("GGUF квантизован: %.1f MB", gguf_quant_path.stat().st_size / 1e6)

        # --- Загружаем GGUF в storage ---
        # URI строим по той же схеме что остальные артефакты:
        # local://decoder_pipeline/<model_name>/<filename>
        gguf_storage_uri = f"local://{_PIPELINE_NAME}/{model_name}/{gguf_quant_name}"
        logger.info("Загружаем GGUF в storage: %s", gguf_storage_uri)
        router.upload_file_to_uri(
            local_path=gguf_quant_path,
            uri=gguf_storage_uri,
        )
        logger.info("GGUF загружен.")

        # --- Обновляем манифест ---
        # download_manifest уже скачал файл в cache_base/manifest/<filename>
        manifest_filename = manifest_uri.split("/")[-1]
        manifest_local_path = cache_base / "manifest" / manifest_filename

        # Читаем актуальный файл (тот что download_manifest положил в кеш)
        with open(manifest_local_path, encoding="utf-8") as f:
            raw_manifest = json.load(f)

        raw_manifest = _patch_manifest(
            manifest=raw_manifest,
            pipeline_name=_PIPELINE_NAME,
            gguf_uri=gguf_storage_uri,
            quant_type=QUANT_TYPE,
        )

        # Атомарно записываем обновлённый манифест локально
        tmp_manifest = manifest_local_path.with_suffix(".tmp")
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            json.dump(raw_manifest, f, ensure_ascii=False, indent=4)
        tmp_manifest.replace(manifest_local_path)

        # Загружаем обратно в storage
        logger.info("Загружаем обновлённый манифест → %s", manifest_uri)
        router.upload_file_to_uri(
            local_path=manifest_local_path,
            uri=manifest_uri,
        )
        logger.info("Манифест обновлён в storage.")

    # ---------------------------------------------------------------------------
    # Итог
    # ---------------------------------------------------------------------------
    logger.info(
        "\n=== prepare_gguf.py: готово ===\n"
        "  gguf_uri:   %s\n"
        "  gguf_quant: %s\n\n"
        "Следующий шаг — запустить llama-server:\n"
        "  llama-server \\\n"
        "    --model <локальный_путь_к_gguf> \\\n"
        "    --port 8080 \\\n"
        "    --ctx-size 4096 \\\n"
        "    --n-gpu-layers 35\n\n"
        "Локальный путь к GGUF в storage можно найти через:\n"
        "  python -c \"from src.tools.storage.router import StorageRouter; ...\"\n"
        "или прочитать из конфига system.paths.storage_root",
        gguf_storage_uri,
        QUANT_TYPE,
    )


if __name__ == "__main__":
    main()