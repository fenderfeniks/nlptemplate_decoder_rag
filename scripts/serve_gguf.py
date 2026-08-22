# scripts/serve_gguf.py
"""Запуск llama-server с автоматическим резолвингом пути к GGUF из манифеста.

Что делает:
    1. Читает манифест через StorageRouter.
    2. Берёт gguf_uri из секции decoder_pipeline.
    3. Резолвит local:// URI в абсолютный путь на диске.
    4. Запускает llama-server с нужными параметрами.

Запуск:
    python scripts/serve_gguf.py

Env-переменные:
    LLAMACPP_DIR     — путь к собранному llama.cpp (по умолчанию ~/llama.cpp)
    MANIFEST_URI     — переопределить URI манифеста из конфига
    PIPELINE_NAME    — секция манифеста (по умолчанию decoder_pipeline)
    LLAMA_PORT       — порт сервера (по умолчанию 8080)
    LLAMA_CTX_SIZE   — размер контекста (по умолчанию 4096)
    LLAMA_GPU_LAYERS — кол-во слоёв на GPU (по умолчанию 0 = CPU only)
    HYDRA_CONFIG_DIR — путь к папке configs/
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


load_dotenv()

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG_DIR = str(_SCRIPT_DIR.parent / "configs")
_DEFAULT_LLAMACPP_DIR = Path.home() / "llama.cpp"
_PIPELINE_NAME = os.getenv("PIPELINE_NAME", "decoder_pipeline")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_llamacpp_dir() -> Path:
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
        "llama.cpp не найден. Укажите путь через env LLAMACPP_DIR или склонируйте в ~/llama.cpp."
    )
    sys.exit(1)


def _find_server_binary(llamacpp_dir: Path) -> Path:
    for name in ("llama-server", "llama-server.exe"):
        # корень (Linux/Mac)
        p = llamacpp_dir / name
        if p.exists():
            return p
        # Windows CMake build
        p = llamacpp_dir / "build" / "bin" / "Release" / name
        if p.exists():
            return p

    import shutil as _shutil

    found = _shutil.which("llama-server")
    if found:
        return Path(found)

    logger.error(
        "Бинарник llama-server не найден в %s. "
        "Пересоберите llama.cpp: cmake -B build && cmake --build build --config Release",
        llamacpp_dir,
    )
    sys.exit(1)


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


def _resolve_local_uri(gguf_uri: str, storage_root: Path) -> Path:
    """Резолвит local://path/to/file в абсолютный путь на диске."""
    if not gguf_uri.startswith("local://"):
        logger.error("Поддерживается только local:// URI, получен: %s", gguf_uri)
        sys.exit(1)
    relative = gguf_uri[len("local://") :]
    return storage_root / relative


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logger.info("=== serve_gguf.py: старт ===")

    llamacpp_dir = _find_llamacpp_dir()
    logger.info("llama.cpp: %s", llamacpp_dir)

    server_bin = _find_server_binary(llamacpp_dir)
    logger.info("Бинарник сервера: %s", server_bin)

    cfg = _load_config()

    manifest_uri: str = os.getenv(
        "MANIFEST_URI",
        OmegaConf.select(cfg, "system.manifest.uri", default=None),
    )
    if not manifest_uri:
        logger.error(
            "manifest_uri не задан ни в конфиге (system.manifest.uri) ни в env MANIFEST_URI."
        )
        sys.exit(1)

    storage_root = Path(OmegaConf.select(cfg, "system.paths.storage_root", default="prod_storage"))
    cache_base = Path(OmegaConf.select(cfg, "system.paths.model_dir", default="/tmp/nlp_cache"))
    router = hydra.utils.instantiate(cfg.system.storage_router)

    # --- Читаем манифест ---
    logger.info("Читаем манифест: %s", manifest_uri)
    full_manifest = router.download_manifest(manifest_uri, cache_base / "manifest")

    if _PIPELINE_NAME not in full_manifest:
        logger.error(
            "Секция '%s' не найдена в манифесте. Доступные: %s",
            _PIPELINE_NAME,
            list(full_manifest.keys()),
        )
        sys.exit(1)

    pipeline_manifest = full_manifest[_PIPELINE_NAME]

    gguf_uri = pipeline_manifest.get("gguf_uri")
    if not gguf_uri:
        logger.error(
            "Поле 'gguf_uri' не найдено в секции '%s'. Сначала запустите prepare_gguf.py.",
            _PIPELINE_NAME,
        )
        sys.exit(1)

    gguf_quant = pipeline_manifest.get("gguf_quant", "unknown")
    logger.info("GGUF URI: %s (quant: %s)", gguf_uri, gguf_quant)

    gguf_path = _resolve_local_uri(gguf_uri, storage_root)
    if not gguf_path.exists():
        logger.error("GGUF файл не найден на диске: %s", gguf_path)
        sys.exit(1)

    logger.info("Путь к модели: %s", gguf_path)

    # --- Параметры сервера ---
    port = int(os.getenv("LLAMA_PORT", "8081"))
    ctx_size = int(os.getenv("LLAMA_CTX_SIZE", "4096"))
    gpu_layers = int(os.getenv("LLAMA_GPU_LAYERS", "0"))

    model_name = pipeline_manifest.get("mlflow_model_name", "")

    cmd = [
        str(server_bin),
        "--model",
        str(gguf_path),
        "--alias",
        model_name,
        "--port",
        str(port),
        "--ctx-size",
        str(ctx_size),
        "--n-gpu-layers",
        str(gpu_layers),
    ]

    logger.info("Запускаем llama-server:")
    logger.info("$ %s", " ".join(cmd))
    logger.info("Сервер будет доступен на http://localhost:%d", port)

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Сервер остановлен.")
    except subprocess.CalledProcessError as e:
        logger.error("llama-server завершился с ошибкой: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
