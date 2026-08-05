# scripts/rag/run_api.py
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

# Добавляем корень проекта в sys.path чтобы uvicorn-воркер нашёл src.*
# Это нужно потому что uvicorn запускает ASGI-приложение в отдельном процессе,
# который не наследует PYTHONPATH родительского процесса.
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.logger import setup_logging  # noqa: E402


setup_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    import uvicorn

    api_port = int(os.getenv("RAG_API_PORT", "8001"))
    workers = int(os.getenv("RAG_API_WORKERS", "1"))
    log_level = os.getenv("RAG_API_LOG_LEVEL", "info").lower()

    logger.info("Запуск RAG API: host=0.0.0.0, port=%d, workers=%d", api_port, workers)

    uvicorn.run(
        "src.pipelines.rag.api.rest.server:app",
        host="0.0.0.0",
        port=api_port,
        workers=workers,
        reload=False,
        log_level=log_level,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
