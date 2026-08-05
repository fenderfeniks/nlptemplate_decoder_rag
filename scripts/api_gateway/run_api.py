# scripts/api_gateway/run_api.py
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

# Добавляем корень проекта в sys.path, чтобы uvicorn-воркер нашёл src.*
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.logger import setup_logging  # noqa: E402


setup_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    import uvicorn

    # Gateway обычно вешают на 8080 или 3000 порт
    api_port = int(os.getenv("GATEWAY_API_PORT", "8080"))
    workers = int(os.getenv("GATEWAY_API_WORKERS", "1"))
    log_level = os.getenv("GATEWAY_API_LOG_LEVEL", "info").lower()

    logger.info("Запуск API Gateway: host=0.0.0.0, port=%d, workers=%d", api_port, workers)

    # Используем factory=True, так как в server.py у нас функция create_gateway_app()
    uvicorn.run(
        "src.api_gateway.server:create_gateway_app",
        host="0.0.0.0",
        port=api_port,
        workers=workers,
        reload=False,
        log_level=log_level,
        factory=True,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
