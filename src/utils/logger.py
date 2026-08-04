import logging
import os
import sys


# Константа на уровне модуля, легко расширять при необходимости
NOISY_LOGGERS = (
    "httpx",
    "huggingface_hub",
    "urllib3",
)


def setup_logging(default_level: int = logging.INFO) -> None:
    """Настраивает централизованное логирование для проекта."""
    env = os.getenv("ENVIRONMENT", "dev")

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    formatter: logging.Formatter

    if env == "prod":
        try:
            from pythonjsonlogger import jsonlogger

            formatter = jsonlogger.JsonFormatter(
                fmt="%(asctime)s %(levelname)s %(name)s %(message)s"
            )
        except ImportError:
            formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s] %(message)s")
    else:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(default_level)

    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
