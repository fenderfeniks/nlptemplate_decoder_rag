import logging
import os

from dotenv import load_dotenv


# Загружаем локальный .env (если скрипт запущен не в K8s)
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    db_url = os.getenv("DB_CONN")
    if not db_url:
        raise ValueError(
            "Environment variable DB_CONN is not set! Check your .env or K8s variables."
        )

    logger.info("Starting batch analytics...")
    logger.info(f"Connecting to {db_url}...")
    logger.info("Processing reviews...")
    logger.info("Batch analytics completed successfully.")


if __name__ == "__main__":
    main()
