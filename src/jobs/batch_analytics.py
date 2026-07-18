# src/jobs/batch_analytics.py
import logging
import os

import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import pipeline


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

    logger.info(f"Connecting to database via DB_CONN (Mock)...")

    # 1. Загрузка данных (Заглушка)
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "text": [
                "Отличный сервис, очень помогли с кредитом!",
                "Ужасно медленно работает приложение.",
                "Где найти реквизиты договора?",
            ],
        }
    )
    logger.info(f"Loaded {len(df)} records for batch inference.")

    # 2. Инициализация модели
    # На практике путь должен браться из конфига (например, /app/models/prod)
    model_path = os.getenv("MODEL_PATH", "DeepPavlov/rubert-base-cased")
    device = 0 if torch.cuda.is_available() else -1

    logger.info(f"Loading HF pipeline from {model_path} on device {device}...")
    classifier = pipeline("text-classification", model=model_path, device=device)

    # 3. Батч-инференс
    logger.info("Running batch inference...")
    results = classifier(df["text"].tolist())

    # 4. Сохранение результатов
    df["predictions"] = [res["label"] for res in results]
    df["confidence"] = [res["score"] for res in results]

    logger.info("Sample results:")
    logger.info(f"\n{df.head()}")

    logger.info("Writing results back to target database (Mock)...")
    logger.info("Batch analytics completed successfully.")


if __name__ == "__main__":
    main()
