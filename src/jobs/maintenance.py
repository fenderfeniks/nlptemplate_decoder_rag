import argparse
import logging
import os
import shutil
import time

from dotenv import load_dotenv

# Загружаем локальный .env (если запуск вне K8s)
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cleanup_mlruns(days: int):
    """
    Удаляет папки и файлы артефактов (MLflow или чекпоинты),
    которые старше указанного количества дней.
    """
    # Путь по умолчанию соответствует нашему mount_path в Airflow и Docker
    target_dir = os.getenv("MLRUNS_DIR", "/app/mlruns")
    logger.info(f"Запуск очистки логов в {target_dir} старше {days} дней...")

    if not os.path.exists(target_dir):
        logger.warning(f"Директория {target_dir} не существует. Очистка пропущена.")
        return

    # Вычисляем timestamp отсечения (текущее время минус days в секундах)
    cutoff_time = time.time() - (days * 24 * 60 * 60)
    deleted_count = 0

    for item in os.listdir(target_dir):
        item_path = os.path.join(target_dir, item)
        try:
            # Получаем время последней модификации файла/папки
            mtime = os.path.getmtime(item_path)

            # Если объект старше точки отсечения — удаляем
            if mtime < cutoff_time:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # Рекурсивное удаление папки
                else:
                    os.remove(item_path)  # Удаление файла
                deleted_count += 1
                logger.debug(f"Удалено: {item_path}")
        except Exception as e:
            logger.error(f"Ошибка при удалении {item_path}: {e}")

    logger.info(f"Очистка завершена. Удалено старых объектов: {deleted_count}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Скрипт обслуживания инфраструктуры (Очистка старых логов)"
    )
    # ИСПРАВЛЕНИЕ: Убрали вариант backup, оставили только cleanup
    parser.add_argument(
        "--action", choices=["cleanup"], required=True, help="Какое действие выполнить"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Возраст файлов в днях для удаления",
    )
    args = parser.parse_args()

    if args.action == "cleanup":
        cleanup_mlruns(args.days)