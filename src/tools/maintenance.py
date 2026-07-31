# src/tools/maintenance.py
import argparse
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cleanup_mlruns(days: int) -> None:
    """Рекурсивно удаляет файлы и пустые папки старше указанного количества дней.

    Очищает директорию из переменной MLRUNS_DIR (по умолчанию /app/logs).
    Обход рекурсивный: вложенные чекпоинты MLflow тоже удаляются.

    Args:
        days: Возраст файлов в днях для удаления.
    """
    target_dir = Path(os.getenv("MLRUNS_DIR", "/app/logs"))
    logger.info("Запуск очистки в %s (старше %d дней)...", target_dir, days)

    if not target_dir.exists():
        logger.warning("Директория %s не существует. Очистка пропущена.", target_dir)
        return

    cutoff_time = time.time() - days * 24 * 60 * 60
    deleted_files = 0
    deleted_dirs = 0

    # Сначала удаляем файлы рекурсивно (снизу вверх через sorted reverse)
    for item_path in sorted(target_dir.rglob("*"), reverse=True):
        try:
            if not item_path.exists():
                # Уже удалено (например, вместе с родительской папкой)
                continue

            mtime = item_path.stat().st_mtime
            if mtime >= cutoff_time:
                continue

            if item_path.is_file():
                item_path.unlink()
                deleted_files += 1
                logger.debug("Удалён файл: %s", item_path)
            elif item_path.is_dir():
                # Удаляем директорию только если она пуста после чистки файлов
                try:
                    item_path.rmdir()  # Упадёт, если внутри ещё есть свежие файлы
                    deleted_dirs += 1
                    logger.debug("Удалена папка: %s", item_path)
                except OSError:
                    pass  # Папка не пуста — пропускаем

        except Exception as e:
            logger.error("Ошибка при обработке %s: %s", item_path, e)

    logger.info("Очистка завершена. Удалено файлов: %d, папок: %d.", deleted_files, deleted_dirs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Скрипт обслуживания инфраструктуры (очистка старых логов)"
    )
    parser.add_argument(
        "--action",
        choices=["cleanup"],
        required=True,
        help="Действие для выполнения",
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
