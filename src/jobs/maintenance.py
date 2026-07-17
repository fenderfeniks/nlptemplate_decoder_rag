import argparse
import logging
import os
import time
import shutil
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Загружаем локальный .env (если запуск вне K8s)
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def backup_qdrant():
    """
    Создает снапшот (резервную копию) векторной базы Qdrant.
    Снапшот сохраняется внутри самого Qdrant (в примонтированном volume),
    откуда его потом можно забрать в S3 или другое хранилище.
    """
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION", "knowledge_base")
    
    logger.info(f"Инициализация бекапа Qdrant для коллекции '{collection_name}' по адресу {qdrant_url}...")
    
    try:
        client = QdrantClient(url=qdrant_url, api_key=api_key)
        
        # Запрашиваем создание снапшота
        snapshot_info = client.create_snapshot(collection_name=collection_name)
        
        logger.info(f"Снапшот успешно создан: {snapshot_info.name}")
        logger.info(f"Размер: {snapshot_info.size / 1024 / 1024:.2f} MB")
    except Exception as e:
        logger.error(f"Критическая ошибка при создании бекапа Qdrant: {e}")
        raise e

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
                    os.remove(item_path)      # Удаление файла
                deleted_count += 1
                logger.debug(f"Удалено: {item_path}")
        except Exception as e:
            logger.error(f"Ошибка при удалении {item_path}: {e}")

    logger.info(f"Очистка завершена. Удалено старых объектов: {deleted_count}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт обслуживания инфраструктуры (Бекапы и Очистка)")
    parser.add_argument("--action", choices=["backup", "cleanup"], required=True, help="Какое действие выполнить")
    parser.add_argument("--days", type=int, default=30, help="Возраст файлов в днях для удаления (только для action=cleanup)")
    args = parser.parse_args()

    if args.action == "backup":
        backup_qdrant()
    elif args.action == "cleanup":
        cleanup_mlruns(args.days)