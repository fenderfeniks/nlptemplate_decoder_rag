import argparse
import logging
import os
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def backup_qdrant():
    logger.info("Triggering Qdrant snapshot...")
    # Тут логика вызова API Qdrant для создания snapshot
    pass

def cleanup_mlruns(days):
    logger.info(f"Cleaning logs older than {days} days...")
    # Тут логика удаления файлов по дате
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["backup", "cleanup"], required=True)
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    if args.action == "backup":
        backup_qdrant()
    elif args.action == "cleanup":
        cleanup_mlruns(args.days)