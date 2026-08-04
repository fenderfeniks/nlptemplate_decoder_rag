import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig

from src.tools.storage.base import BaseStorage


logger = logging.getLogger(__name__)


class S3Storage(BaseStorage):
    """S3 хранилище с поддержкой многопоточности и атомарных файловых операций."""

    def __init__(
        self,
        bucket_name: str,
        endpoint_url: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        max_concurrency: int = 10,
    ) -> None:
        self.bucket_name = bucket_name
        self.max_concurrency = max_concurrency
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        self.transfer_config = TransferConfig(
            multipart_threshold=1024 * 1024 * 25,  # 25 MB
            max_concurrency=max_concurrency,
            multipart_chunksize=1024 * 1024 * 25,
            use_threads=True,
        )

    def upload(self, local_dir: Path | str, remote_path: str) -> None:
        local_path = Path(local_dir)
        if not local_path.is_dir():
            raise NotADirectoryError(f"Путь {local_path} должен быть директорией.")

        remote_path = remote_path.strip("/")
        files_to_upload = []

        for root, _, files in os.walk(local_path):
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(local_path)
                s3_key = f"{remote_path}/{rel_path}".replace("\\", "/")
                files_to_upload.append((file_path, s3_key))

        logger.info("Запуск многопоточной загрузки %d файлов в S3...", len(files_to_upload))

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = {
                executor.submit(
                    self.s3_client.upload_file,
                    str(file_path),
                    self.bucket_name,
                    s3_key,
                    Config=self.transfer_config,
                ): (file_path, s3_key)
                for file_path, s3_key in files_to_upload
            }

            for future in as_completed(futures):
                file_path, s3_key = futures[future]
                try:
                    future.result()
                    logger.debug("Успешно загружен: s3://%s/%s", self.bucket_name, s3_key)
                except Exception as e:
                    logger.error("Сбой загрузки файла %s: %s", file_path, e)
                    raise RuntimeError(f"Сбой загрузки S3: {e}") from e

        logger.info("Успешно загружена модель в S3: %s", remote_path)

    def download(self, remote_path: str, local_dir: Path | str) -> Path:
        target_path = Path(local_dir)
        remote_path = remote_path.strip("/")
        tmp_path = target_path.with_name(target_path.name + ".tmp")

        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)

        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=remote_path)

            files_to_download = []
            for page in pages:
                for obj in page.get("Contents", []):
                    s3_key = obj["Key"]
                    if s3_key.endswith("/"):
                        continue

                    rel_path = s3_key[len(remote_path) :].lstrip("/")
                    local_file = tmp_path / rel_path
                    files_to_download.append((s3_key, local_file))

            if not files_to_download:
                raise FileNotFoundError(f"В S3 не найдены объекты по префиксу: {remote_path}")

            logger.info("Скачивание %d файлов из S3...", len(files_to_download))

            with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
                futures = []
                for s3_key, local_file in files_to_download:
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    futures.append(
                        executor.submit(
                            self.s3_client.download_file,
                            self.bucket_name,
                            s3_key,
                            str(local_file),
                            Config=self.transfer_config,
                        )
                    )

                for future in as_completed(futures):
                    future.result()

            if target_path.exists():
                shutil.rmtree(target_path)
            tmp_path.rename(target_path)

            logger.info("Модель атомарно скачана из S3 в: %s", target_path)
            return target_path

        except Exception as e:
            if tmp_path.exists():
                shutil.rmtree(tmp_path)
            logger.error("Критический сбой скачивания из S3. Временные файлы очищены.")
            raise e
