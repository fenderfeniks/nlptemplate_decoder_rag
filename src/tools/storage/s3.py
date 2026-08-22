import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError

from src.tools.storage.base import BaseStorage


logger = logging.getLogger(__name__)


class S3Storage(BaseStorage):
    """S3 хранилище с поддержкой многопоточности и атомарных файловых операций.

    Совместимо с AWS S3, Cloudflare R2, MinIO и любым S3-совместимым бэкендом.
    Для MinIO передай endpoint_url="http://localhost:9000".
    Для AWS S3 оставь endpoint_url=None — boto3 использует дефолтный endpoint.
    """

    def __init__(
        self,
        bucket_name: str,
        uri_prefix: str,
        endpoint_url: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        max_concurrency: int = 10,
    ) -> None:
        super().__init__(uri_prefix=uri_prefix)
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

    def upload_file(self, local_path: str | Path, remote_path: str) -> None:
        """Безопасная загрузка одного файла в S3-бакет без удаления других объектов."""
        local_path = Path(local_path)

        if not local_path.exists():
            raise FileNotFoundError(f"Локальный файл не найден: {local_path}")

        s3_key = remote_path.lstrip("/")

        logger.info("S3: загрузка файла %s -> s3://%s/%s", local_path, self.bucket_name, s3_key)

        try:
            # Исправлено: был self.client (AttributeError), должен быть self.s3_client
            self.s3_client.upload_file(
                Filename=str(local_path),
                Bucket=self.bucket_name,
                Key=s3_key,
                Config=self.transfer_config,
            )
        except ClientError as e:
            logger.error("Сбой загрузки файла в S3: %s", e)
            raise

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

        logger.info("Успешно загружена директория в S3: %s", remote_path)

    def download(self, remote_path: str, local_dir: Path | str) -> Path:
        target_path = Path(local_dir)
        remote_path = remote_path.strip("/")

        # Проверяем, является ли remote_path одиночным файлом
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=remote_path)
            # Если не упало — это файл, а не директория
            if target_path.suffix == "":
                target_path = target_path / Path(remote_path).name
            return self.download_file(remote_path, target_path)
        except ClientError as e:
            # 404 — объекта нет, идём скачивать как префикс (директорию).
            # Любая другая ошибка (403, 500) — пробрасываем.
            if e.response["Error"]["Code"] != "404":
                raise

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

            logger.info("Директория атомарно скачана из S3 в: %s", target_path)
            return target_path

        except Exception:
            if tmp_path.exists():
                shutil.rmtree(tmp_path)
            logger.error("Критический сбой скачивания из S3. Временные файлы очищены.")
            raise

    def download_file(self, remote_path: str, local_path: Path | str) -> Path:
        target = Path(local_path)
        remote_path = remote_path.strip("/")
        tmp = target.with_name(target.name + ".tmp")

        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            logger.info("Скачивание файла из S3: s3://%s/%s", self.bucket_name, remote_path)
            self.s3_client.download_file(
                self.bucket_name,
                remote_path,
                str(tmp),
                Config=self.transfer_config,
            )
            if target.exists():
                target.unlink()
            tmp.rename(target)

            logger.info("Файл атомарно скачан из S3 в: %s", target)
            return target

        except Exception as e:
            if tmp.exists():
                tmp.unlink()
            logger.error("Сбой скачивания файла из S3: %s", e)
            raise

    def exists(self, remote_path: str) -> bool:
        """Проверяет существование файла или директории (префикса) в S3."""
        remote_path = remote_path.strip("/")
        try:
            # Сначала проверяем точный ключ (файл)
            self.s3_client.head_object(Bucket=self.bucket_name, Key=remote_path)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] != "404":
                # 403, 500 и т.п. — реальная ошибка, не отсутствие объекта
                logger.error(
                    "Ошибка при проверке существования '%s' в S3: %s",
                    remote_path,
                    e,
                )
                return False

        # Файла нет — проверяем как префикс (директорию)
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=remote_path.rstrip("/") + "/",
                MaxKeys=1,
            )
            return "Contents" in response
        except ClientError as e:
            logger.error(
                "Ошибка при проверке префикса '%s' в S3: %s",
                remote_path,
                e,
            )
            return False
