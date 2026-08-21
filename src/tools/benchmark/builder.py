# src/tools/benchmark/builder.py
"""BenchmarkBuilder — главный класс построения эталонного датасета RAG.

Архитектура полностью симметрична index_db.py:
  DataModule (indexing mode) -> итерируем чанки
      ↓
  BaseQAGenerator            -> генерируем (question, answer) из чанка
      ↓
  NLIJudge                   -> фильтруем галлюцинации (entailment_score < threshold)
      ↓
  BenchmarkDataset.save_jsonl()  -> сохраняем локально
      ↓
  storage_client.upload()    -> выгружаем в Storage (S3 / Local / HF)
      ↓
  manifest["benchmark_uri"]  -> обновляем манифест (аналог vector_db_uri)

chunk_id рассчитывается через тот же алгоритм что и в KnowledgeBaseIndexer._generate_doc_id,
чтобы ground_truth в бенчмарке точно совпадал с doc_id в векторной БД.

Deduplicate по question_embedding опциональна — включается через max_question_similarity < 1.0.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset as HFDataset
from tqdm import tqdm

from src.tools.benchmark.generator import BaseQAGenerator
from src.tools.benchmark.schema import BenchmarkDataset, BenchmarkRecord
from src.tools.evaluation.judges.nli_judge import NLIJudge
from src.tools.evaluation.schema import EvalInput


logger = logging.getLogger(__name__)


class BenchmarkBuilder:
    """Строит эталонный QA-датасет из индексируемого корпуса.

    Args:
        generator:              Инстанс BaseQAGenerator (API или Local).
        nli_judge:              Инстанс NLIJudge для фильтрации галлюцинаций.
        nli_threshold:          Минимальный entailment score чтобы принять пару.
                                Рекомендуется 0.60–0.75 в зависимости от модели.
        max_samples_per_chunk:  Сколько QA-пар генерировать на один чанк (обычно 1).
                                При >1 чанк используется несколько раз — полезно для
                                коротких корпусов.
        max_question_similarity: Порог косинусного сходства для дедупликации вопросов.
                                 1.0 = дедупликация отключена (дефолт — без энкодера дёшево).
        min_chunk_length:       Чанки короче этого значения (символов) пропускаются —
                                из них сложно сгенерировать нетривиальный вопрос.
    """

    def __init__(
        self,
        generator: BaseQAGenerator,
        nli_judge: NLIJudge,
        nli_threshold: float = 0.60,
        max_samples_per_chunk: int = 1,
        max_question_similarity: float = 1.0,
        min_chunk_length: int = 100,
    ) -> None:
        self.generator = generator
        self.nli_judge = nli_judge
        self.nli_threshold = nli_threshold
        self.max_samples_per_chunk = max_samples_per_chunk
        self.max_question_similarity = max_question_similarity
        self.min_chunk_length = min_chunk_length

    # ------------------------------------------------------------------
    # chunk_id — идентичен KnowledgeBaseIndexer._generate_doc_id
    # Вынесен сюда как статический метод для прозрачности и тестирования.
    # ------------------------------------------------------------------

    @staticmethod
    def compute_chunk_id(text: str, metadata: dict[str, Any]) -> str:
        """SHA-256[:16] по тексту + url + title.

        Алгоритм должен быть побайтово идентичен KnowledgeBaseIndexer._generate_doc_id —
        именно этот chunk_id используется как ground_truth для FNR/Recall@K.
        """
        composite = f"{text}_{metadata.get('url', '')}_{metadata.get('title', '')}"
        return hashlib.sha256(composite.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Фильтрация
    # ------------------------------------------------------------------

    def _is_chunk_too_short(self, text: str) -> bool:
        return len(text.strip()) < self.min_chunk_length

    def _nli_filter(
        self,
        chunk_text: str,
        answer: str,
        chunk_id: str,
    ) -> tuple[bool, float]:
        """Проверяет: answer ∈ entailment(chunk_text).

        premise   = chunk_text (источник факта)
        hypothesis = answer    (утверждение которое должно следовать из источника)

        Returns:
            (passed, score) — True если score >= nli_threshold.
        """
        eval_input = EvalInput(
            prompt=chunk_text,
            response=answer,
            reference=chunk_text,   # premise = chunk
            metadata={"chunk_id": chunk_id},
        )
        result = self.nli_judge.evaluate(eval_input)
        score = result.score or 0.0
        return score >= self.nli_threshold, score

    # ------------------------------------------------------------------
    # Общая логика обработки одного чанка
    # ------------------------------------------------------------------

    def _process_chunk(
        self,
        text: str,
        item_meta: dict[str, Any],
        chunk_id: str,
        stats: dict[str, int],
        generator_model_name: str,
    ) -> list[BenchmarkRecord]:
        """Генерирует и фильтрует QA-пары для одного чанка.

        Вынесено из обоих публичных методов чтобы не дублировать логику NLI-фильтрации
        и счётчики статистики. Возвращает список принятых записей (обычно 0 или 1).
        """
        if self._is_chunk_too_short(text):
            stats["skipped_short"] += 1
            logger.debug("Чанк пропущен (слишком короткий): %d симв.", len(text))
            return []

        records: list[BenchmarkRecord] = []

        for _ in range(self.max_samples_per_chunk):
            qa = self.generator.generate(text)
            if qa is None:
                stats["failed_generation"] += 1
                logger.debug("Генерация не вернула QA для chunk_id=%s", chunk_id)
                continue

            stats["generated"] += 1
            question, answer = qa

            passed, nli_score = self._nli_filter(text, answer, chunk_id)
            if not passed:
                stats["failed_nli"] += 1
                logger.debug(
                    "NLI-фильтр отклонил (score=%.3f < %.3f): chunk_id=%s",
                    nli_score, self.nli_threshold, chunk_id,
                )
                continue

            records.append(BenchmarkRecord(
                chunk_id=chunk_id,
                chunk_text=text,
                question=question,
                answer=answer,
                nli_score=nli_score,
                metadata=item_meta,
                generator_model=generator_model_name,
            ))
            stats["accepted"] += 1

        return records

    def _log_stats(self, stats: dict[str, int]) -> None:
        logger.info(
            "BenchmarkBuilder завершён. "
            "Чанков: %d | Пропущено (короткие): %d | "
            "Сгенерировано: %d | Отклонено (генерация): %d | "
            "Отклонено (NLI): %d | Принято: %d",
            stats["total_chunks"],
            stats["skipped_short"],
            stats["generated"],
            stats["failed_generation"],
            stats["failed_nli"],
            stats["accepted"],
        )
        if stats["accepted"] == 0:
            logger.warning(
                "Бенчмарк пустой. Проверьте nli_threshold (%.2f) и качество генератора.",
                self.nli_threshold,
            )

    # ------------------------------------------------------------------
    # Основные методы
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def build_from_dataset(
        self,
        dataset: HFDataset,
        text_column: str = "text",
        id_column: str = "chunk_id",
        generator_model_name: str = "",
    ) -> BenchmarkDataset:
        """Строит BenchmarkDataset из HF Dataset (после трансформов, до токенизации).

        Это основной метод для build_benchmark.py: DataModule.train_dataset —
        это уже HF Dataset с текстовыми полями, без тензоров.

        chunk_id берётся из колонки id_column если она есть в датасете.
        Если колонки нет — вычисляется через compute_chunk_id() по тексту и
        метаданным (аналог KnowledgeBaseIndexer._generate_doc_id).

        Args:
            dataset:              HF Dataset с колонкой text_column.
            text_column:          Имя колонки с текстом чанка.
            id_column:            Имя колонки с chunk_id (если есть).
            generator_model_name: Для логирования в BenchmarkRecord.generator_model.

        Returns:
            BenchmarkDataset с отфильтрованными QA-парами.
        """
        result = BenchmarkDataset()
        has_id_column = id_column in dataset.column_names

        stats = {
            "total_chunks": 0,
            "skipped_short": 0,
            "generated": 0,
            "failed_generation": 0,
            "failed_nli": 0,
            "accepted": 0,
        }

        for item in tqdm(dataset, desc="Building benchmark", unit="chunk"):
            stats["total_chunks"] += 1

            text: str = item.get(text_column, "") if isinstance(item, dict) else ""
            if not text:
                stats["skipped_short"] += 1
                continue

            item_meta: dict[str, Any] = {
                k: v for k, v in item.items()
                if k not in (text_column, id_column)
            } if isinstance(item, dict) else {}

            if has_id_column:
                chunk_id = str(item[id_column])
            else:
                chunk_id = self.compute_chunk_id(text, item_meta)

            accepted = self._process_chunk(
                text=text,
                item_meta=item_meta,
                chunk_id=chunk_id,
                stats=stats,
                generator_model_name=generator_model_name,
            )
            for record in accepted:
                result.append(record)

        self._log_stats(stats)
        return result

    @torch.inference_mode()
    def build_from_dataloader(
        self,
        dataloader: torch.utils.data.DataLoader,
        text_column: str = "text",
        generator_model_name: str = "",
    ) -> BenchmarkDataset:
        """Строит BenchmarkDataset из DataLoader (устаревший путь).

        Оставлен для обратной совместимости. Новый код должен использовать
        build_from_dataset() — DataLoader отдаёт токенизированные тензоры,
        поэтому text_column там обычно недоступен (нужно смотреть на батч вручную).

        DataLoader должен возвращать батчи с полями:
        - ``text_column`` (str): исходный текст чанка
        - ``metadata`` (list[dict]): метаданные чанка (url, title, ...)
        """
        dataset_obj = BenchmarkDataset()

        stats = {
            "total_chunks": 0,
            "skipped_short": 0,
            "generated": 0,
            "failed_generation": 0,
            "failed_nli": 0,
            "accepted": 0,
        }

        for batch in tqdm(dataloader, desc="Building benchmark", unit="batch"):
            batch_len = len(batch["input_ids"])
            texts: list[str] = batch.get(text_column, [""] * batch_len)
            raw_metadata: list[dict[str, Any]] = batch.get("metadata") or [
                {} for _ in range(batch_len)
            ]

            for i, text in enumerate(texts):
                stats["total_chunks"] += 1
                item_meta: dict[str, Any] = dict(raw_metadata[i]) if raw_metadata[i] else {}
                chunk_id = self.compute_chunk_id(text, item_meta)

                accepted = self._process_chunk(
                    text=text,
                    item_meta=item_meta,
                    chunk_id=chunk_id,
                    stats=stats,
                    generator_model_name=generator_model_name,
                )
                for record in accepted:
                    dataset_obj.append(record)

        self._log_stats(stats)
        return dataset_obj