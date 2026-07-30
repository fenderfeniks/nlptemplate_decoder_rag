# src/rag_pipeline/core/data/transforms/validation.py
import logging
from typing import Any, Optional

from datasets import Dataset as HFDataset
from pydantic import ValidationError

from src.rag_pipeline.core.data.schemas import RAGIndexingRecord, RAGTrainingRecord
from src.rag_pipeline.core.data.transforms.base import BaseDatasetTransform

logger = logging.getLogger(__name__)


class ValidationTransform(BaseDatasetTransform):
    """Фильтрует датасет через Pydantic-схемы RAG.

    Режимы:
    - 'indexing' -> проверяет колонку `text` и опционально `metadata`
    - 'contrastive' -> проверяет `query`, `positive_doc` и `negative_doc`
    """

    def __init__(
        self,
        mode: str = "indexing",
        text_column: str = "text",
        query_column: str = "query",
        positive_column: str = "positive_doc",
        negative_column: str = "negative_doc",
        num_proc: int = 4,
        batch_size: int = 1000,
    ) -> None:
        self.mode = mode
        self.text_column = text_column
        self.query_column = query_column
        self.positive_column = positive_column
        self.negative_column = negative_column
        self.num_proc = num_proc
        self.batch_size = batch_size

    def __call__(self, dataset: HFDataset) -> HFDataset:
        logger.info("Применение Pydantic-валидации (режим: %s)...", self.mode)
        initial_count = len(dataset)

        if self.mode == "indexing":
            dataset = dataset.map(
                self._validate_indexing_batch,
                batched=True,
                batch_size=self.batch_size,
                num_proc=self.num_proc,
                desc="Validating indexing records",
            )
            dataset = dataset.filter(lambda x: bool(x[self.text_column]), num_proc=self.num_proc)
        elif self.mode == "contrastive":
            dataset = dataset.map(
                self._validate_contrastive_batch,
                batched=True,
                batch_size=self.batch_size,
                num_proc=self.num_proc,
                desc="Validating contrastive records",
            )
            dataset = dataset.filter(lambda x: bool(x[self.query_column]), num_proc=self.num_proc)
        else:
            raise ValueError(f"Неизвестный режим валидации: {self.mode}")

        logger.info("Валидация завершена: %d -> %d записей", initial_count, len(dataset))
        return dataset

    def _validate_indexing_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        valid_texts, valid_meta = [], []
        # Если метаданных нет в датасете, создаем пустые словари
        meta_col = batch.get("metadata", [{}] * len(batch[self.text_column]))

        for text, meta in zip(batch.get(self.text_column, []), meta_col):
            try:
                record = RAGIndexingRecord(text=text, metadata=meta)
                valid_texts.append(record.text)
                valid_meta.append(record.metadata)
            except ValidationError as e:
                logger.debug("Отброшена битая запись (indexing): %s", e)
                valid_texts.append("")
                valid_meta.append({})
                
        return {self.text_column: valid_texts, "metadata": valid_meta}

    def _validate_contrastive_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        valid_queries, valid_pos, valid_neg = [], [], []
        
        queries = batch.get(self.query_column, [])
        positives = batch.get(self.positive_column, [])
        negatives = batch.get(self.negative_column, [None] * len(queries))

        for q, p, n in zip(queries, positives, negatives):
            try:
                record = RAGTrainingRecord(query=q, positive_doc=p, negative_doc=n)
                valid_queries.append(record.query)
                valid_pos.append(record.positive_doc)
                valid_neg.append(record.negative_doc)
            except ValidationError as e:
                logger.debug("Отброшена битая запись (contrastive): %s", e)
                valid_queries.append("")
                valid_pos.append("")
                valid_neg.append(None)
                
        return {
            self.query_column: valid_queries,
            self.positive_column: valid_pos,
            self.negative_column: valid_neg
        }


class CleaningTransform(BaseDatasetTransform):
    """Трансформация для очистки текста через кастомные клинеры."""

    def __init__(
        self,
        pipeline,
        columns_to_clean: list[str] = ["text", "query", "positive_doc", "negative_doc"],
        num_proc: int = 4,
        batch_size: int = 1000,
    ) -> None:
        self.pipeline = pipeline
        self.columns_to_clean = columns_to_clean
        self.num_proc = num_proc
        self.batch_size = batch_size

    def __call__(self, dataset: HFDataset) -> HFDataset:
        logger.info("Применение пайплайна очистки текста...")

        # Оставляем в списке только те колонки, которые реально есть в датасете
        active_cols = [col for col in self.columns_to_clean if col in dataset.column_names]

        def _clean_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
            res = {}
            for col in active_cols:
                res[col] = [self.pipeline(t) if t is not None else None for t in batch[col]]
            return res

        return dataset.map(
            _clean_batch,
            batched=True,
            batch_size=self.batch_size,
            num_proc=self.num_proc,
            desc="Cleaning text",
        )