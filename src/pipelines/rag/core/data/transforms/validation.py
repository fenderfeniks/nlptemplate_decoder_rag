# src/pipelines/rag/core/data/transforms/validation.py
import logging
from typing import Any, Optional

from pydantic import ValidationError

from src.pipelines.rag.core.data.schemas import RAGIndexingRecord, RAGTrainingRecord
from src.pipelines.base.core.data.transforms.validation import BaseValidationTransform

logger = logging.getLogger(__name__)

class RAGValidationTransform(BaseValidationTransform):
    """Фильтрует датасет через Pydantic-схемы RAG.
    
    Режимы:
    - 'indexing': проверяет колонку текста и опционально метаданные.
    - 'contrastive': проверяет запрос, позитивный и негативный документы.
    """
    
    _VALID_MODES = ("indexing", "contrastive")

    def __init__(
        self,
        mode: str = "indexing",
        text_column: str = "text",
        query_column: str = "query",
        positive_column: str = "positive_doc",
        negative_column: Optional[str] = "negative_doc",
        num_proc: int = 4,
        batch_size: int = 1000,
    ) -> None:
        self.text_column = text_column
        self.query_column = query_column
        self.positive_column = positive_column
        self.negative_column = negative_column
        
        # Передаем общие параметры в базовый класс, который вызовет _validate_mode
        super().__init__(mode=mode, num_proc=num_proc, batch_size=batch_size)

    def _validate_mode(self) -> None:
        if self.mode not in self._VALID_MODES:
            raise ValueError(
                f"Неизвестный режим валидации: '{self.mode}'. "
                f"Допустимые значения: {self._VALID_MODES}"
            )

    def _get_required_columns(self) -> list[str]:
        """Определяет обязательные колонки в зависимости от режима."""
        if self.mode == "indexing":
            return [self.text_column]
        # Для contrastive логично требовать наличие и запроса, и позитивного документа
        return [self.query_column, self.positive_column]

    def _get_filter_column(self) -> str:
        """Определяет колонку для финальной фильтрации пустых строк."""
        return self.text_column if self.mode == "indexing" else self.query_column

    def _validate_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Маршрутизирует батч в нужный метод обработки."""
        if self.mode == "indexing":
            return self._validate_indexing_batch(batch)
        return self._validate_contrastive_batch(batch)

    # ------------------------------------------------------------------
    # Внутренние функции валидации (специфичные для RAG)
    # ------------------------------------------------------------------

    def _validate_indexing_batch(
        self, batch: dict[str, list[Any]]
    ) -> dict[str, list[Any]]:
        valid_texts: list[str] = []
        valid_meta: list[dict] = []

        texts = batch.get(self.text_column, [])
        meta_col: list = batch.get("metadata", [{}] * len(texts))

        for text, meta in zip(texts, meta_col, strict=True):
            try:
                record = RAGIndexingRecord(text=text, metadata=meta or {})
                valid_texts.append(record.text)
                valid_meta.append(record.metadata)
            except ValidationError as e:
                logger.debug("Отброшена битая запись (indexing): %s", e)
                valid_texts.append("")
                valid_meta.append({})

        return {self.text_column: valid_texts, "metadata": valid_meta}

    def _validate_contrastive_batch(
        self, batch: dict[str, list[Any]]
    ) -> dict[str, list[Any]]:
        valid_queries: list[str] = []
        valid_pos: list[str] = []
        valid_neg: list[Optional[str]] = []

        queries = batch.get(self.query_column, [])
        positives = batch.get(self.positive_column, [])

        if self.negative_column and self.negative_column in batch:
            negatives = batch[self.negative_column]
        else:
            negatives = [None] * len(queries)

        for q, p, n in zip(queries, positives, negatives, strict=True):
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

        result = {
            self.query_column: valid_queries,
            self.positive_column: valid_pos,
        }
        
        if self.negative_column:
            result[self.negative_column] = valid_neg

        return result