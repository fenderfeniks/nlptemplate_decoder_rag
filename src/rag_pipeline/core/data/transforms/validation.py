# src/rag_pipeline/core/data/transforms/validation.py
import logging
from typing import Any, Optional

from datasets import Dataset as HFDataset
from pydantic import ValidationError

from src.rag_pipeline.core.data.cleaners import TextCleaningPipeline
from src.rag_pipeline.core.data.schemas import RAGIndexingRecord, RAGTrainingRecord
from src.rag_pipeline.core.data.transforms.base import BaseDatasetTransform

logger = logging.getLogger(__name__)


class ValidationTransform(BaseDatasetTransform):
    """Фильтрует датасет через Pydantic-схемы RAG.

    Режимы:
    - ``'indexing'``: проверяет колонку ``text`` и опционально ``metadata``.
    - ``'contrastive'``: проверяет ``query``, ``positive_doc`` и ``negative_doc``.

    Невалидные записи помечаются пустой строкой и удаляются вторым проходом filter,
    что позволяет запускать map параллельно (filter по условию быстрее, чем
    сохранение индексов внутри map).
    """

    _VALID_MODES = ("indexing", "contrastive")

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
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"Неизвестный режим валидации: '{mode}'. "
                f"Допустимые значения: {self._VALID_MODES}"
            )
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
            dataset = dataset.filter(
                lambda x: bool(x[self.text_column]),
                num_proc=self.num_proc,
            )

        elif self.mode == "contrastive":
            dataset = dataset.map(
                self._validate_contrastive_batch,
                batched=True,
                batch_size=self.batch_size,
                num_proc=self.num_proc,
                desc="Validating contrastive records",
            )
            dataset = dataset.filter(
                lambda x: bool(x[self.query_column]),
                num_proc=self.num_proc,
            )

        logger.info(
            "Валидация завершена: %d → %d записей (отброшено %d)",
            initial_count, len(dataset), initial_count - len(dataset),
        )
        return dataset

    def _validate_indexing_batch(
        self, batch: dict[str, list[Any]]
    ) -> dict[str, list[Any]]:
        valid_texts: list[str] = []
        valid_meta: list[dict] = []

        texts = batch.get(self.text_column, [])
        # Если колонки metadata нет — подставляем пустые словари
        meta_col: list = batch.get("metadata", [{}] * len(texts))

        for text, meta in zip(texts, meta_col):
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
        
        # Безопасно извлекаем негативы: если колонка задана и присутствует в батче, берем её, иначе заполняем None
        if self.negative_column and self.negative_column in batch:
            negatives = batch.get(self.negative_column)
        else:
            negatives = [None] * len(queries)

        for q, p, n in zip(queries, positives, negatives):
            try:
                # Pydantic корректно пропустит n, если он None (при условии что в RAGTrainingRecord negative_doc: Optional[str] = None)
                record = RAGTrainingRecord(query=q, positive_doc=p, negative_doc=n)
                valid_queries.append(record.query)
                valid_pos.append(record.positive_doc)
                valid_neg.append(record.negative_doc)
            except ValidationError as e:
                logger.debug("Отброшена битая запись (contrastive): %s", e)
                valid_queries.append("")
                valid_pos.append("")
                valid_neg.append(None)

        # Возвращаем словарь только с теми ключами, которые реально задействованы,
        # либо гарантируем возврат существующей колонки негативов
        result = {
            self.query_column: valid_queries,
            self.positive_column: valid_pos,
        }
        
        # Если колонка негативов ожидалась/существовала, возвращаем её тоже, чтобы Arrow не терял схему
        if self.negative_column:
            result[self.negative_column] = valid_neg

        return result


class CleaningTransform(BaseDatasetTransform):
    """Трансформация для очистки текста через кастомные клинеры.

    Список ``columns_to_clean`` может содержать ``None`` (Hydra интерполирует
    отсутствующие поля как ``null``). Такие значения фильтруются автоматически.
    Колонки, которых нет в датасете, тоже пропускаются без ошибки.
    """

    def __init__(
        self,
        pipeline: TextCleaningPipeline,
        columns_to_clean: list[Optional[str]],
        num_proc: int = 4,
        batch_size: int = 1000,
    ) -> None:
        """
        Args:
            pipeline: Инстанс TextCleaningPipeline с набором клинеров.
            columns_to_clean: Список имён колонок для очистки. Значения None
                (например, от Hydra-интерполяции null-полей) игнорируются.
            num_proc: Число процессов для параллельного map.
            batch_size: Размер батча.
        """
        self.pipeline = pipeline
        # Убираем None сразу при инициализации — нет смысла проверять их каждый раз
        self.columns_to_clean: list[str] = [c for c in columns_to_clean if c is not None]
        self.num_proc = num_proc
        self.batch_size = batch_size

    def __call__(self, dataset: HFDataset) -> HFDataset:
        # Пересекаем с реальными колонками датасета
        active_cols = [c for c in self.columns_to_clean if c in dataset.column_names]

        if not active_cols:
            logger.info(
                "CleaningTransform: ни одна из колонок %s не найдена в датасете — пропущено.",
                self.columns_to_clean,
            )
            return dataset

        logger.info("Применение пайплайна очистки текста по колонкам: %s...", active_cols)

        def _clean_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
            return {
                col: [self.pipeline(t) for t in batch[col]]
                for col in active_cols
            }

        return dataset.map(
            _clean_batch,
            batched=True,
            batch_size=self.batch_size,
            num_proc=self.num_proc,
            desc="Cleaning text",
        )