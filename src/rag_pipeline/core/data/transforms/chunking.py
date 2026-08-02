# src/rag_pipeline/core/data/transforms/chunking.py
import logging
from typing import Any

from datasets import Dataset as HFDataset

from src.rag_pipeline.core.data.transforms.base import BaseDatasetTransform

logger = logging.getLogger(__name__)


class OverlappingChunkingTransform(BaseDatasetTransform):
    """Разбивает длинные тексты на чанки с перекрытием.

    Алгоритм работает на уровне слов (разделитель по умолчанию — пробел),
    чтобы не разрывать слова посередине. Из одной исходной записи
    создаётся N новых записей; остальные колонки (например, metadata) дублируются.

    .. warning:: ``chunk_size`` и ``chunk_overlap`` задаются в **символах**, а не в токенах.
        Один токен в среднем ≈ 4 символа, но это сильно зависит от модели и языка.
        При ``chunk_size=500`` и ``max_length=128`` токенов гарантий непревышения нет.
        Рекомендуется калибровать ``chunk_size`` эмпирически под конкретный токенизатор,
        либо использовать достаточный запас (например, ``chunk_size=400`` при ``max_length=128``).

    .. note:: ``chunk_size`` является мягкой границей: последнее слово чанка добавляется
        целиком, даже если оно незначительно превышает лимит. Реальный размер чанка
        может быть больше ``chunk_size`` на длину одного слова минус один символ.
    """

    def __init__(
        self,
        text_column: str = "text",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separator: str = " ",
        num_proc: int = 4,
        batch_size: int = 1000,
    ) -> None:
        """
        Args:
            text_column: Колонка с исходным текстом.
            chunk_size: Максимальный размер чанка в **символах**. Мягкая граница —
                последнее слово добавляется целиком. Должен быть положительным числом.
            chunk_overlap: Размер перекрытия между соседними чанками в **символах**.
                Должен быть строго меньше ``chunk_size``.
            separator: Разделитель слов. По умолчанию пробел.
            num_proc: Число процессов для параллельного map.
            batch_size: Размер батча для map.

        Raises:
            ValueError: Если ``chunk_size`` не является положительным числом.
            ValueError: Если ``chunk_overlap`` >= ``chunk_size``.
        """
        if chunk_size <= 0:
            raise ValueError(
                f"chunk_size должен быть положительным числом, получено: {chunk_size}"
            )
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) должен быть строго меньше "
                f"chunk_size ({chunk_size})"
            )
        self.text_column = text_column
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.num_proc = num_proc
        self.batch_size = batch_size

    def __call__(self, dataset: HFDataset) -> HFDataset:
        if self.text_column not in dataset.column_names:
            logger.warning(
                "Колонка '%s' не найдена в датасете — чанкинг пропущен. "
                "Убедитесь, что колонка с текстом задана корректно через параметр text_column.",
                self.text_column,
            )
            return dataset

        logger.info(
            "Нарезка на чанки (size=%d симв., overlap=%d симв.) по колонке '%s'...",
            self.chunk_size,
            self.chunk_overlap,
            self.text_column,
        )

        sep_len = len(self.separator)

        def _chunk_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
            new_batch: dict[str, list[Any]] = {k: [] for k in batch.keys()}

            for i in range(len(batch[self.text_column])):
                raw = batch[self.text_column][i]
                # Приводим к строке; пустые значения превращаем в единственный пустой чанк
                text = str(raw) if raw is not None else ""
                words = text.split(self.separator) if text else []

                chunks: list[str] = []
                current_words: list[str] = []
                current_length: int = 0

                for word in words:
                    word_len = len(word) + sep_len

                    if current_length + word_len > self.chunk_size and current_words:
                        chunks.append(self.separator.join(current_words))

                        # Откатываемся назад, набирая слова перекрытия
                        overlap_words: list[str] = []
                        overlap_length: int = 0
                        for w in reversed(current_words):
                            w_len = len(w) + sep_len
                            if overlap_length + w_len <= self.chunk_overlap:
                                overlap_words.insert(0, w)
                                overlap_length += w_len
                            else:
                                break

                        current_words = overlap_words
                        current_length = overlap_length

                    current_words.append(word)
                    current_length += word_len

                # Последний хвост
                if current_words:
                    chunks.append(self.separator.join(current_words))

                # Пустой текст — возвращаем как есть (один пустой чанк)
                if not chunks:
                    chunks = [text]

                # Дублируем все остальные колонки для каждого чанка
                for chunk in chunks:
                    for key in batch.keys():
                        new_batch[key].append(
                            chunk if key == self.text_column else batch[key][i]
                        )

            return new_batch

        initial_count = len(dataset)

        # remove_columns намеренно не передаётся: _chunk_batch сам контролирует
        # состав колонок через new_batch — передача remove_columns=dataset.column_names
        # хрупка при изменении схемы датасета и зависит от версии HF.
        chunked_dataset = dataset.map(
            _chunk_batch,
            batched=True,
            batch_size=self.batch_size,
            num_proc=self.num_proc,
            desc="Chunking documents",
        )

        logger.info(
            "Чанкинг завершён: %d документов → %d чанков",
            initial_count,
            len(chunked_dataset),
        )
        return chunked_dataset