# src/rag_pipeline/core/data/transforms/chunking.py
import logging
from typing import Any

from datasets import Dataset as HFDataset

from src.rag_pipeline.core.data.transforms.base import BaseDatasetTransform

logger = logging.getLogger(__name__)


class OverlappingChunkingTransform(BaseDatasetTransform):
    """Разбивает длинные тексты на чанки заданного размера с перекрытием.
    
    Алгоритм работает на уровне слов (разделитель по умолчанию - пробел), 
    чтобы не разрывать слова посередине. При этом из одной исходной записи
    создается N новых записей, а остальные колонки (например, metadata) дублируются.
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
        self.text_column = text_column
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.num_proc = num_proc
        self.batch_size = batch_size

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap должен быть строго меньше chunk_size")

    def __call__(self, dataset: HFDataset) -> HFDataset:
        if self.text_column not in dataset.column_names:
            logger.warning("Колонка '%s' не найдена, чанкинг пропущен.", self.text_column)
            return dataset

        logger.info(
            "Нарезка на чанки (size=%d, overlap=%d) по колонке '%s'...",
            self.chunk_size, self.chunk_overlap, self.text_column
        )

        def _chunk_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
            new_batch = {k: [] for k in batch.keys()}
            
            for i in range(len(batch[self.text_column])):
                text = str(batch[self.text_column][i])
                words = text.split(self.separator)
                
                chunks = []
                current_chunk_words = []
                current_length = 0
                
                for word in words:
                    word_len = len(word) + len(self.separator)
                    
                    # Если добавление слова превысит лимит и чанк не пустой — сохраняем чанк
                    if current_length + word_len > self.chunk_size and current_chunk_words:
                        chunks.append(self.separator.join(current_chunk_words))
                        
                        # Вычисляем перекрытие (отматываем слова назад)
                        overlap_length = 0
                        overlap_words = []
                        for w in reversed(current_chunk_words):
                            if overlap_length + len(w) + len(self.separator) <= self.chunk_overlap:
                                overlap_words.insert(0, w)
                                overlap_length += len(w) + len(self.separator)
                            else:
                                break
                                
                        current_chunk_words = overlap_words
                        current_length = overlap_length
                        
                    current_chunk_words.append(word)
                    current_length += word_len
                    
                # Добавляем последний хвост
                if current_chunk_words:
                    chunks.append(self.separator.join(current_chunk_words))
                    
                # Если текст был слишком короткий и чанков нет — берем его целиком
                if not chunks:
                    chunks = [text]
                    
                # Дублируем запись для каждого получившегося чанка
                for chunk in chunks:
                    for key in batch.keys():
                        if key == self.text_column:
                            new_batch[key].append(chunk)
                        else:
                            new_batch[key].append(batch[key][i])
                            
            return new_batch

        initial_count = len(dataset)
        chunked_dataset = dataset.map(
            _chunk_batch,
            batched=True,
            batch_size=self.batch_size,
            num_proc=self.num_proc,
            remove_columns=dataset.column_names, # HF требует явного удаления старых при изменении длины батча
            desc="Chunking documents",
        )
        
        logger.info("Чанкинг завершен: %d документов превратились в %d чанков", initial_count, len(chunked_dataset))
        return chunked_dataset