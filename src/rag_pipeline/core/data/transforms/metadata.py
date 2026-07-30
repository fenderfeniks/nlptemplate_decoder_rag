# src/rag_pipeline/core/data/transforms/metadata.py
import logging
from typing import Any

from datasets import Dataset as HFDataset

from src.rag_pipeline.core.data.transforms.base import BaseDatasetTransform

logger = logging.getLogger(__name__)


class MetadataInjectorTransform(BaseDatasetTransform):
    """Вклеивает словарь метаданных в начало текста.
    
    Пример:
    {'title': 'Transformer', 'date': '2017'} + 'Текст статьи...' 
    превращается в:
    'Title: Transformer\nDate: 2017\n\nТекст статьи...'
    """

    def __init__(
        self,
        text_column: str = "text",
        metadata_column: str = "metadata",
        template: str = "{meta_string}\n\n{text}",
        num_proc: int = 4,
        batch_size: int = 1000,
    ) -> None:
        self.text_column = text_column
        self.metadata_column = metadata_column
        self.template = template
        self.num_proc = num_proc
        self.batch_size = batch_size

    def __call__(self, dataset: HFDataset) -> HFDataset:
        if self.metadata_column not in dataset.column_names:
            logger.info("Колонка '%s' не найдена, инъекция метаданных пропущена.", self.metadata_column)
            return dataset

        logger.info("Инъекция метаданных в текст документа...")

        def _inject(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
            new_texts = []
            for text, meta in zip(batch[self.text_column], batch[self.metadata_column]):
                if not meta:
                    new_texts.append(text)
                    continue
                    
                meta_parts = [f"{str(k).capitalize()}: {v}" for k, v in meta.items() if v]
                meta_string = "\n".join(meta_parts)
                
                if meta_string:
                    enriched_text = self.template.format(meta_string=meta_string, text=text)
                    new_texts.append(enriched_text)
                else:
                    new_texts.append(text)
                    
            return {self.text_column: new_texts}

        return dataset.map(
            _inject,
            batched=True,
            batch_size=self.batch_size,
            num_proc=self.num_proc,
            desc="Injecting metadata",
        )