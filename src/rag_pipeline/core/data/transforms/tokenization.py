# src/rag_pipeline/core/data/transforms/tokenization.py
import logging
from typing import Any

from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizerBase

from src.rag_pipeline.core.data.transforms.base import BaseDatasetTransform

logger = logging.getLogger(__name__)


class RAGTokenizationTransform(BaseDatasetTransform):
    """Токенизация текстов для RAG пайплайна.
    
    Режимы:
    - 'indexing': Токенизирует колонку text, возвращает input_ids.
    - 'contrastive': Токенизирует query, positive_doc и negative_doc в отдельные тензоры.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mode: str = "indexing",
        text_column: str = "text",
        query_column: str = "query",
        positive_column: str = "positive_doc",
        negative_column: str = "negative_doc",
        max_length: int = 512,
        num_proc: int = 4,
        batch_size: int = 1000,
    ) -> None:
        self.tokenizer = tokenizer
        self.mode = mode
        self.text_column = text_column
        self.query_column = query_column
        self.positive_column = positive_column
        self.negative_column = negative_column
        self.max_length = max_length
        self.num_proc = num_proc
        self.batch_size = batch_size

    def __call__(self, dataset: HFDataset) -> HFDataset:
        logger.info("Токенизация (режим: %s, max_length=%d)...", self.mode, self.max_length)

        def _tokenize_indexing(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
            return self.tokenizer(
                examples[self.text_column],
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )

        def _tokenize_contrastive(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
            res = {}
            
            # Токенизируем запрос
            q_enc = self.tokenizer(
                examples[self.query_column], truncation=True, max_length=self.max_length, add_special_tokens=True
            )
            res["query_input_ids"] = q_enc["input_ids"]
            res["query_attention_mask"] = q_enc["attention_mask"]
            
            # Токенизируем позитивный документ
            p_enc = self.tokenizer(
                examples[self.positive_column], truncation=True, max_length=self.max_length, add_special_tokens=True
            )
            res["pos_input_ids"] = p_enc["input_ids"]
            res["pos_attention_mask"] = p_enc["attention_mask"]
            
            # Если есть негативный документ, токенизируем и его
            if self.negative_column in examples and examples[self.negative_column][0] is not None:
                n_enc = self.tokenizer(
                    examples[self.negative_column], truncation=True, max_length=self.max_length, add_special_tokens=True
                )
                res["neg_input_ids"] = n_enc["input_ids"]
                res["neg_attention_mask"] = n_enc["attention_mask"]
                
            return res

        func = _tokenize_indexing if self.mode == "indexing" else _tokenize_contrastive

        return dataset.map(
            func,
            batched=True,
            batch_size=self.batch_size,
            num_proc=self.num_proc,
            desc=f"Tokenizing ({self.mode})",
            # В отличие от SFT, здесь мы НЕ удаляем оригинальные колонки (remove_columns=False).
            # При индексации нам нужен оригинальный текст и метаданные, чтобы положить их в БД!
        )