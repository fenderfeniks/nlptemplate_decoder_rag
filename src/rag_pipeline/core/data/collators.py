# src/rag_pipeline/core/data/collators.py
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


class IndexingDataCollator:
    """Коллатор для режима подготовки векторной базы (Offline Indexing).
    
    Собирает батчи из текстов (и метаданных) и выполняет динамический 
    паддинг токенизированных представлений (input_ids, attention_mask).
    """

    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        text_column: str = "text"
    ) -> None:
        self.tokenizer = tokenizer
        self.text_column = text_column
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids = [f["input_ids"] for f in features]
        attention_masks = [f["attention_mask"] for f in features]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_masks},
            padding=True,
            return_tensors="pt",
        )

        # Берем имя колонки динамически из конфига
        if self.text_column in features[0]:
            batch["text"] = [f[self.text_column] for f in features]
            
        if "metadata" in features[0]:
            batch["metadata"] = [f["metadata"] for f in features]

        return batch


class ContrastiveDataCollator:
    """Коллатор для режима обучения (Contrastive Learning).
    
    Собирает независимые батчи для запросов (queries) и документов (positives/negatives),
    так как они прогоняются через энкодер раздельно (или через два разных энкодера 
    в архитектуре bi-encoder).
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = {}
        
        # Паддинг для Queries
        q_batch = self.tokenizer.pad(
            {
                "input_ids": [f["query_input_ids"] for f in features],
                "attention_mask": [f["query_attention_mask"] for f in features],
            },
            padding=True,
            return_tensors="pt",
        )
        batch["query_input_ids"] = q_batch["input_ids"]
        batch["query_attention_mask"] = q_batch["attention_mask"]

        # Паддинг для Positive Documents
        p_batch = self.tokenizer.pad(
            {
                "input_ids": [f["pos_input_ids"] for f in features],
                "attention_mask": [f["pos_attention_mask"] for f in features],
            },
            padding=True,
            return_tensors="pt",
        )
        batch["pos_input_ids"] = p_batch["input_ids"]
        batch["pos_attention_mask"] = p_batch["attention_mask"]

        # Паддинг для Negative Documents (если есть)
        if "neg_input_ids" in features[0]:
            n_batch = self.tokenizer.pad(
                {
                    "input_ids": [f["neg_input_ids"] for f in features],
                    "attention_mask": [f["neg_attention_mask"] for f in features],
                },
                padding=True,
                return_tensors="pt",
            )
            batch["neg_input_ids"] = n_batch["input_ids"]
            batch["neg_attention_mask"] = n_batch["attention_mask"]

        return batch