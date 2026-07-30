# src/core/data/collators.py
import logging
from typing import Any, Optional

import torch
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# ==============================================================================
# RAG коллаторы
# ==============================================================================

class IndexingDataCollator:
    """Коллатор для RAG-индексации (indexing-режим).

    Собирает батч из ``input_ids`` / ``attention_mask`` для прогона через энкодер
    с целью получения эмбеддингов документов. Оригинальные текст и метаданные
    сохраняются в батче для последующей записи в векторную БД.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        text_column: str = "text",
        metadata_column: str = "metadata",
    ) -> None:
        """
        Args:
            tokenizer: Токенизатор модели-энкодера.
            max_length: Максимальная длина последовательности (hard clip).
            text_column: Имя колонки с текстом документа.
            metadata_column: Имя колонки с метаданными (для записи в векторную БД).
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.metadata_column = metadata_column

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Args:
            features: Список словарей с ключами ``input_ids``, ``attention_mask``
                и опционально ``text``, ``metadata``.

        Returns:
            Словарь с тензорами ``input_ids``, ``attention_mask``, а также
            списками ``texts`` и ``metadata`` для записи в векторную БД.
        """
        input_ids = [f["input_ids"][: self.max_length] for f in features]
        attention_masks = [f["attention_mask"][: self.max_length] for f in features]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_masks},
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Пробрасываем текст и метаданные: они нужны при вставке в FAISSVectorDB
        batch["texts"] = [f.get(self.text_column, "") for f in features]
        batch["metadata"] = [f.get(self.metadata_column, {}) for f in features]

        return batch


class ContrastiveDataCollator:
    """Коллатор для контрастивного обучения энкодера (contrastive-режим).

    Собирает раздельные тензоры для query, positive_doc и (опционально)
    negative_doc. Поддерживает батчи со смешанным наличием hard negatives —
    примеры без негатива получают нулевые тензоры и маску ``has_negative``.

    Типичное использование: Multiple Negatives Ranking Loss (MNRL), где
    in-batch negatives дополняются hard negatives из явного поля ``neg_input_ids``.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ) -> None:
        """
        Args:
            tokenizer: Токенизатор модели-энкодера.
            max_length: Максимальная длина последовательности (hard clip).
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Args:
            features: Список словарей с ключами:
                - ``query_input_ids``, ``query_attention_mask``
                - ``pos_input_ids``, ``pos_attention_mask``
                - (опционально) ``neg_input_ids``, ``neg_attention_mask``

        Returns:
            Словарь с тензорами:
            - ``query_input_ids``, ``query_attention_mask``
            - ``pos_input_ids``, ``pos_attention_mask``
            - ``neg_input_ids``, ``neg_attention_mask`` (нули для примеров без негатива)
            - ``has_negative`` (bool-маска, shape [B])
        """
        batch = {}

        # Query и Positive всегда присутствуют
        for prefix in ("query", "pos"):
            ids = [f[f"{prefix}_input_ids"][: self.max_length] for f in features]
            masks = [f[f"{prefix}_attention_mask"][: self.max_length] for f in features]
            padded = self.tokenizer.pad(
                {"input_ids": ids, "attention_mask": masks},
                padding="longest",
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch[f"{prefix}_input_ids"] = padded["input_ids"]
            batch[f"{prefix}_attention_mask"] = padded["attention_mask"]

        # Negative — опциональный; None-элементы заменяем нулевыми тензорами
        has_neg_field = "neg_input_ids" in features[0]
        has_negative = torch.zeros(len(features), dtype=torch.bool)

        if has_neg_field:
            neg_ids_raw = [f.get("neg_input_ids") for f in features]
            neg_masks_raw = [f.get("neg_attention_mask") for f in features]

            # Определяем наличие негатива для каждого примера
            for i, v in enumerate(neg_ids_raw):
                has_negative[i] = v is not None

            # Заменяем None на пустой список — будет заполнено паддингом
            neg_ids = [
                (v[: self.max_length] if v is not None else [self.tokenizer.pad_token_id])
                for v in neg_ids_raw
            ]
            neg_masks = [
                (v[: self.max_length] if v is not None else [0])
                for v in neg_masks_raw
            ]

            padded_neg = self.tokenizer.pad(
                {"input_ids": neg_ids, "attention_mask": neg_masks},
                padding="longest",
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch["neg_input_ids"] = padded_neg["input_ids"]
            batch["neg_attention_mask"] = padded_neg["attention_mask"]
        else:
            # Нет ни одного негатива в датасете — передаём нулевые тензоры
            seq_len = batch["pos_input_ids"].shape[1]
            batch["neg_input_ids"] = torch.zeros_like(batch["pos_input_ids"])
            batch["neg_attention_mask"] = torch.zeros(
                len(features), seq_len, dtype=torch.long
            )

        batch["has_negative"] = has_negative
        return batch