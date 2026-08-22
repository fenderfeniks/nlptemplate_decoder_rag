# src/pipelines/rag/inference/reranker.py
import logging
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Реранкер на базе архитектуры Cross-Encoder.
    Принимает уже инстанцированные через HFModelBuilder модель и токенизатор.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Получаем устройство напрямую из параметров загруженной модели
        self.device = next(self.model.parameters()).device

        # Принудительно переводим в eval, чтобы отключить dropout и т.д.
        self.model.eval()

    @torch.inference_mode()
    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        text_key: str = "text",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Пересортировывает список документов по запросу."""
        if not documents:
            return []

        pairs = [[query, doc["metadata"].get(text_key, "")] for doc in documents]

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        scores = self.model(**inputs).logits.squeeze(-1)
        scores_np = scores.float().cpu().numpy()

        for doc, new_score in zip(documents, scores_np, strict=True):
            doc["cross_encoder_score"] = float(new_score)

        documents.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        return documents[:top_k]
