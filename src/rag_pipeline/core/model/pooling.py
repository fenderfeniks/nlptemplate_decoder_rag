# src/rag_pipeline/core/models/pooling.py
import torch
from torch import nn


class Pooler(nn.Module):
    """Преобразует выходы токенов из энкодера в единый эмбеддинг документа."""

    def __init__(self, pooling_mode: str = "mean", normalize: bool = True):
        super().__init__()
        self.pooling_mode = pooling_mode
        self.normalize = normalize

        if self.pooling_mode not in ["mean", "cls"]:
            raise ValueError(f"Неизвестный режим пулинга: {self.pooling_mode}")

    def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embeddings: Тензор [batch_size, seq_len, hidden_size]
            attention_mask: Тензор [batch_size, seq_len]
        """
        if self.pooling_mode == "cls":
            # Берем первый токен (обычно [CLS] или <s>)
            embeddings = token_embeddings[:, 0, :]

        elif self.pooling_mode == "mean":
            # Усредняем только не-pad токены
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            embeddings = sum_embeddings / sum_mask

        if self.normalize:
            # L2 нормализация эмбеддингов (обязательно для Cosine Similarity)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings
