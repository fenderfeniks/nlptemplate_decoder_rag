# src/pipelines/rag/core/models/pooling.py
import torch
import torch.nn.functional as F
from torch import nn


_VALID_POOLING_MODES = frozenset({"mean", "cls", "last_token"})


class Pooler(nn.Module):
    """Преобразует выходы токенов энкодера в единый эмбеддинг документа.

    Поддерживаемые режимы:
    - ``'cls'``: берёт первый токен ([CLS] / <s>). Стандарт для BERT-like моделей.
    - ``'mean'``: усредняет не-pad токены взвешенной суммой по attention_mask.
      Устойчивее CLS для задач semantic similarity, особенно для длинных текстов.
    - ``'last_token'``: берёт последний не-pad токен. Используется для
      decoder-only моделей (GPT, Llama), у которых нет CLS-токена.
    """

    def __init__(self, pooling_mode: str = "mean", normalize: bool = True) -> None:
        """
        Args:
            pooling_mode: Метод агрегации — ``'mean'``, ``'cls'`` или ``'last_token'``.
            normalize: L2-нормализовать ли эмбеддинги после пулинга.
                Обязательно ``True`` для косинусного сходства и FAISS IndexFlatIP / HNSW.
                Использует ``eps=1e-12`` чтобы избежать nan при нулевых векторах
                (возможны после mean-пулинга на полностью padding-батче).

        Raises:
            ValueError: При передаче неизвестного ``pooling_mode``.
        """
        super().__init__()

        if pooling_mode not in _VALID_POOLING_MODES:
            raise ValueError(
                f"Неизвестный режим пулинга: '{pooling_mode}'. "
                f"Допустимые: {sorted(_VALID_POOLING_MODES)}."
            )

        self.pooling_mode = pooling_mode
        self.normalize = normalize

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Агрегирует токеновые эмбеддинги в вектор документа.

        Args:
            token_embeddings: Тензор ``[batch_size, seq_len, hidden_size]`` —
                последний hidden state энкодера.
            attention_mask: Тензор ``[batch_size, seq_len]``, 1 для реальных токенов,
                0 для padding.

        Returns:
            Тензор ``[batch_size, hidden_size]`` — эмбеддинги документов.
        """
        if self.pooling_mode == "cls":
            embeddings = self._pool_cls(token_embeddings)
        elif self.pooling_mode == "mean":
            embeddings = self._pool_mean(token_embeddings, attention_mask)
        else:  # last_token
            embeddings = self._pool_last_token(token_embeddings, attention_mask)

        if self.normalize:
            # eps=1e-12 защищает от nan при нулевых векторах.
            # F.normalize внутри делает x / max(norm, eps) — без clamp на норме.
            # Это корректнее чем clamp(min=1.0) который смещает нормировку
            # коротких векторов с нормой < 1.0.
            embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-12)

        return embeddings

    # ------------------------------------------------------------------
    # Приватные методы пулинга
    # ------------------------------------------------------------------

    @staticmethod
    def _pool_cls(token_embeddings: torch.Tensor) -> torch.Tensor:
        """Берёт первый токен ([CLS] / <s>)."""
        return token_embeddings[:, 0, :]

    @staticmethod
    def _pool_mean(
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Усредняет не-pad токены по attention_mask.

        Маска приводится к dtype эмбеддингов чтобы избежать неявного каста
        при смешанной точности (bf16 / fp16). clamp(min=1.0) вместо 1e-9 —
        полностью padding-батч физически невозможен в нормальном пайплайне,
        а 1e-9 в bf16 округляется до нуля и защиты не даёт.
        """
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
        sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1.0)
        return sum_embeddings / sum_mask

    @staticmethod
    def _pool_last_token(
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Берёт последний не-pad токен каждой последовательности.

        Для decoder-only моделей (GPT, Llama) смысловое представление
        концентрируется в последнем токене, а не в первом.
        """
        # Находим индекс последнего реального токена (не padding)
        # attention_mask: [B, L], sum по dim=1 даёт длины, -1 -> индекс последнего
        last_token_indices = attention_mask.sum(dim=1) - 1  # [B]
        batch_size = token_embeddings.shape[0]
        batch_indices = torch.arange(batch_size, device=token_embeddings.device)
        return token_embeddings[batch_indices, last_token_indices, :]  # [B, H]