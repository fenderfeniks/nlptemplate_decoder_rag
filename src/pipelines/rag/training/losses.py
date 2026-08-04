# src/pipelines/rag/training/losses.py
import torch
from torch import nn


class MultipleNegativesRankingLoss(nn.Module):
    """Multiple Negatives Ranking Loss (InfoNCE / NTXent).

    Работает как с 2 колонками (только in-batch negatives), так и с 3 колонками
    (добавляет явные hard negatives). Опционально поддерживает симметризацию лосса
    (query→doc + doc→query), что улучшает качество эмбеддингов на небольших датасетах.

    .. warning:: Утечка меток при hard negatives.
        При конкатенации ``[pos_scores | neg_scores]`` и ``labels = arange(B)``
        модель обязана предсказать i-й позитив из (B + B) кандидатов.
        Если какой-то neg_i совпадает с pos_j — это ложный негатив, но потери
        на нём будут корректно посчитаны только в scale >> 1. Это стандартное
        упрощение для MNRL; для строгой обработки ложных негативов используйте
        маски или GradCache.
    """

    def __init__(self, scale: float = 20.0, symmetric: bool = False) -> None:
        """
        Args:
            scale: Температурный коэффициент (обратная температура τ = 1/scale).
                Типичные значения: 20–50. Слишком высокий → нестабильное обучение;
                слишком низкий → слабая дискриминация.
            symmetric: Если True — считает лосс в обе стороны (q→d и d→q)
                и усредняет. Полезно при малых датасетах: удваивает эффективный
                размер батча без дополнительных вычислений.
        """
        super().__init__()
        self.scale = scale
        self.symmetric = symmetric
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        pos_embeddings: torch.Tensor,
        neg_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query_embeddings: ``[B, H]`` — нормализованные эмбеддинги запросов.
            pos_embeddings: ``[B, H]`` — нормализованные эмбеддинги позитивов.
            neg_embeddings: ``[B, H]`` — нормализованные эмбеддинги хард-негативов
                (опционально). Ожидается по одному негативу на запрос.

        Returns:
            Скалярный loss.
        """
        # Матрица сходства query × pos: [B, B]
        scores = torch.matmul(query_embeddings, pos_embeddings.T) * self.scale

        if neg_embeddings is not None:
            # Добавляем hard negatives: [B, B] → [B, 2B]
            # Каждый запрос получает B in-batch negatives + B hard negatives
            neg_scores = torch.matmul(query_embeddings, neg_embeddings.T) * self.scale
            scores = torch.cat([scores, neg_scores], dim=1)

        # Метки: i-й запрос соответствует i-му позитиву (по диагонали)
        labels = torch.arange(scores.size(0), device=scores.device)
        loss = self.cross_entropy(scores, labels)

        if self.symmetric:
            # Симметричный лосс: теперь pos_i — запрос, query_j — кандидат
            # Используем только pos×query матрицу (без neg) для обратного направления
            sym_scores = torch.matmul(pos_embeddings, query_embeddings.T) * self.scale
            sym_labels = torch.arange(sym_scores.size(0), device=sym_scores.device)
            loss = (loss + self.cross_entropy(sym_scores, sym_labels)) / 2.0

        return loss


class TripletLossWrapper(nn.Module):
    """Обёртка над ``nn.TripletMarginLoss``.

    Требует явных hard negatives. При их отсутствии — понятная ошибка
    вместо ``RuntimeError`` из глубины PyTorch.
    """

    def __init__(self, margin: float = 1.0, p: float = 2.0) -> None:
        """
        Args:
            margin: Минимальная разница между pos и neg расстоянием.
            p: Норма расстояния (2.0 = евклидово).
        """
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=p)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        pos_embeddings: torch.Tensor,
        neg_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if neg_embeddings is None:
            raise ValueError(
                "TripletLossWrapper требует hard negatives (neg_embeddings != None). "
                "Убедитесь, что датасет содержит колонку 'negative_doc' "
                "и ContrastiveDataCollator получает непустые neg_input_ids."
            )
        return self.loss_fn(query_embeddings, pos_embeddings, neg_embeddings)
