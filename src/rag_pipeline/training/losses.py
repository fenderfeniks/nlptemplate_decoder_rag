# src/rag_pipeline/training/losses.py
import torch
from torch import nn


class MultipleNegativesRankingLoss(nn.Module):
    """Multiple Negatives Ranking Loss (InfoNCE).

    Идеально работает как с 2 колонками (использует только in-batch negatives),
    так и с 3 колонками (добавляет hard negatives для усложнения).
    """

    def __init__(self, scale: float = 20.0):
        super().__init__()
        self.scale = scale
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        pos_embeddings: torch.Tensor,
        neg_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = torch.matmul(query_embeddings, pos_embeddings.transpose(0, 1)) * self.scale

        if neg_embeddings is not None:
            neg_scores = torch.matmul(query_embeddings, neg_embeddings.transpose(0, 1)) * self.scale
            scores = torch.cat([scores, neg_scores], dim=1)

        labels = torch.arange(scores.size(0), device=scores.device)
        return self.cross_entropy_loss(scores, labels)


class TripletLossWrapper(nn.Module):
    """Обертка над стандартным TripletMarginLoss из PyTorch.

    Требует строго 3 колонки. Если хард-негативов нет, выбрасывает понятную ошибку.
    """

    def __init__(self, margin: float = 1.0, p: float = 2.0):
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
                "ОШИБКА: TripletLoss требует передачи негативных примеров (hard negatives). "
                "В вашем датасете нет колонки negative_doc (передано 2 колонки вместо 3)."
            )

        return self.loss_fn(query_embeddings, pos_embeddings, neg_embeddings)
