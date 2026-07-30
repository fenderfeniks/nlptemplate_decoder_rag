# src/rag_pipeline/indexing/indexer.py
import hashlib
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.vector_db import FAISSVectorDB  # Импорт из общего utils


logger = logging.getLogger(__name__)


class KnowledgeBaseIndexer:
    """Оркестратор оффлайн-индексации с батчингом и генерацией UUID."""

    def __init__(
        self,
        model: torch.nn.Module,
        pooler: torch.nn.Module,
        vector_db: FAISSVectorDB,
        device: str = "cuda",
        precision: str = "bf16",
        push_batch_size: int = 10000,
    ):
        self.model = model.to(device).eval()
        self.pooler = pooler.to(device).eval()
        self.vector_db = vector_db
        self.device = device
        self.dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        self.push_batch_size = push_batch_size

    def _generate_doc_id(self, text: str, metadata: dict) -> str:
        """Генерирует детерминированный ID на основе текста и метаданных."""
        unique_string = f"{text}_{metadata.get('url', '')}_{metadata.get('title', '')}"
        return hashlib.md5(unique_string.encode("utf-8")).hexdigest()

    @torch.inference_mode()
    def index_dataloader(self, dataloader: DataLoader) -> None:
        logger.info("Запуск масштабируемой индексации...")

        buffer_embeddings = []
        buffer_metadata = []
        total_indexed = 0

        for _batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            texts = batch.get("text", [""] * len(input_ids))
            metadata = batch.get("metadata", [{}] * len(input_ids))

            with torch.autocast(device_type=self.device, dtype=self.dtype):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = self.pooler(outputs.last_hidden_state, attention_mask)

            emb_np = embeddings.cpu().to(torch.float32).numpy()

            for i in range(len(emb_np)):
                item_meta = dict(metadata[i]) if metadata[i] else {}
                item_meta["text"] = texts[i]
                item_meta["doc_id"] = self._generate_doc_id(texts[i], item_meta)

                buffer_embeddings.append(emb_np[i])
                buffer_metadata.append(item_meta)

            # Пушим в БД порциями, чтобы не перегружать RAM
            if len(buffer_embeddings) >= self.push_batch_size:
                self.vector_db.insert(np.array(buffer_embeddings), buffer_metadata)
                total_indexed += len(buffer_embeddings)
                buffer_embeddings, buffer_metadata = [], []
                logger.info("Проиндексировано: %d чанков...", total_indexed)

        # Очищаем остатки буфера
        if buffer_embeddings:
            self.vector_db.insert(np.array(buffer_embeddings), buffer_metadata)
            total_indexed += len(buffer_embeddings)

        logger.info("Индексация завершена. Итого в базе: %d чанков.", total_indexed)
