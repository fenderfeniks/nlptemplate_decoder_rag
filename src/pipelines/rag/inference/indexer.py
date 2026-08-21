# src/pipelines/rag/indexing/indexer.py
from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.pipelines.rag.inference.embedder import RAGInferenceEmbedder
from src.vector_store.base import BaseVectorStore
from src.vector_store.lsh import LSHIndex


logger = logging.getLogger(__name__)


class KnowledgeBaseIndexer:
    """Инкрементальный индексатор документов в векторное хранилище.

    Делегирует:
    - Векторизацию -> ``RAGInferenceEmbedder``
    - Хранение -> ``BaseVectorStore`` (FAISS, Qdrant, ...)
    - Нечёткую дедупликацию -> ``LSHIndex``

    Смена бэкенда хранилища не требует изменений здесь.
    """

    def __init__(
        self,
        embedder: RAGInferenceEmbedder,
        store: BaseVectorStore,
        lsh: LSHIndex | None = None,
        push_batch_size: int = 10_000,
    ) -> None:
        """
        Args:
            embedder: Инстанс ``RAGInferenceEmbedder`` — единая точка векторизации.
            store: Векторное хранилище — любой ``BaseVectorStore``-совместимый бэкенд.
            lsh: ``LSHIndex`` для нечёткой дедупликации (опционально).
                ``None`` — только точная дедупликация по SHA-256.
            push_batch_size: Сколько эмбеддингов накапливать перед вставкой в store.
        """
        self.embedder = embedder
        self.store = store
        self.lsh = lsh
        self.push_batch_size = push_batch_size

    # ------------------------------------------------------------------
    # Дедупликация
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_doc_id(text: str, metadata: dict[str, Any]) -> str:
        """SHA-256 по тексту + метаданным. Первые 16 символов достаточно."""
        composite = f"{text}_{metadata.get('url', '')}_{metadata.get('title', '')}"
        return hashlib.sha256(composite.encode("utf-8")).hexdigest()[:16]

    def _is_exact_duplicate(
        self,
        doc_id: str,
        existing_ids: set[str],
        new_ids: set[str],
    ) -> bool:
        """Точный предикат без side-effects."""
        return doc_id in existing_ids or doc_id in new_ids

    def _is_fuzzy_duplicate(self, text: str) -> bool:
        """Нечёткий предикат через LSHIndex. Без side-effects."""
        if self.lsh is None:
            return False
        return self.lsh.is_duplicate(text)

    def _register(self, doc_id: str, text: str) -> None:
        """Регистрирует документ в LSH после принятия решения об уникальности."""
        if self.lsh is not None:
            self.lsh.register(doc_id, text)

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def index_dataloader(
        self,
        dataloader: torch.utils.data.DataLoader,
        text_column: str = "text",
    ) -> None:
        """Индексирует документы из DataLoader инкрементально.

        Args:
            dataloader: DataLoader с полями ``text_column``, ``metadata``,
                ``input_ids``, ``attention_mask``.
            text_column: Имя колонки с исходным текстом документа.
        """
        logger.info("Запуск индексации (колонка: '%s')...", text_column)

        existing_ids: set[str] = self.store.existing_doc_ids
        new_ids: set[str] = set()

        buffer_embeddings: list[np.ndarray] = []
        buffer_metadata: list[dict[str, Any]] = []
        total_indexed = 0
        total_skipped_exact = 0
        total_skipped_fuzzy = 0

        for batch in tqdm(dataloader, desc="Indexing", unit="batch"):
            batch_len = len(batch["input_ids"])
            texts: list[str] = batch.get(text_column, [""] * batch_len)
            raw_metadata: list[dict[str, Any]] = batch.get("metadata") or [
                {} for _ in range(batch_len)
            ]

            valid_indices: list[int] = []
            valid_doc_ids: list[str] = []

            for i, text in enumerate(texts):
                item_meta: dict[str, Any] = dict(raw_metadata[i]) if raw_metadata[i] else {}
                doc_id = self._generate_doc_id(text, item_meta)

                if self._is_exact_duplicate(doc_id, existing_ids, new_ids):
                    total_skipped_exact += 1
                    continue

                if self._is_fuzzy_duplicate(text):
                    total_skipped_fuzzy += 1
                    continue

                self._register(doc_id, text)
                new_ids.add(doc_id)
                valid_indices.append(i)
                valid_doc_ids.append(doc_id)

            if not valid_indices:
                continue

            valid_texts = [texts[i] for i in valid_indices]
            emb_np = self.embedder.encode(valid_texts)

            for idx, original_i in enumerate(valid_indices):
                meta: dict[str, Any] = (
                    dict(raw_metadata[original_i]) if raw_metadata[original_i] else {}
                )
                meta["text"] = texts[original_i]
                meta["doc_id"] = valid_doc_ids[idx]
                buffer_embeddings.append(emb_np[idx])
                buffer_metadata.append(meta)

            if len(buffer_embeddings) >= self.push_batch_size:
                self.store.insert(np.stack(buffer_embeddings), buffer_metadata)
                total_indexed += len(buffer_embeddings)
                buffer_embeddings, buffer_metadata = [], []
                logger.info(
                    "Чанк проиндексирован. Всего: %d. Пропущено: точных=%d, нечётких=%d.",
                    total_indexed,
                    total_skipped_exact,
                    total_skipped_fuzzy,
                )

        if buffer_embeddings:
            self.store.insert(np.stack(buffer_embeddings), buffer_metadata)
            total_indexed += len(buffer_embeddings)

        logger.info(
            "Индексация завершена. Добавлено: %d. Пропущено: точных=%d, нечётких=%d. ntotal=%d.",
            total_indexed,
            total_skipped_exact,
            total_skipped_fuzzy,
            self.store.ntotal,
        )
