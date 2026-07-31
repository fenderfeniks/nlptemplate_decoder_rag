# src/rag_pipeline/indexing/indexer.py
from __future__ import annotations

import hashlib
import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm


if TYPE_CHECKING:
    from datasketch import MinHash as MinHashType

try:
    from datasketch import MinHash
except ImportError:
    MinHash = None  # type: ignore[assignment, misc]

from src.utils.vector_db import FAISSVectorDB


logger = logging.getLogger(__name__)
_VALID_PRECISIONS = frozenset({"bf16", "fp16", "fp32"})


class KnowledgeBaseIndexer:
    def __init__(
        self,
        model: torch.nn.Module,
        pooler: torch.nn.Module,
        vector_db: FAISSVectorDB,
        device: str = "cuda",
        precision: str = "bf16",
        push_batch_size: int = 10_000,
        ngram_size: int = 5,
    ) -> None:
        if precision not in _VALID_PRECISIONS:
            raise ValueError(
                f"Недопустимое значение precision: '{precision}'. "
                f"Допустимые: {sorted(_VALID_PRECISIONS)}."
            )

        self.model = model.to(device).eval()
        self.pooler = pooler.to(device).eval()
        self.vector_db = vector_db
        self.device = device
        self.precision = precision
        self.push_batch_size = push_batch_size

        self.ngram_size = ngram_size
        self.word_pattern = re.compile(r"(?u)\b\w+\b")

        self._dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        self.dtype = self._dtype_map[precision]
        self._autocast_device = "cuda" if device.startswith("cuda") else "cpu"

    def _generate_doc_id(self, text: str, metadata: dict) -> str:
        composite = f"{text}_{metadata.get('url', '')}_{metadata.get('title', '')}"
        return hashlib.md5(composite.encode("utf-8")).hexdigest()

    def _compute_minhash(self, text: str) -> MinHashType | None:
        """Вычисляет MinHash для текста. Возвращает None если datasketch не установлен."""
        if MinHash is None:
            return None

        tokens = self.word_pattern.findall(text.lower())
        m = MinHash(num_perm=self.vector_db.lsh_num_perm, scheme="legacy")

        if len(tokens) < self.ngram_size:
            m.update(" ".join(tokens).encode("utf-8"))
        else:
            for i in range(len(tokens) - self.ngram_size + 1):
                shingle = " ".join(tokens[i : i + self.ngram_size]).encode("utf-8")
                m.update(shingle)
        return m

    def _to_tensor_indices(self, indices: list[int], batch_size: int) -> torch.Tensor:
        """Конвертирует список индексов в LongTensor для безопасного индексирования батча."""
        return torch.tensor(indices, dtype=torch.long)

    @torch.inference_mode()
    def index_dataloader(
        self, dataloader: torch.utils.data.DataLoader, text_column: str = "text"
    ) -> None:
        logger.info("Запуск инкрементальной индексации (колонка текста: '%s')...", text_column)

        existing_ids = self.vector_db.existing_doc_ids
        new_ids_in_this_run: set[str] = set()

        buffer_embeddings: list[np.ndarray] = []
        buffer_metadata: list[dict] = []
        total_indexed = 0
        total_skipped_exact = 0
        total_skipped_fuzzy = 0

        use_autocast = self.precision != "fp32"

        for batch in tqdm(dataloader, desc="Indexing", unit="batch"):
            texts: list[str] = batch.get(text_column, [""] * len(batch["input_ids"]))
            metadata: list[dict] = batch.get("metadata", [{}] * len(batch["input_ids"]))

            valid_indices: list[int] = []
            valid_doc_ids: list[str] = []

            for i, text in enumerate(texts):
                item_meta = dict(metadata[i]) if metadata[i] else {}
                doc_id = self._generate_doc_id(text, item_meta)

                # 1. Проверка на точный дубликат (MD5)
                if doc_id in existing_ids or doc_id in new_ids_in_this_run:
                    total_skipped_exact += 1
                    continue

                # 2. Проверка на нечеткий дубликат (MinHash LSH)
                if self.vector_db.lsh is not None:
                    m = self._compute_minhash(text)
                    if m is not None:
                        similar = self.vector_db.lsh.query(m)
                        if similar:
                            total_skipped_fuzzy += 1
                            continue
                        # Добавляем в глобальный LSH для отлова дублей внутри батча
                        self.vector_db.lsh.insert(doc_id, m)

                valid_indices.append(i)
                valid_doc_ids.append(doc_id)
                new_ids_in_this_run.add(doc_id)

            if not valid_indices:
                continue

            # 3. Инференс только для уникальных документов
            # Используем torch.tensor для безопасного индексирования независимо от типа батча
            idx_tensor = self._to_tensor_indices(valid_indices, len(texts))

            input_ids_batch = batch["input_ids"]
            attention_mask_batch = batch["attention_mask"]

            # Поддержка как Tensor, так и list-батчей из DataLoader
            if not isinstance(input_ids_batch, torch.Tensor):
                input_ids_batch = torch.stack(input_ids_batch)
            if not isinstance(attention_mask_batch, torch.Tensor):
                attention_mask_batch = torch.stack(attention_mask_batch)

            input_ids = input_ids_batch[idx_tensor].to(self.device)
            attention_mask = attention_mask_batch[idx_tensor].to(self.device)

            ctx = (
                torch.autocast(device_type=self._autocast_device, dtype=self.dtype)
                if use_autocast
                else torch.no_grad()
            )
            with ctx:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = self.pooler(outputs.last_hidden_state, attention_mask)

            emb_np: np.ndarray = embeddings.cpu().to(torch.float32).numpy()

            for idx, original_i in enumerate(valid_indices):
                item_meta = dict(metadata[original_i]) if metadata[original_i] else {}
                item_meta["text"] = texts[original_i]
                item_meta["doc_id"] = valid_doc_ids[idx]

                buffer_embeddings.append(emb_np[idx])
                buffer_metadata.append(item_meta)

            if len(buffer_embeddings) >= self.push_batch_size:
                self.vector_db.insert(np.stack(buffer_embeddings), buffer_metadata)
                total_indexed += len(buffer_embeddings)
                buffer_embeddings, buffer_metadata = [], []
                logger.info(
                    "Чанк проиндексирован. Отброшено: точных=%d, нечетких=%d.",
                    total_skipped_exact,
                    total_skipped_fuzzy,
                )

        if buffer_embeddings:
            self.vector_db.insert(np.stack(buffer_embeddings), buffer_metadata)
            total_indexed += len(buffer_embeddings)

        logger.info(
            "Индексация завершена. Добавлено: %d. Отброшено: точных=%d, нечетких=%d. ntotal=%d.",
            total_indexed,
            total_skipped_exact,
            total_skipped_fuzzy,
            self.vector_db.index.ntotal,
        )
