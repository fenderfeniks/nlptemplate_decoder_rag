# src/rag_pipeline/indexing/indexer.py
import hashlib
import logging

import numpy as np
import torch
from tqdm import tqdm

from src.utils.vector_db import FAISSVectorDB


logger = logging.getLogger(__name__)

_VALID_PRECISIONS = frozenset({"bf16", "fp16", "fp32"})


class KnowledgeBaseIndexer:
    """Оркестратор оффлайн-индексации корпуса документов в FAISSVectorDB.

    Прогоняет датасет через энкодер батчами, буферизирует эмбеддинги
    и пушит их в векторную БД порциями ``push_batch_size`` для контроля RAM.
    Поддерживает mixed precision через ``torch.autocast``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        pooler: torch.nn.Module,
        vector_db: FAISSVectorDB,
        device: str = "cuda",
        precision: str = "bf16",
        push_batch_size: int = 10_000,
    ) -> None:
        """
        Args:
            model: Энкодер (HF PreTrainedModel или любой nn.Module с last_hidden_state).
            pooler: Пулер для агрегации токеновых эмбеддингов.
            vector_db: Инстанс FAISSVectorDB для записи результатов.
            device: Устройство вычислений — ``'cuda'``, ``'cpu'``, ``'mps'``.
            precision: Точность вычислений — ``'bf16'``, ``'fp16'``, ``'fp32'``.
                ``'bf16'`` и ``'fp16'`` используют ``torch.autocast``; ``'fp32'`` — без каста.
                На CPU допустимо только ``'fp32'`` (``'bf16'`` на CPU через autocast
                экспериментально, ``'fp16'`` не поддерживается).
            push_batch_size: Сколько эмбеддингов накапливать в буфере перед
                вставкой в FAISSVectorDB. Контролирует пиковое потребление RAM.

        Raises:
            ValueError: При недопустимом значении ``precision``.
        """
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

        # Маппинг строки в torch.dtype для autocast
        self._dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        self.dtype = self._dtype_map[precision]

        # autocast device_type: 'cuda' для GPU, 'cpu' для CPU/MPS
        # MPS не поддерживает autocast — используем cpu-path
        self._autocast_device = "cuda" if device.startswith("cuda") else "cpu"

    def _generate_doc_id(self, text: str, metadata: dict) -> str:
        """Генерирует детерминированный MD5-идентификатор документа.

        Составной ключ: текст + URL + title. При отсутствии полей — пустая строка.
        Детерминирован: одинаковый текст + метаданные всегда дают одинаковый id,
        что позволяет инкрементально переиндексировать только новые документы.
        """
        composite = f"{text}_{metadata.get('url', '')}_{metadata.get('title', '')}"
        return hashlib.md5(composite.encode("utf-8")).hexdigest()

    @torch.inference_mode()
    def index_dataloader(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Индексирует все документы из DataLoader.

        Args:
            dataloader: DataLoader с батчами от ``IndexingDataCollator``.
                Ожидает ключи ``'input_ids'``, ``'attention_mask'``,
                и опционально ``'texts'``, ``'metadata'``.
        """
        logger.info("Запуск индексации...")

        buffer_embeddings: list[np.ndarray] = []
        buffer_metadata: list[dict] = []
        total_indexed = 0

        use_autocast = self.precision != "fp32"

        for batch in tqdm(dataloader, desc="Indexing", unit="batch"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # IndexingDataCollator передаёт 'texts' и 'metadata' как списки
            texts: list[str] = batch.get("texts", [""] * len(input_ids))
            metadata: list[dict] = batch.get("metadata", [{}] * len(input_ids))

            if use_autocast:
                ctx = torch.autocast(device_type=self._autocast_device, dtype=self.dtype)
            else:
                ctx = torch.no_grad()  # уже в inference_mode, но явно безопаснее

            with ctx:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = self.pooler(outputs.last_hidden_state, attention_mask)

            # Приводим к float32 перед numpy: FAISS не принимает bf16/fp16
            emb_np: np.ndarray = embeddings.cpu().to(torch.float32).numpy()

            for i in range(len(emb_np)):
                item_meta = dict(metadata[i]) if metadata[i] else {}
                item_meta["text"] = texts[i]
                item_meta["doc_id"] = self._generate_doc_id(texts[i], item_meta)

                buffer_embeddings.append(emb_np[i])
                buffer_metadata.append(item_meta)

            # Пушим в БД порциями — контролируем пиковую RAM
            if len(buffer_embeddings) >= self.push_batch_size:
                self.vector_db.insert(np.stack(buffer_embeddings), buffer_metadata)
                total_indexed += len(buffer_embeddings)
                buffer_embeddings, buffer_metadata = [], []
                logger.info("Проиндексировано: %d чанков...", total_indexed)

        # Остатки буфера
        if buffer_embeddings:
            self.vector_db.insert(np.stack(buffer_embeddings), buffer_metadata)
            total_indexed += len(buffer_embeddings)

        logger.info(
            "Индексация завершена. Всего в базе: %d чанков (ntotal=%d).",
            total_indexed,
            self.vector_db.index.ntotal,
        )
