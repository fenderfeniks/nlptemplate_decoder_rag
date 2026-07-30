# src/rag_pipeline/inference/embedder.py
import logging

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase


logger = logging.getLogger(__name__)

_VALID_PRECISIONS = frozenset({"bf16", "fp16", "fp32"})


class RAGInferenceEmbedder:
    """Векторизатор текста для RAG-инференса.

    Поддерживает mixed precision (bf16/fp16/fp32), динамический паддинг
    внутри батча и опциональный прогресс-бар через tqdm.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        pooler: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        device: str = "cuda",
        precision: str = "bf16",
        max_length: int = 512,
    ) -> None:
        """
        Args:
            model: Энкодер (HF PreTrainedModel или nn.Module с ``last_hidden_state``).
            pooler: Пулер из ``src.rag_pipeline.core.models.pooling.Pooler``.
            tokenizer: Токенизатор для подготовки входных данных.
            device: Устройство — ``'cuda'``, ``'cpu'``, ``'mps'``, ``'cuda:1'`` и т.д.
            precision: Точность вычислений — ``'bf16'``, ``'fp16'``, ``'fp32'``.
                На CPU автоматически используется ``'fp32'``, даже если указан ``'bf16'``.
            max_length: Максимальная длина последовательности в токенах.

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
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.precision = precision

        self.dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[precision]

        # autocast работает с device_type 'cuda' или 'cpu'.
        # MPS пока не поддерживает autocast — используем cpu-path (без каста).
        self._autocast_device = "cuda" if device.startswith("cuda") else "cpu"
        self._use_autocast = precision != "fp32"

    @torch.inference_mode()
    def encode(
        self,
        texts: list[str] | str,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Векторизует тексты с динамическим паддингом внутри батча.

        Args:
            texts: Строка или список строк для векторизации.
            batch_size: Размер батча инференса.
            show_progress: Показывать tqdm-прогресс-бар.

        Returns:
            ``np.ndarray`` формы ``(N, hidden_size)`` в dtype ``float32``.
        """
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings: list[np.ndarray] = []

        batches = range(0, len(texts), batch_size)
        if show_progress:
            batches = tqdm(batches, desc="Encoding", unit="batch")

        for i in batches:
            batch_texts = texts[i : i + batch_size]

            # Динамический паддинг: padding="longest" не тратит VRAM на лишние токены
            encoded = self.tokenizer(
                batch_texts,
                padding="longest",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            if self._use_autocast:
                ctx = torch.autocast(device_type=self._autocast_device, dtype=self.dtype)
            else:
                ctx = torch.no_grad()  # в inference_mode уже, но явный ctx для единообразия

            with ctx:
                outputs = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )
                embeddings = self.pooler(outputs.last_hidden_state, encoded["attention_mask"])

            # Приводим к float32 перед numpy: FAISS и numpy не работают с bf16
            all_embeddings.append(embeddings.cpu().to(torch.float32).numpy())

        return np.concatenate(all_embeddings, axis=0)
