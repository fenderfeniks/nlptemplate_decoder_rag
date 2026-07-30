# src/rag_pipeline/inference/embedder.py
import logging

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase


logger = logging.getLogger(__name__)


class RAGInferenceEmbedder:
    """Индустриальный класс для векторизации текста.

    Поддерживает mixed precision (bf16/fp16), динамический паддинг
    и оптимизированный inference_mode.
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
        self.model = model
        self.pooler = pooler
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

        self.dtype = (
            torch.bfloat16
            if precision == "bf16"
            else torch.float16
            if precision == "fp16"
            else torch.float32
        )

        self.model.to(self.device).eval()
        self.pooler.to(self.device).eval()

    @torch.inference_mode()
    def encode(
        self, texts: list[str] | str, batch_size: int = 32, show_progress: bool = False
    ) -> np.ndarray:
        """Векторизует текст с динамическим паддингом внутри батча."""
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Динамический паддинг: padding="longest" экономит VRAM
            encoded = self.tokenizer(
                batch_texts,
                padding="longest",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.autocast(device_type=self.device.replace("cuda", "cuda"), dtype=self.dtype):
                outputs = self.model(
                    input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
                )
                embeddings = self.pooler(outputs.last_hidden_state, encoded["attention_mask"])

            all_embeddings.append(embeddings.cpu().to(torch.float32).numpy())

        return np.concatenate(all_embeddings, axis=0)
