# src/pipelines/rag/inference/embedder.py
import logging
from contextlib import nullcontext

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

    Является единственным местом где живёт логика autocast/precision/dtype —
    ``KnowledgeBaseIndexer`` делегирует векторизацию сюда вместо дублирования.

    .. note:: Контракт нормализации.
        Этот класс **не нормализует** эмбеддинги — нормализация выполняется
        в ``Pooler`` (``normalize=True``, L2 через ``F.normalize``). Повторная
        нормализация здесь была бы дублированием и сломала бы случай
        ``Pooler(normalize=False)``. Убедитесь что ``Pooler`` инициализирован
        с ``normalize=True`` если используете FAISS ``IndexFlatIP`` для
        косинусного поиска.
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
            pooler: Пулер из ``src.pipelines.rag.core.models.pooling.Pooler``.
                Должен быть инициализирован с ``normalize=True`` для корректного
                косинусного поиска через FAISS ``IndexFlatIP``.
            tokenizer: Токенизатор для подготовки входных данных.
            device: Устройство — ``'cuda'``, ``'cpu'``, ``'mps'``, ``'cuda:1'`` и т.д.
            precision: Точность вычислений — ``'bf16'``, ``'fp16'``, ``'fp32'``.
                На MPS и CPU автоматически используется ``'fp32'``:
                - CPU autocast с bf16 крашится на PyTorch < 2.0.
                - MPS не поддерживает autocast (PyTorch 2.x).
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

        # MPS не поддерживает autocast (PyTorch 2.x), CPU autocast с bf16
        # крашится на PyTorch < 2.0 — форсируем fp32 для обоих случаев.
        is_cuda = device.startswith("cuda")
        if not is_cuda:
            if precision != "fp32":
                logger.warning(
                    "device='%s': precision='%s' не поддерживается — "
                    "автоматически используется fp32.",
                    device,
                    precision,
                )
            self.dtype = torch.float32
            self._use_autocast = False
        else:
            self.dtype = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
            }[precision]
            self._use_autocast = precision != "fp32"

        # autocast требует device_type 'cuda' или 'cpu' — не 'cuda:1', не 'mps'
        self._autocast_device = "cuda" if is_cuda else "cpu"

    def _autocast_ctx(self):
        """Возвращает контекст autocast или nullcontext для fp32/MPS/CPU."""
        if self._use_autocast:
            return torch.autocast(device_type=self._autocast_device, dtype=self.dtype)
        # nullcontext семантически точнее torch.no_grad() внутри inference_mode:
        # явно говорит "здесь нет дополнительного контекста"
        return nullcontext()

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
            FAISS и numpy не работают с bf16 — приведение выполняется автоматически.
            Нормализация не применяется здесь — делегирована ``Pooler``.
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

            with self._autocast_ctx():
                outputs = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )
                embeddings = self.pooler(outputs.last_hidden_state, encoded["attention_mask"])

            all_embeddings.append(embeddings.cpu().to(torch.float32).numpy())

        return np.concatenate(all_embeddings, axis=0)
