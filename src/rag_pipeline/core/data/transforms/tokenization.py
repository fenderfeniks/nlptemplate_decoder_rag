# src/rag_pipeline/core/data/transforms/tokenization.py
import logging
from typing import Any, Optional

from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizerBase

from src.rag_pipeline.core.data.transforms.base import BaseDatasetTransform

logger = logging.getLogger(__name__)


class RAGTokenizationTransform(BaseDatasetTransform):
    """Токенизация текстов для RAG-пайплайна.

    Режимы:
    - ``'indexing'``: Токенизирует колонку ``text``, добавляет ``input_ids``
      и ``attention_mask``. Оригинальные колонки **сохраняются** — они нужны
      для последующей записи в векторную БД.
    - ``'contrastive'``: Токенизирует ``query``, ``positive_doc`` и опционально
      ``negative_doc`` в раздельные тензоры (``query_*``, ``pos_*``, ``neg_*``).
      Негативный документ токенизируется только если хотя бы один элемент батча
      не является ``None``.
    """

    _VALID_MODES = ("indexing", "contrastive")

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mode: str = "indexing",
        text_column: str = "text",
        query_column: str = "query",
        positive_column: str = "positive_doc",
        negative_column: str = "negative_doc",
        max_length: int = 512,
        num_proc: int = 4,
        batch_size: int = 1000,
    ) -> None:
        """
        Args:
            tokenizer: Токенизатор модели (PreTrainedTokenizerBase).
            mode: Режим работы — 'indexing' или 'contrastive'.
            text_column: Колонка с текстом (используется в режиме indexing).
            query_column: Колонка с запросом (contrastive).
            positive_column: Колонка с позитивным документом (contrastive).
            negative_column: Колонка с негативным документом (contrastive, опционально).
            max_length: Максимальная длина последовательности в токенах.
            num_proc: Число процессов для параллельного map.
            batch_size: Размер батча для map.
        """
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"Неизвестный режим токенизации: '{mode}'. "
                f"Допустимые значения: {self._VALID_MODES}"
            )
        self.tokenizer = tokenizer
        self.mode = mode
        self.text_column = text_column
        self.query_column = query_column
        self.positive_column = positive_column
        self.negative_column = negative_column
        self.max_length = max_length
        self.num_proc = num_proc
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Внутренние функции токенизации
    # ------------------------------------------------------------------

    def _tokenize_indexing(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        return self.tokenizer(
            examples[self.text_column],
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )

    def _tokenize_contrastive(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        res: dict[str, list[Any]] = {}

        # Query
        q_enc = self.tokenizer(
            examples[self.query_column],
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        res["query_input_ids"] = q_enc["input_ids"]
        res["query_attention_mask"] = q_enc["attention_mask"]

        # Positive
        p_enc = self.tokenizer(
            examples[self.positive_column],
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        res["pos_input_ids"] = p_enc["input_ids"]
        res["pos_attention_mask"] = p_enc["attention_mask"]

        # Negative — токенизируем только если колонка есть И хотя бы один элемент не None.
        # Проверяем весь батч, а не только [0], чтобы не потерять частичные негативы.
        neg_values: Optional[list] = examples.get(self.negative_column)
        has_any_negative = neg_values is not None and any(v is not None for v in neg_values)

        if has_any_negative:
            # Подменяем None на пустую строку перед токенизацией, затем обнуляем маски.
            # Токенизатор не принимает None в списке.
            filled = [v if v is not None else "" for v in neg_values]
            n_enc = self.tokenizer(
                filled,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )

            neg_input_ids = []
            neg_attention_mask = []
            for v, ids, mask in zip(neg_values, n_enc["input_ids"], n_enc["attention_mask"]):
                if v is None:
                    # Запись без негатива — передаём None, коллатор должен их обработать
                    neg_input_ids.append(None)
                    neg_attention_mask.append(None)
                else:
                    neg_input_ids.append(ids)
                    neg_attention_mask.append(mask)

            res["neg_input_ids"] = neg_input_ids
            res["neg_attention_mask"] = neg_attention_mask

        return res

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    def __call__(self, dataset: HFDataset) -> HFDataset:
        logger.info(
            "Токенизация (режим: %s, max_length=%d)...", self.mode, self.max_length
        )

        func = (
            self._tokenize_indexing
            if self.mode == "indexing"
            else self._tokenize_contrastive
        )

        return dataset.map(
            func,
            batched=True,
            batch_size=self.batch_size,
            num_proc=self.num_proc,
            desc=f"Tokenizing ({self.mode})",
            # Оригинальные колонки НЕ удаляем:
            # - indexing: текст и метаданные нужны для записи в векторную БД
            # - contrastive: query/positive_doc могут использоваться в логировании
        )