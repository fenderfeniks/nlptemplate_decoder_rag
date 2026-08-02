# src/decoder_pipeline/core/data/transforms/tokenization.py
import logging
from typing import Any, Optional

from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizerBase

from src.decoder_pipeline.core.data.transforms.base import BaseDatasetTransform

logger = logging.getLogger(__name__)


class TokenizationTransform(BaseDatasetTransform):
    """Токенизация текстов для decoder-пайплайна.

    Режимы:
    - ``'cpt'``: Continual Pre-Training. Токенизирует колонку ``text`` целиком.
      Оригинальные колонки удаляются — на выходе только ``input_ids``
      и ``attention_mask``.
    - ``'sft'``: Supervised Fine-Tuning. Склеивает ``prompt`` и ``target``
      через ``separator``, токенизирует полный текст и отдельно промпт
      для вычисления ``prompt_len``. Loss маскируется по ``prompt_len``
      в коллаторе — separator включается в промпт и не обучается.
    - ``'chat'``: Применяет ``chat_template`` токенизатора к колонке
      ``messages`` (список словарей ``[{'role': ..., 'content': ...}]``).
      Оригинальные колонки удаляются.

    .. note:: Все режимы применяют ``truncation=True`` с заданным ``max_length``.
        Последовательности длиннее ``max_length`` будут обрезаны без предупреждения.
        Используйте ``LengthFilterTransform`` до токенизации чтобы контролировать
        состав датасета, или после — чтобы отфильтровать обрезанные записи.
    """

    _VALID_MODES = ("cpt", "sft", "chat")

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mode: str = "cpt",
        text_column: str = "text",
        prompt_column: str = "prompt",
        target_column: str = "target",
        messages_column: str = "messages",
        max_length: int = 2048,
        separator: str = " ",
        num_proc: int = 4,
        batch_size: int = 1000,
        writer_batch_size: int = 200,
    ) -> None:
        """
        Args:
            tokenizer: Токенизатор модели (PreTrainedTokenizerBase).
            mode: Режим работы — ``'cpt'``, ``'sft'`` или ``'chat'``.
            text_column: Колонка с текстом (режим ``cpt``).
            prompt_column: Колонка с промптом (режим ``sft``).
            target_column: Колонка с целевым ответом (режим ``sft``).
            messages_column: Колонка со списком сообщений (режим ``chat``).
            max_length: Максимальная длина последовательности в токенах.
                Должен быть положительным числом.
            separator: Разделитель между промптом и таргетом (режим ``sft``).
                Включается в промпт при вычислении ``prompt_len``.
            num_proc: Число процессов для параллельного map.
            batch_size: Размер батча для map.
            writer_batch_size: Размер батча при записи на диск. Уменьшите при
                нехватке RAM.

        Raises:
            ValueError: Если ``mode`` не входит в допустимые значения.
            ValueError: Если ``max_length`` не является положительным числом.
        """
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"Неизвестный режим токенизации: '{mode}'. "
                f"Допустимые значения: {self._VALID_MODES}"
            )
        if max_length <= 0:
            raise ValueError(
                f"max_length должен быть положительным числом, получено: {max_length}"
            )
        self.tokenizer = tokenizer
        self.mode = mode
        self.text_column = text_column
        self.prompt_column = prompt_column
        self.target_column = target_column
        self.messages_column = messages_column
        self.max_length = max_length
        self.separator = separator
        self.num_proc = num_proc
        self.batch_size = batch_size
        self.writer_batch_size = writer_batch_size

    # ------------------------------------------------------------------
    # Внутренние функции токенизации
    # ------------------------------------------------------------------

    def _tokenize_cpt(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        return self.tokenizer(
            examples[self.text_column],
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )

    def _tokenize_sft(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        prompts = examples[self.prompt_column]
        targets = examples[self.target_column]

        full_texts = [p + self.separator + t for p, t in zip(prompts, targets)]
        encodings = self.tokenizer(
            full_texts,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )

        # Включаем separator в prompt_len — loss считается только по target,
        # separator является частью промпта и не должен обучаться.
        prompts_with_sep = [p + self.separator for p in prompts]
        prompt_encodings = self.tokenizer(
            prompts_with_sep,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "prompt_len": [len(p) for p in prompt_encodings["input_ids"]],
        }

    def _tokenize_chat(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        return self.tokenizer.apply_chat_template(
            examples[self.messages_column],
            tokenize=True,
            truncation=True,
            max_length=self.max_length,
            return_dict=True,
        )

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    def __call__(self, dataset: HFDataset) -> HFDataset:
        mode_column_map: dict[str, str] = {
            "cpt": self.text_column,
            "sft": self.prompt_column,
            "chat": self.messages_column,
        }
        required_column = mode_column_map[self.mode]
        if required_column not in dataset.column_names:
            logger.warning(
                "Колонка '%s' не найдена в датасете — токенизация пропущена. "
                "Убедитесь, что режим '%s' соответствует составу датасета.",
                required_column,
                self.mode,
            )
            return dataset

        # SFT дополнительно требует target_column
        if self.mode == "sft" and self.target_column not in dataset.column_names:
            logger.warning(
                "Колонка '%s' не найдена в датасете — токенизация пропущена. "
                "Для режима 'sft' необходимы обе колонки: '%s' и '%s'.",
                self.target_column,
                self.prompt_column,
                self.target_column,
            )
            return dataset

        logger.info(
            "Токенизация (режим: %s, max_length=%d)...", self.mode, self.max_length
        )

        func_map = {
            "cpt": self._tokenize_cpt,
            "sft": self._tokenize_sft,
            "chat": self._tokenize_chat,
        }

        # Оригинальные текстовые колонки удаляем — в отличие от RAG,
        # decoder после токенизации работает только с тензорами.
        result = dataset.map(
            func_map[self.mode],
            batched=True,
            batch_size=self.batch_size,
            writer_batch_size=self.writer_batch_size,
            num_proc=self.num_proc,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing ({self.mode})",
        )

        logger.info(
            "Токенизация завершена: %d записей, режим '%s'.",
            len(result),
            self.mode,
        )
        return result