# src/pipelines/decoder/core/data/transforms/tokenization.py
import logging
from typing import Any

from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizerBase

from src.pipelines.base.core.data.transforms.base import BaseDatasetTransform

logger = logging.getLogger(__name__)


class TokenizationTransform(BaseDatasetTransform):
    """Токенизация текстов для decoder-пайплайна (SFT и Chat).

    Режимы:
    - ``'sft'``: Supervised Fine-Tuning. Склеивает ``prompt`` и ``target``
      через ``separator``, токенизирует полный текст и отдельно промпт
      для вычисления ``prompt_len``. Loss маскируется по ``prompt_len``
      в коллаторе — separator включается в промпт и не обучается.
    - ``'chat'``: Применяет ``chat_template`` токенизатора к колонке
      ``messages`` (список словарей ``[{'role': ..., 'content': ...}]``).

    .. note:: Все режимы применяют ``truncation=True`` с заданным ``max_length``.
        Последовательности длиннее ``max_length`` будут обрезаны без предупреждения.
    """

    _VALID_TASKS = ("sft", "chat", "sft_with_template")

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        task: str = "sft",
        prompt_column: str = "prompt",
        target_column: str = "target",
        messages_column: str = "messages",
        max_length: int = 2048,
        separator: str = " ",
        num_proc: int = 4,
        batch_size: int = 1000,
        writer_batch_size: int = 200,
    ) -> None:
        if task not in self._VALID_TASKS:
            raise ValueError(
                f"Неизвестный режим токенизации: '{task}'. "
                f"Допустимые значения: {self._VALID_TASKS}"
            )
        if max_length <= 0:
            raise ValueError(
                f"max_length должен быть положительным числом, получено: {max_length}"
            )
        self.tokenizer = tokenizer
        self.task = task
        self.prompt_column = prompt_column
        self.target_column = target_column
        self.messages_column = messages_column
        self.max_length = max_length
        self.separator = separator
        self.num_proc = num_proc
        self.batch_size = batch_size
        self.writer_batch_size = writer_batch_size

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

    def _tokenize_sft_with_template(
        self, examples: dict[str, list[Any]]
    ) -> dict[str, list[Any]]:
        results: dict[str, list] = {
            "input_ids": [],
            "attention_mask": [],
            "prompt_len": [],
        }

        for prompt, target in zip(
            examples[self.prompt_column], examples[self.target_column]
        ):
            messages_full = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ]
            messages_prompt_only = [
                {"role": "user", "content": prompt},
            ]

            full_ids = self.tokenizer.apply_chat_template(
                messages_full,
                tokenize=True,
                truncation=True,
                max_length=self.max_length,
                add_generation_prompt=False,
            )
            prompt_ids = self.tokenizer.apply_chat_template(
                messages_prompt_only,
                tokenize=True,
                truncation=True,
                max_length=self.max_length,
                add_generation_prompt=True,
            )

            results["input_ids"].append(full_ids)
            results["attention_mask"].append([1] * len(full_ids))
            results["prompt_len"].append(len(prompt_ids))

        return results

    def __call__(self, dataset: HFDataset) -> HFDataset:
        task_column_map: dict[str, str] = {
            "sft": self.prompt_column,
            "chat": self.messages_column,
            "sft_with_template": self.prompt_column,  # добавить
        }
        required_column = task_column_map[self.task]
        
        if required_column not in dataset.column_names:
            logger.warning(
                "Колонка '%s' не найдена в датасете — токенизация пропущена.",
                required_column,
            )
            return dataset

        if self.task == "sft" and self.target_column not in dataset.column_names:
            logger.warning(
                "Для режима 'sft' необходимы обе колонки: '%s' и '%s'.",
                self.prompt_column,
                self.target_column,
            )
            return dataset

        logger.info(
            "Токенизация (режим: %s, max_length=%d)...", self.task, self.max_length
        )

        func_map = {
            "sft": self._tokenize_sft,
            "chat": self._tokenize_chat,
            "sft_with_template": self._tokenize_sft_with_template,  # добавить
        }

        result = dataset.map(
            func_map[self.task],
            batched=True,
            batch_size=self.batch_size,
            writer_batch_size=self.writer_batch_size,
            num_proc=self.num_proc,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing ({self.task})",
        )

        logger.info(
            "Токенизация завершена: %d записей, режим '%s'.",
            len(result),
            self.task,
        )
        return result