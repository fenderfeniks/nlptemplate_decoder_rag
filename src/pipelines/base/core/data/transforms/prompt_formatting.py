# src/pipelines/base/core/data/transforms/prompt_formatting.py
import logging
from typing import Any

from datasets import Dataset as HFDataset

from src.pipelines.base.core.data.transforms.base import BaseDatasetTransform
from src.pipelines.decoder.core.prompts.manager import PromptManager

logger = logging.getLogger(__name__)


class PromptFormattingTransform(BaseDatasetTransform):
    """Форматирует примеры датасета в строки промптов через Jinja2-шаблон.

    Параметры берутся из конфига data через Hydra-интерполяцию:

        output_column:   "${data.prompt_column}"    — куда писать готовый промпт
        retrieve_column: "${data.retrieve_column}"  — колонка RAG-контекста (null если нет)

    Логика прокидывания контекста:
        Если retrieve_column задан и присутствует в датасете, его значение
        передаётся в шаблон под ключом ``context``. Пустая строка заменяется
        на None — шаблон может использовать ``{% if context %}`` для условного
        рендеринга блока с контекстом.

    Шаблон получает все колонки строки как kwargs плюс ``context`` (если задан).
    Имя переменной в шаблоне всегда ``context`` — не ``chunk_text`` и не
    название колонки из конфига. Это единственный контракт между этим
    трансформом и Jinja2-шаблонами.
    """

    def __init__(
        self,
        prompt_manager: PromptManager,
        template_name: str,
        output_column: str = "prompt",
        retrieve_column: str | None = None,
        num_proc: int = 4,
        batch_size: int = 1000,
    ) -> None:
        self.prompt_manager = prompt_manager
        self.template_name = template_name
        self.output_column = output_column
        self.retrieve_column = retrieve_column
        self.num_proc = num_proc
        self.batch_size = batch_size

    def _format_batch(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        keys = list(examples.keys())
        batch_length = len(examples[keys[0]])

        formatted_prompts = []

        for i in range(batch_length):
            kwargs = {k: examples[k][i] for k in keys}

            # Всегда передаём context в шаблон если retrieve_column задан.
            # Если колонки нет в датасете (бенчмарк без контекста) → None.
            # Если колонка есть но пустая строка → None.
            # Шаблон обрабатывает оба случая через {% if context %}.
            if self.retrieve_column is not None:
                if self.retrieve_column in examples:
                    raw_ctx = examples[self.retrieve_column][i]
                    kwargs["context"] = raw_ctx if raw_ctx else None
                else:
                    kwargs["context"] = None

            formatted = self.prompt_manager.render(self.template_name, **kwargs)
            formatted_prompts.append(formatted)

        return {self.output_column: formatted_prompts}

    def __call__(self, dataset: HFDataset) -> HFDataset:
        logger.info(
            "Сборка промптов: output='%s', шаблон='%s', retrieve_column='%s'",
            self.output_column, self.template_name, self.retrieve_column,
        )
        return dataset.map(
            self._format_batch,
            batched=True,
            batch_size=self.batch_size,
            num_proc=self.num_proc,
            desc=f"Formatting Prompts ({self.template_name})",
        )