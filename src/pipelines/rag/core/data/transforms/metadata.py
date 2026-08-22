# src/pipelines/rag/core/data/transforms/metadata.py
import logging
from typing import Any

from datasets import Dataset as HFDataset

from src.pipelines.base.core.data.transforms.base import BaseDatasetTransform

logger = logging.getLogger(__name__)


class MetadataInjectorTransform(BaseDatasetTransform):
    """Вклеивает словарь метаданных в начало текста через шаблон.

    Пример результата при шаблоне по умолчанию::

        {'title': 'Transformer', 'date': '2017'} + 'Текст статьи...'
        ->
        'Title: Transformer\\nDate: 2017\\n\\nТекст статьи...'

    Записи с пустыми или отсутствующими метаданными возвращаются без изменений.
    Ключи метаданных с пустыми значениями молча пропускаются.

    .. note:: Шаблон должен содержать плейсхолдеры ``{meta_string}`` и ``{text}``.
        Произвольный порядок и дополнительное форматирование допустимы,
        например: ``'[DOC]\\n{meta_string}\\n\\n{text}\\n[/DOC]'``.
    """

    def __init__(
        self,
        text_column: str = "text",
        metadata_column: str = "metadata",
        template: str = "{meta_string}\n\n{text}",
        num_proc: int = 4,
        batch_size: int = 1000,
    ) -> None:
        """
        Args:
            text_column: Колонка с исходным текстом документа.
            metadata_column: Колонка со словарём метаданных. Если колонка
                отсутствует в датасете — трансформ пропускается с предупреждением.
            template: Шаблон для форматирования результата. Должен содержать
                плейсхолдеры ``{meta_string}`` и ``{text}``.
            num_proc: Число процессов для параллельного map.
            batch_size: Размер батча для map.

        Raises:
            ValueError: Если ``template`` не содержит плейсхолдеры
                ``{meta_string}`` и ``{text}``.
        """
        if "{meta_string}" not in template or "{text}" not in template:
            raise ValueError(
                f"template должен содержать плейсхолдеры '{{meta_string}}' и '{{text}}', "
                f"получено: {template!r}"
            )
        self.text_column = text_column
        self.metadata_column = metadata_column
        self.template = template
        self.num_proc = num_proc
        self.batch_size = batch_size

    def __call__(self, dataset: HFDataset) -> HFDataset:
        if self.text_column not in dataset.column_names:
            logger.warning(
                "Колонка '%s' не найдена в датасете — инъекция метаданных пропущена. "
                "Убедитесь, что колонка с текстом задана корректно через параметр text_column.",
                self.text_column,
            )
            return dataset

        if self.metadata_column not in dataset.column_names:
            logger.warning(
                "Колонка '%s' не найдена в датасете — инъекция метаданных пропущена. "
                "Убедитесь, что колонка с метаданными задана корректно через параметр metadata_column.",
                self.metadata_column,
            )
            return dataset

        logger.info(
            "Инъекция метаданных из '%s' в '%s'...",
            self.metadata_column,
            self.text_column,
        )

        def _inject(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
            new_texts = []
            for text, meta in zip(batch[self.text_column], batch[self.metadata_column], strict=True):
                if not meta:
                    new_texts.append(text)
                    continue

                meta_parts = [
                    f"{str(k).capitalize()}: {v}"
                    for k, v in meta.items()
                    if v is not None and str(v).strip()
                ]
                meta_string = "\n".join(meta_parts)

                if meta_string:
                    new_texts.append(self.template.format(meta_string=meta_string, text=text))
                else:
                    new_texts.append(text)

            return {self.text_column: new_texts}

        result = dataset.map(
            _inject,
            batched=True,
            batch_size=self.batch_size,
            num_proc=self.num_proc,
            desc="Injecting metadata",
        )

        logger.info(
            "Инъекция метаданных завершена: обработано %d записей.",
            len(result),
        )
        return result