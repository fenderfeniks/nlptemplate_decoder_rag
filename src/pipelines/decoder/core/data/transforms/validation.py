# src/pipelines/decoder/core/data/transforms/validation.py
import logging
from typing import Any

from pydantic import ValidationError

from src.pipelines.decoder.core.data.schemas import RawDatasetRecord
from src.pipelines.base.core.data.transforms.validation import BaseValidationTransform

logger = logging.getLogger(__name__)


class DecoderValidationTransform(BaseValidationTransform):
    """Валидация сырых записей датасета для decoder-пайплайна (SFT / RAG).

    Поддерживает три колонки:
        - prompt_column  (обязательная) — вопрос / инструкция
        - target_column  (обязательная) — ожидаемый ответ
        - context_column (опциональная) — RAG-контекст; если колонка есть
          в датасете, её значение прокидывается в схему, но пустая строка
          считается допустимой (модель должна уметь отвечать "не знаю").

    Имена колонок берутся из конфига data через Hydra-интерполяцию:
        prompt_column:  "${data.prompt_column}"
        target_column:  "${data.target_column}"
        context_column: "${data.retrieve_column}"   # null если не RAG

    Записи с невалидными prompt/target заменяются пустыми строками и затем
    вырезаются финальным filter в BaseValidationTransform.__call__.
    """

    def __init__(
        self,
        prompt_column: str = "prompt",
        target_column: str = "target",
        context_column: str | None = None,   # опциональный RAG-контекст
        num_proc: int = 4,
        batch_size: int = 1000,
    ) -> None:
        self.prompt_column = prompt_column
        self.target_column = target_column
        self.context_column = context_column
        super().__init__(mode="sft", num_proc=num_proc, batch_size=batch_size)

    def _validate_mode(self) -> None:
        pass

    def _get_required_columns(self) -> list[str]:
        # context_column — опциональная; отсутствие в датасете не ошибка
        return [self.prompt_column, self.target_column]

    def _get_filter_column(self) -> str:
        return self.prompt_column

    def _validate_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        keys = list(batch.keys())
        valid_batch: dict[str, list[Any]] = {k: [] for k in keys}

        prompts = batch.get(self.prompt_column, [])
        targets = batch.get(self.target_column, [])

        # Контекст — читаем если колонка задана И присутствует в батче;
        # иначе передаём None (схема допускает Optional[str])
        has_context = (
            self.context_column is not None
            and self.context_column in batch
        )

        for i, (p, t) in enumerate(zip(prompts, targets)):
            ctx = batch[self.context_column][i] if has_context else None

            # Нормализуем пустую строку контекста → None, чтобы схема
            # не поднимала ошибку на записях без RAG-контекста
            if ctx == "":
                ctx = None

            try:
                RawDatasetRecord(prompt=p, target=t, context=ctx)

                # Валидация прошла — сохраняем всю строку как есть
                for k in keys:
                    valid_batch[k].append(batch[k][i])

            except ValidationError as e:
                logger.debug(
                    "Отброшена битая запись (prompt=%r, target=%r): %s",
                    p[:80] if isinstance(p, str) else p,
                    t[:80] if isinstance(t, str) else t,
                    e,
                )
                # Помечаем запись пустой строкой — финальный filter её вырежет
                for k in keys:
                    valid_batch[k].append("" if isinstance(batch[k][i], str) else None)

        return valid_batch