# src/pipelines/decoder/core/data/transforms/validation.py
import logging
from typing import Any
from pydantic import ValidationError

from src.pipelines.decoder.core.data.schemas import RawDatasetRecord
from src.pipelines.base.core.data.transforms.validation import BaseValidationTransform

logger = logging.getLogger(__name__)

class DecoderValidationTransform(BaseValidationTransform):
    _VALID_MODES = ("cpt", "sft")

    def __init__(
        self,
        mode: str = "cpt",
        text_column: str = "text",
        prompt_column: str = "prompt",
        target_column: str = "target",
        num_proc: int = 4,
        batch_size: int = 1000,
    ) -> None:
        self.text_column = text_column
        self.prompt_column = prompt_column
        self.target_column = target_column
        super().__init__(mode=mode, num_proc=num_proc, batch_size=batch_size)

    def _validate_mode(self) -> None:
        if self.mode not in self._VALID_MODES:
            raise ValueError(f"Неизвестный режим: '{self.mode}'. Допустимые: {self._VALID_MODES}")

    def _get_required_columns(self) -> list[str]:
        if self.mode == "cpt":
            return [self.text_column]
        return [self.prompt_column, self.target_column]

    def _get_filter_column(self) -> str:
        return self.text_column if self.mode == "cpt" else self.prompt_column

    def _validate_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if self.mode == "cpt":
            return self._validate_cpt_batch(batch)
        return self._validate_sft_batch(batch)

    def _validate_cpt_batch(
            self, batch: dict[str, list[Any]]
        ) -> dict[str, list[Any]]:
            valid_texts: list[str] = []
            for text in batch.get(self.text_column, []):
                try:
                    record = RawDatasetRecord(text=text)
                    valid_texts.append(record.text)
                except ValidationError as e:
                    logger.debug("Отброшена битая запись (cpt): %s", e)
                    valid_texts.append("")
            return {self.text_column: valid_texts}
        
    def _validate_sft_batch(
            self, batch: dict[str, list[Any]]
        ) -> dict[str, list[Any]]:
            valid_prompts: list[str] = []
            valid_targets: list[str] = []
            prompts = batch.get(self.prompt_column, [])
            targets = batch.get(self.target_column, [])
            for p, t in zip(prompts, targets):
                try:
                    record = RawDatasetRecord(prompt=p, target=t)
                    valid_prompts.append(record.prompt)
                    valid_targets.append(record.target)
                except ValidationError as e:
                    logger.debug("Отброшена битая запись (sft): %s", e)
                    valid_prompts.append("")
                    valid_targets.append("")
            return {self.prompt_column: valid_prompts, self.target_column: valid_targets}
