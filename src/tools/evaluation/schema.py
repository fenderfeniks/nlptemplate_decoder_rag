# src/tools/evaluation/schema.py
"""Общие схемы данных для всех evaluation judges."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalInput:
    """Один пример для оценки.

    Attributes:
        prompt:     Исходный запрос / контекст (то что подавалось модели).
        response:   Ответ оцениваемой модели.
        reference:  Эталонный ответ (опционально — не все метрики требуют).
        metadata:   Произвольные поля для логирования (id примера, источник и т.д.).
    """

    prompt: str
    response: str
    reference: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Результат оценки одного примера.

    Attributes:
        score:      Нормализованный скор [0.0, 1.0] или None если не применимо.
        verdict:    Бинарный вердикт pass/fail или None.
        reasoning:  Текстовое объяснение от judge (только LLM-judge).
        raw:        Сырой ответ judge без обработки — для дебага.
        metadata:   Проброс metadata из EvalInput + служебные поля judge.
    """

    score: float | None = None
    verdict: bool | None = None
    reasoning: str | None = None
    raw: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Удобные свойства
    # ------------------------------------------------------------------

    @property
    def passed(self) -> bool | None:
        """True если verdict=True или score >= порога (если verdict не выставлен)."""
        if self.verdict is not None:
            return self.verdict
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "verdict": self.verdict,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
        }
