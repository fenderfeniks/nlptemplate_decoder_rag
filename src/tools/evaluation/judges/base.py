# src/tools/evaluation/judges/base.py
"""Базовый интерфейс для всех evaluation judges."""

from abc import ABC, abstractmethod

from src.tools.evaluation.schema import EvalInput, EvalResult


class BaseJudge(ABC):
    """Абстрактный judge.

    Все реализации обязаны поддерживать батчевую оценку.
    Одиночный evaluate() — удобная обёртка над evaluate_batch().
    """

    @abstractmethod
    def evaluate_batch(self, inputs: list[EvalInput]) -> list[EvalResult]:
        """Оценивает список примеров и возвращает результаты в том же порядке."""
        ...

    def evaluate(self, input_: EvalInput) -> EvalResult:
        """Оценивает один пример. Делегирует в evaluate_batch()."""
        return self.evaluate_batch([input_])[0]

    def __call__(self, inputs: list[EvalInput] | EvalInput) -> list[EvalResult] | EvalResult:
        if isinstance(inputs, EvalInput):
            return self.evaluate(inputs)
        return self.evaluate_batch(inputs)
