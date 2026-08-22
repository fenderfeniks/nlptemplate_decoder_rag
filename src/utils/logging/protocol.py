# src/utils/logging/protocol.py
"""Протокол логгера эксперимента.

Единственное место где определён интерфейс взаимодействия с experiment tracker'ом.
Бизнес-код импортирует только его — никогда не импортирует mlflow, wandb и т.д. напрямую.

Замена бэкенда = новый файл реализации + одна строка в конфиге Hydra.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class ExperimentLogger(Protocol):
    def log_metrics(
        self,
        metrics: dict[str, float],
        stage: str,
        step: int = 0,
    ) -> None:
        """Логирует числовые метрики с префиксом stage."""
        ...

    def log_table(
        self,
        df: pd.DataFrame,
        stage: str,
        step: int = 0,
        artifact_suffix: str = "",
    ) -> None:
        """Логирует DataFrame как артефакт (таблица примеров генерации)."""
        ...

    def save_adapter(
        self,
        cfg: Any,
        model_module: Any,
        tokenizer: Any,
        run_id: str,
        pipeline_name: str,
        best_score: float | None = None,
    ) -> None:
        """Сохраняет PEFT LoRA-адаптер как артефакт эксперимента."""
        ...

    def load_adapter(
        self,
        resume_cfg: Any,
        tracking_uri: str | None = None,
    ) -> str | None:
        """Загружает PEFT LoRA-адаптер, возвращает локальный путь или None."""
        ...

    def get_run_id(self, trainer: Any = None) -> str | None:
        """Возвращает run_id текущего эксперимента."""
        ...

    def promote_model(
        self,
        reg_model_name: str,
        staging_alias: str = "Staging",
        production_alias: str = "Production",
        metric_tag: str = "val_loss",
    ) -> bool:
        """Продвигает модель из staging в production если она лучше текущей.

        Returns:
            True если продвижение выполнено, False если отклонено.
        """
        ...

    def get_production_version(
        self,
        reg_model_name: str,
        production_alias: str = "Production",
    ) -> str:
        """Возвращает номер версии текущей Production модели."""
        ...

    @contextmanager
    def start_run(self, run_name: str = "") -> Generator[None, None, None]:
        """Открывает новый run (контекстный менеджер для standalone-скриптов)."""
        ...
        yield

    @contextmanager
    def reopen_run(self, run_id: str) -> Generator[None, None, None]:
        """Переоткрывает существующий run для дологирования метрик.

        Используется в post-training когда Lightning цикл уже завершён
        но нужно записать метрики генерации в тот же run что и обучение.
        Реализация сама разбирается с деталями трекера (MLflow experiment_id и т.д.).
        """
        ...
        yield
