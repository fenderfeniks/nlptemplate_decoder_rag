# src/tools/benchmark/schema.py
"""Схемы данных для эталонного бенчмарка RAG."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkRecord:
    """Одна QA-пара эталонного датасета.

    Attributes:
        chunk_id:       SHA-256[:16] чанка — идентичен doc_id в KnowledgeBaseIndexer.
                        Используется как ground_truth при вычислении FNR/Recall@K.
        chunk_text:     Исходный текст чанка (контекст для генерации и верификации).
        question:       Сгенерированный вопрос, ответ на который есть только в chunk_text.
        answer:         Эталонный ответ, выведенный из chunk_text.
        nli_score:      Entailment score от NLIJudge (chunk_text -> answer).
                        Записи с score < threshold отсеяны до сохранения.
        metadata:       Исходные метаданные чанка (url, title, source_type, ...).
        generator_model: Имя модели, которая генерировала QA (для воспроизводимости).
    """

    chunk_id: str
    chunk_text: str
    question: str
    answer: str
    nli_score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    generator_model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkRecord:
        return cls(**data)


@dataclass
class BenchmarkDataset:
    """Коллекция BenchmarkRecord с утилитами сохранения/загрузки.

    Формат хранения — JSONL (один JSON-объект на строку).
    Выбран потому что:
    - совместим с datasets.load_dataset('json', ...)
    - позволяет инкрементальную дозапись без чтения всего файла
    - читаем даже частично (при падении генерации)
    """

    records: list[BenchmarkRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[BenchmarkRecord]:
        return iter(self.records)

    def append(self, record: BenchmarkRecord) -> None:
        self.records.append(record)

    # ------------------------------------------------------------------
    # Сохранение / загрузка
    # ------------------------------------------------------------------

    def save_jsonl(self, path: Path | str) -> None:
        """Сохраняет все записи в JSONL-файл (перезапись)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load_jsonl(cls, path: Path | str) -> BenchmarkDataset:
        """Загружает JSONL-файл в BenchmarkDataset."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Бенчмарк не найден: {path}")
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(BenchmarkRecord.from_dict(json.loads(line)))
        return cls(records=records)

    # ------------------------------------------------------------------
    # Статистика
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Возвращает сводку по датасету для логирования."""
        if not self.records:
            return {"total": 0}
        scores = [r.nli_score for r in self.records]
        return {
            "total": len(self.records),
            "unique_chunks": len({r.chunk_id for r in self.records}),
            "nli_score_mean": round(sum(scores) / len(scores), 4),
            "nli_score_min": round(min(scores), 4),
            "nli_score_max": round(max(scores), 4),
            "models_used": list({r.generator_model for r in self.records}),
        }
