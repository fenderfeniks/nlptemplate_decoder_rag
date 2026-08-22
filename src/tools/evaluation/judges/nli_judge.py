# src/tools/evaluation/judges/nli_judge.py
"""NLI-Judge на базе RoBERTa (или любой NLI-модели).

Модель загружается либо снаружи (готовый pipeline), либо через фабричный
метод from_manifest — он резолвит путь из манифеста и собирает pipeline.
Сам класс содержит только логику оценки.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.tools.evaluation.judges.base import BaseJudge
from src.tools.evaluation.schema import EvalInput, EvalResult


logger = logging.getLogger(__name__)

_DEFAULT_LABEL_MAP = {
    "entailment": 1.0,
    "neutral": 0.5,
    "contradiction": 0.0,
}


class NLIJudge(BaseJudge):
    """Judge на базе NLI-модели (RoBERTa-large-mnli и аналоги).

    Логика оценки:
    - premise    = reference (эталонный ответ)
    - hypothesis = response (ответ модели)
    - entailment score -> EvalResult.score ∈ [0.0, 1.0]
    - verdict = score >= verdict_threshold

    Если reference отсутствует — оценивает (prompt, response) как (premise, hypothesis).

    Инициализация:
        # Через готовый pipeline (инъекция снаружи):
        judge = NLIJudge(pipeline=pipe)

        # Через манифест (самостоятельная загрузка):
        judge = NLIJudge.from_manifest(router, manifest_uri, cache_base)
    """

    def __init__(
        self,
        pipeline: Any,
        entailment_label: str = "entailment",
        label_map: dict[str, float] | None = None,
        verdict_threshold: float = 0.5,
        return_score: bool = True,
        return_verdict: bool = True,
        return_reasoning: bool = False,
    ) -> None:
        self._pipeline = pipeline
        self.verdict_threshold = verdict_threshold
        self.return_score = return_score
        self.return_verdict = return_verdict
        self.return_reasoning = return_reasoning
        self.entailment_label = entailment_label
        self.label_map = label_map or _DEFAULT_LABEL_MAP
        logger.info("NLIJudge: готов.")

    # ------------------------------------------------------------------
    # Фабричный метод — загрузка через манифест
    # ------------------------------------------------------------------

    @classmethod
    def from_manifest(
        cls,
        router,
        manifest_uri: str,
        cache_base: Path,
        verdict_threshold: float = 0.5,
        return_score: bool = True,
        return_verdict: bool = True,
        return_reasoning: bool = False,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> NLIJudge:
        """Загружает NLI-модель из manifest["nli_pipeline"] и возвращает готовый judge.

        Args:
            router:           StorageRouter — умеет download_manifest и download_from_uri.
            manifest_uri:     URI единого манифеста (system.manifest.uri).
            cache_base:       Локальная директория для кэша весов.
            verdict_threshold: Порог для binary verdict.
            batch_size:       Размер батча для HF pipeline.
            max_length:       Максимальная длина входа (токенов).

        Raises:
            KeyError:   Если "nli_pipeline" не найден в манифесте.
            ValueError: Если load_type != "full_model".
        """
        import torch
        from transformers import pipeline as hf_pipeline

        logger.info("NLIJudge: загрузка из манифеста '%s'", manifest_uri)
        full_manifest = router.download_manifest(manifest_uri, cache_base / "nli_manifest")

        pipeline_key = "nli_pipeline"
        if pipeline_key not in full_manifest:
            raise KeyError(
                f"Пайплайн '{pipeline_key}' не найден в манифесте {manifest_uri}. "
                "Запустите prepare_artifacts.py pipeline_name=nli_pipeline"
            )

        manifest = full_manifest[pipeline_key]
        if manifest.get("load_type") != "full_model":
            raise ValueError(
                f"NLI-модель ожидает load_type=full_model, получено: {manifest.get('load_type')}."
            )

        model_path = router.download_from_uri(manifest["model_uri"], cache_base / "nli_model")
        logger.info("NLIJudge: веса получены из storage: %s", model_path)

        device = 0 if torch.cuda.is_available() else -1
        pipe = hf_pipeline(
            task="text-classification",
            model=str(model_path),
            tokenizer=str(model_path),
            device=device,
            batch_size=batch_size,
            truncation=True,
            max_length=max_length,
            top_k=None,
        )
        logger.info("NLIJudge: pipeline готов (device=%d).", device)

        return cls(
            pipeline=pipe,
            verdict_threshold=verdict_threshold,
            return_score=return_score,
            return_verdict=return_verdict,
            return_reasoning=return_reasoning,
        )

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _make_pairs(self, inputs: list[EvalInput]) -> list[dict]:
        pairs = []
        for inp in inputs:
            premise = inp.reference if inp.reference else inp.prompt
            pairs.append({"text": premise, "text_pair": inp.response})
        return pairs

    def _extract_score(self, label_scores: list[dict]) -> float:
        for item in label_scores:
            if item["label"].lower() == self.entailment_label.lower():
                return float(item["score"])
        for item in label_scores:
            mapped = self.label_map.get(item["label"].lower())
            if mapped is not None:
                return float(item["score"]) * mapped
        return 0.0

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    def evaluate_batch(self, inputs: list[EvalInput]) -> list[EvalResult]:
        pairs = self._make_pairs(inputs)

        try:
            raw_outputs = self._pipeline(pairs)
        except Exception as e:
            logger.error("NLIJudge: сбой pipeline: %s", e)
            return [EvalResult(metadata=inp.metadata) for inp in inputs]

        results = []
        for inp, label_scores in zip(inputs, raw_outputs, strict=True):
            score = self._extract_score(label_scores)
            verdict = score >= self.verdict_threshold if self.return_verdict else None

            reasoning = None
            if self.return_reasoning:
                scores_str = ", ".join(f"{d['label']}={d['score']:.3f}" for d in label_scores)
                reasoning = f"NLI distribution: [{scores_str}]"

            results.append(
                EvalResult(
                    score=score if self.return_score else None,
                    verdict=verdict,
                    reasoning=reasoning,
                    raw=label_scores,
                    metadata=inp.metadata,
                )
            )

        return results
