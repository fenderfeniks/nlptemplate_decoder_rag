# src/tools/evaluation/judges/nli_judge.py
"""NLI-Judge на базе RoBERTa (или любой NLI-модели) через манифест + HFModelBuilder."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import pipeline as hf_pipeline

from src.tools.evaluation.judges.base import BaseJudge
from src.tools.evaluation.schema import EvalInput, EvalResult


logger = logging.getLogger(__name__)

# Метки NLI в порядке как их возвращают большинство моделей (cross-encoder/nli)
# Переопределяются через конфиг если у конкретной модели другой порядок
_DEFAULT_LABEL_MAP = {
    "entailment": 1.0,  # ответ соответствует референсу
    "neutral": 0.5,
    "contradiction": 0.0,
}


class NLIJudge(BaseJudge):
    """Judge на базе NLI-модели (RoBERTa-large-mnli и аналоги).

    Использует манифест для получения пути к весам — точно так же как
    ArtifactResolver в eval.py. Загружает модель один раз при инициализации.

    Логика оценки:
    - premise   = reference (эталонный ответ)
    - hypothesis = response (ответ модели)
    - entailment score → EvalResult.score ∈ [0.0, 1.0]
    - verdict = score >= verdict_threshold

    Если reference отсутствует — оценивает (prompt, response) как (premise, hypothesis).
    Это менее точно, но позволяет работать без разметки.

    Конфигурируется через configs/evaluation/nli/default.yaml.
    """

    def __init__(
        self,
        # --- Источник модели ---
        manifest_uri: str,
        router,  # StorageRouter — инжектируется Hydra
        cache_dir: str,  # куда скачивать веса из storage
        # --- Параметры модели ---
        tokenizer_name: str | None = None,  # если None — берётся из manifest model_uri
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 512,
        # --- Логика оценки ---
        entailment_label: str = "entailment",
        label_map: dict[str, float] | None = None,
        verdict_threshold: float = 0.5,
        return_score: bool = True,
        return_verdict: bool = True,
        return_reasoning: bool = False,
    ) -> None:
        self.verdict_threshold = verdict_threshold
        self.return_score = return_score
        self.return_verdict = return_verdict
        self.return_reasoning = return_reasoning
        self.entailment_label = entailment_label
        self.label_map = label_map or _DEFAULT_LABEL_MAP
        self.batch_size = batch_size

        # ------------------------------------------------------------------
        # 1. Резолвим путь к весам через манифест (тот же механизм что в eval.py)
        # ------------------------------------------------------------------
        logger.info("NLIJudge: загрузка манифеста '%s'", manifest_uri)
        cache_base = Path(cache_dir)
        manifest = router.download_manifest(manifest_uri, cache_base / "manifests")

        if manifest.get("load_type") != "full_model":
            raise ValueError(
                f"NLI-модель ожидает load_type=full_model, получено: {manifest.get('load_type')}. "
                "Убедитесь что prepare_artifacts запущен для NLI-пайплайна."
            )

        model_path = router.download_from_uri(manifest["model_uri"], cache_base / "nli_model")
        logger.info("NLIJudge: веса получены из storage: %s", model_path)

        # ------------------------------------------------------------------
        # 2. Определяем устройство
        # ------------------------------------------------------------------
        if device == "auto":
            resolved_device = 0 if torch.cuda.is_available() else -1
        elif device == "cpu":
            resolved_device = -1
        else:
            resolved_device = int(device.replace("cuda:", ""))

        # ------------------------------------------------------------------
        # 3. Загружаем pipeline один раз
        # ------------------------------------------------------------------
        tokenizer_source = str(tokenizer_name or model_path)
        logger.info(
            "NLIJudge: инициализация pipeline (device=%s, batch_size=%d)",
            resolved_device,
            batch_size,
        )
        self._pipeline = hf_pipeline(
            task="text-classification",
            model=str(model_path),
            tokenizer=tokenizer_source,
            device=resolved_device,
            batch_size=batch_size,
            truncation=True,
            max_length=max_length,
            top_k=None,  # возвращаем scores для всех меток
        )
        logger.info("NLIJudge: готов.")

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _make_pairs(self, inputs: list[EvalInput]) -> list[dict]:
        """Формирует пары (premise, hypothesis) для NLI-pipeline."""
        pairs = []
        for inp in inputs:
            premise = inp.reference if inp.reference else inp.prompt
            pairs.append({"text": premise, "text_pair": inp.response})
        return pairs

    def _extract_score(self, label_scores: list[dict]) -> float:
        """Извлекает entailment score из списка {label, score}."""
        for item in label_scores:
            if item["label"].lower() == self.entailment_label.lower():
                return float(item["score"])
        # Fallback: ищем по label_map
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
        for inp, label_scores in zip(inputs, raw_outputs):  # noqa
            # hf pipeline с top_k=None возвращает list[dict] на каждый пример
            score = self._extract_score(label_scores)
            verdict = score >= self.verdict_threshold if self.return_verdict else None

            reasoning = None
            if self.return_reasoning:
                # Формируем человекочитаемое объяснение из распределения меток
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
