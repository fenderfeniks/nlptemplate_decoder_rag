from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Укажи правильный путь импорта в зависимости от структуры проекта
from src.tools.evaluation.judges.nli_judge import NLIJudge
from src.tools.evaluation.schema import EvalInput


# ===========================================================================
# Тесты базовой логики и вычислений (Юнит + Бизнес-логика)
# ===========================================================================


class TestNLIJudgeEvaluation:
    def test_make_pairs_uses_reference_when_present(self):
        """Если reference задан, он используется как premise."""
        judge = NLIJudge(pipeline=MagicMock())
        inputs = [
            EvalInput(
                prompt="Какой город столица Франции?",
                response="Париж",
                reference="Столица Франции — Париж.",
            )
        ]

        pairs = judge._make_pairs(inputs)

        assert len(pairs) == 1
        assert pairs[0] == {"text": "Столица Франции — Париж.", "text_pair": "Париж"}

    def test_make_pairs_fallback_to_prompt(self):
        """Если reference отсутствует, premise берется из prompt."""
        judge = NLIJudge(pipeline=MagicMock())
        inputs = [
            EvalInput(prompt="Небо синее?", response="Да, небо синее.", reference=None),
            EvalInput(prompt="Трава зеленая?", response="Да.", reference=""),
        ]

        pairs = judge._make_pairs(inputs)

        assert pairs[0] == {"text": "Небо синее?", "text_pair": "Да, небо синее."}
        assert pairs[1] == {"text": "Трава зеленая?", "text_pair": "Да."}

    def test_extract_score_direct_entailment(self):
        """Извлечение скора по точному совпадению с entailment_label (case-insensitive)."""
        judge = NLIJudge(pipeline=MagicMock(), entailment_label="ENTAILMENT")

        raw_output = [
            {"label": "CONTRADICTION", "score": 0.05},
            {"label": "neutral", "score": 0.15},
            {"label": "entailment", "score": 0.80},
        ]

        score = judge._extract_score(raw_output)
        assert score == 0.80

    def test_extract_score_fallback_label_map(self):
        """Если точный entailment не найден, скор взвешивается по label_map."""
        judge = NLIJudge(
            pipeline=MagicMock(),
            entailment_label="positive",
            label_map={"neutral": 0.5, "contradiction": 0.0},
        )

        raw_output = [
            {"label": "neutral", "score": 0.60},
            {"label": "contradiction", "score": 0.40},
        ]

        # 0.60 * 0.5 = 0.30
        score = judge._extract_score(raw_output)
        assert pytest.approx(score) == 0.30

    def test_evaluate_batch_success(self):
        """Успешный прогон батча с формированием вердикта и reasoning."""
        mock_pipeline = MagicMock()
        # Симулируем ответ от HF pipeline для двух пар текстов
        mock_pipeline.return_value = [
            [{"label": "entailment", "score": 0.9}, {"label": "contradiction", "score": 0.1}],
            [{"label": "entailment", "score": 0.3}, {"label": "contradiction", "score": 0.7}],
        ]

        judge = NLIJudge(
            pipeline=mock_pipeline,
            verdict_threshold=0.5,
            return_score=True,
            return_verdict=True,
            return_reasoning=True,
        )

        inputs = [
            EvalInput(prompt="Q1", response="A1", reference="R1", metadata={"id": 1}),
            EvalInput(prompt="Q2", response="A2", reference="R2", metadata={"id": 2}),
        ]

        results = judge.evaluate_batch(inputs)

        assert len(results) == 2
        # Первый сэмпл: score=0.9 >= 0.5 -> verdict=True
        assert results[0].score == 0.9
        assert results[0].verdict is True
        assert "NLI distribution:" in results[0].reasoning
        assert results[0].metadata == {"id": 1}

        # Второй сэмпл: score=0.3 < 0.5 -> verdict=False
        assert results[1].score == 0.3
        assert results[1].verdict is False
        assert results[1].metadata == {"id": 2}

    def test_evaluate_batch_handles_pipeline_exception(self):
        """При падении HF pipeline метод возвращает пустые результаты с метаданными."""
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = RuntimeError("CUDA out of memory")

        judge = NLIJudge(pipeline=mock_pipeline)
        inputs = [EvalInput(prompt="Q1", response="A1", metadata={"item_idx": 42})]

        results = judge.evaluate_batch(inputs)

        assert len(results) == 1
        assert results[0].score is None
        assert results[0].verdict is None
        assert results[0].metadata == {"item_idx": 42}


# ===========================================================================
# Тесты фабричного метода from_manifest
# ===========================================================================


class TestNLIJudgeFromManifest:
    def test_missing_pipeline_key_raises_error(self, tmp_path):
        """Ошибка KeyError, если секция nli_pipeline отсутствует в манифесте."""
        mock_router = MagicMock()
        mock_router.download_manifest.return_value = {"other_pipeline": {}}

        with pytest.raises(KeyError, match="Пайплайн 'nli_pipeline' не найден"):
            NLIJudge.from_manifest(
                router=mock_router, manifest_uri="s3://manifest.json", cache_base=tmp_path
            )

    def test_invalid_load_type_raises_error(self, tmp_path):
        """Ошибка ValueError, если load_type не full_model (например, lora)."""
        mock_router = MagicMock()
        mock_router.download_manifest.return_value = {
            "nli_pipeline": {"load_type": "lora", "model_uri": "s3://..."}
        }

        with pytest.raises(ValueError, match="NLI-модель ожидает load_type=full_model"):
            NLIJudge.from_manifest(
                router=mock_router, manifest_uri="s3://manifest.json", cache_base=tmp_path
            )

    def test_from_manifest_success(self, mocker, tmp_path):
        """Успешное создание судьи через манифест с изоляцией torch и HF pipeline."""
        mock_router = MagicMock()
        mock_router.download_manifest.return_value = {
            "nli_pipeline": {"load_type": "full_model", "model_uri": "s3://models/nli_roberta"}
        }
        mock_router.download_from_uri.return_value = Path("/cache/nli_model")

        # Мокаем torch.cuda.is_available и transformers.pipeline
        mock_torch = mocker.patch("torch.cuda.is_available", return_value=False)
        mock_hf_pipeline = mocker.patch("transformers.pipeline")
        fake_pipe_instance = MagicMock()
        mock_hf_pipeline.return_value = fake_pipe_instance

        judge = NLIJudge.from_manifest(
            router=mock_router,
            manifest_uri="s3://manifest.json",
            cache_base=tmp_path,
            verdict_threshold=0.7,
            batch_size=16,
        )

        # Проверки скачивания
        mock_router.download_manifest.assert_called_once_with(
            "s3://manifest.json", tmp_path / "nli_manifest"
        )
        mock_router.download_from_uri.assert_called_once_with(
            "s3://models/nli_roberta", tmp_path / "nli_model"
        )

        # Проверки сборки HF pipeline
        mock_hf_pipeline.assert_called_once_with(
            task="text-classification",
            model="/cache/nli_model",
            tokenizer="/cache/nli_model",
            device=-1,  # CPU, так как is_available=False
            batch_size=16,
            truncation=True,
            max_length=512,
            top_k=None,
        )

        assert isinstance(judge, NLIJudge)
        assert judge.verdict_threshold == 0.7
        assert judge._pipeline == fake_pipe_instance
