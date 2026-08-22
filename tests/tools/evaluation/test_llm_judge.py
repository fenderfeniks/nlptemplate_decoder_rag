from unittest.mock import MagicMock

import pytest

# Укажи правильный путь импорта в зависимости от структуры проекта
from src.tools.evaluation.judges.llm_judge import (
    LLMJudge,
    LocalQAGenerator,
    _parse_llm_json,
)
from src.tools.evaluation.schema import EvalInput


# ===========================================================================
# Фикстуры
# ===========================================================================


@pytest.fixture
def env_api_key(monkeypatch):
    """Обеспечивает наличие API ключа для инициализации LLMJudge."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")


@pytest.fixture
def mock_openai(mocker):
    """Мокает клиент OpenAI, чтобы не делать реальные сетевые запросы."""
    mock = mocker.patch("src.tools.evaluation.judges.llm_judge.OpenAI")
    return mock.return_value


@pytest.fixture
def mock_sleep(mocker):
    """Мокает time.sleep, чтобы тесты с ретраями проходили мгновенно."""
    return mocker.patch("src.tools.evaluation.judges.llm_judge.time.sleep")


# ===========================================================================
# Тесты парсера JSON (Утилиты)
# ===========================================================================


class TestParseLLMJson:
    def test_parse_clean_json(self):
        raw = '{"score": 5.0, "verdict": true, "reasoning": "Good."}'
        score, verdict, reasoning = _parse_llm_json(raw, min_score=1.0, max_score=5.0)

        assert score == 1.0  # Нормализовано: (5-1)/(5-1) = 1.0
        assert verdict is True
        assert reasoning == "Good."

    def test_parse_markdown_wrapper(self):
        raw = '```json\n{"score": 3.0, "verdict": false}\n```'
        score, verdict, reasoning = _parse_llm_json(raw, min_score=1.0, max_score=5.0)

        assert score == 0.5  # Нормализовано: (3-1)/(5-1) = 0.5
        assert verdict is False
        assert reasoning is None

    def test_parse_invalid_json(self):
        raw = "I think the score is 5 because..."
        score, verdict, reasoning = _parse_llm_json(raw, min_score=1.0, max_score=5.0)

        assert score is None
        assert verdict is None
        assert reasoning is None

    def test_score_normalization_out_of_bounds(self):
        """Если LLM выдала оценку за пределами диапазона, она должна клиппаться."""
        raw_over = '{"score": 10.0}'
        score_over, _, _ = _parse_llm_json(raw_over, 1.0, 5.0)
        assert score_over == 1.0  # Максимум 1.0

        raw_under = '{"score": -5.0}'
        score_under, _, _ = _parse_llm_json(raw_under, 1.0, 5.0)
        assert score_under == 0.0  # Минимум 0.0

    def test_verdict_string_parsing(self):
        """Проверка нестандартных строковых булевых значений от LLM."""
        raw = '{"verdict": "Yes"}'
        _, verdict, _ = _parse_llm_json(raw, 1.0, 5.0)
        assert verdict is True

        raw_fail = '{"verdict": "Fail"}'
        _, verdict_fail, _ = _parse_llm_json(raw_fail, 1.0, 5.0)
        assert verdict_fail is False


# ===========================================================================
# Тесты LLMJudge
# ===========================================================================


class TestLLMJudge:
    def test_missing_api_key_raises_error(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(OSError, match="не задана. Добавьте её в .env файл."):
            LLMJudge(model="test/model")

    def test_build_prompt_with_and_without_reference(self, env_api_key, mock_openai):
        judge = LLMJudge(model="test")

        inp_no_ref = EvalInput(prompt="Q1", response="A1")
        prompt_no_ref = judge._build_prompt(inp_no_ref)
        assert "### Reference Answer" not in prompt_no_ref
        assert "A1" in prompt_no_ref

        inp_ref = EvalInput(prompt="Q2", response="A2", reference="Ref2")
        prompt_ref = judge._build_prompt(inp_ref)
        assert "### Reference Answer\nRef2\n" in prompt_ref

    def test_call_api_success_with_rate_limit(self, env_api_key, mock_openai, mock_sleep, mocker):
        # Настраиваем мок ответа OpenAI
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"score": 4}'
        mock_openai.chat.completions.create.return_value = mock_response

        # Искусственно делаем так, чтобы _last_request_time потребовал ожидания
        judge = LLMJudge(model="test", requests_per_minute=60)
        mocker.patch(
            "time.monotonic", side_effect=[0.0, 0.5, 0.5]
        )  # elapsed = 0.5, wait = 1.0 - 0.5 = 0.5

        result = judge._call_api("Test prompt")

        assert result == '{"score": 4}'
        mock_sleep.assert_called_once()
        mock_openai.chat.completions.create.assert_called_once()

    def test_call_api_retries_on_failure(self, env_api_key, mock_openai, mock_sleep):
        """Если API падает, код должен сделать retry_attempts попыток и подождать между ними."""
        mock_openai.chat.completions.create.side_effect = Exception("API Error")

        judge = LLMJudge(model="test", retry_attempts=3, retry_delay=1.0)

        with pytest.raises(Exception, match="API Error"):
            judge._call_api("Prompt")

        assert mock_openai.chat.completions.create.call_count == 3
        # Sleep должен быть вызван 2 раза (после 1 и 2 попытки)
        assert mock_sleep.call_count == 2

    def test_evaluate_batch(self, env_api_key, mock_openai):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"score": 3.0, "verdict": false}'
        mock_openai.chat.completions.create.return_value = mock_response

        judge = LLMJudge(model="test", min_score=1.0, max_score=5.0, return_reasoning=False)
        inputs = [EvalInput(prompt="Q", response="A", metadata={"id": 1})]

        results = judge.evaluate_batch(inputs)

        assert len(results) == 1
        assert results[0].score == 0.5
        assert results[0].verdict is False
        assert results[0].reasoning is None
        assert results[0].metadata == {"id": 1}

    def test_evaluate_batch_exception_handling(self, env_api_key, mock_openai, mock_sleep):
        """Сбой одного примера не должен ронять весь батч, возвращаем пустой EvalResult."""
        mock_openai.chat.completions.create.side_effect = Exception("Fatal API crash")

        judge = LLMJudge(model="test")
        inputs = [EvalInput(prompt="Q", response="A")]

        results = judge.evaluate_batch(inputs)

        assert len(results) == 1
        assert results[0].score is None
        assert results[0].verdict is None
        assert "Fatal API crash" in results[0].raw


# ===========================================================================
# Тесты LocalQAGenerator
# ===========================================================================


class TestLocalQAGenerator:
    def test_parse_valid_output(self):
        raw = '```json\n{"question": "Q?", "answer": "A!"}\n```'
        result = LocalQAGenerator._parse(raw)
        assert result == ("Q?", "A!")

    def test_parse_missing_keys_or_invalid_json(self):
        assert LocalQAGenerator._parse('{"question": "Q"}') is None
        assert LocalQAGenerator._parse("Just text") is None

    def test_generate_success(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": '{"question": "What?", "answer": "Yes."}'}]

        generator = LocalQAGenerator(pipeline=mock_pipeline, max_new_tokens=100)

        result = generator.generate("Chunk of text.")

        assert result == ("What?", "Yes.")
        mock_pipeline.assert_called_once()
        kwargs = mock_pipeline.call_args.kwargs
        assert kwargs["max_new_tokens"] == 100
        assert kwargs["return_full_text"] is False

    def test_generate_pipeline_exception(self):
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = RuntimeError("OOM")

        generator = LocalQAGenerator(pipeline=mock_pipeline)

        assert generator.generate("Text") is None
