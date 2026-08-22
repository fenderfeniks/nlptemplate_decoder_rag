from unittest.mock import MagicMock

import pytest

# Укажи правильный путь импорта в зависимости от структуры проекта
from src.tools.benchmark.generator import (
    APIQAGenerator,
    BaseQAGenerator,
    LocalQAGenerator,
)


# ===========================================================================
# Фикстуры
# ===========================================================================


@pytest.fixture
def env_api_key(monkeypatch):
    """Обеспечивает наличие API ключа для APIQAGenerator."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")


@pytest.fixture
def mock_openai(mocker):
    """Мокает клиент OpenAI, чтобы не делать реальные сетевые запросы."""
    mock = mocker.patch("src.tools.benchmark.generator.OpenAI")
    return mock.return_value


@pytest.fixture
def mock_sleep(mocker):
    """Мокает time.sleep для мгновенного прохождения тестов с rate-limit и ретраями."""
    return mocker.patch("src.tools.benchmark.generator.time.sleep")


# ===========================================================================
# Тесты BaseQAGenerator
# ===========================================================================


class DummyGenerator(BaseQAGenerator):
    """Фиктивная реализация для тестирования базового класса."""

    def generate(self, chunk_text: str) -> tuple[str, str] | None:
        if chunk_text == "fail":
            return None
        return (f"Q: {chunk_text}", f"A: {chunk_text}")


class TestBaseQAGenerator:
    def test_generate_batch(self):
        """Проверка работы дефолтного последовательного батчинга."""
        gen = DummyGenerator()
        chunks = ["chunk1", "fail", "chunk2"]

        results = gen.generate_batch(chunks)

        assert len(results) == 3
        assert results[0] == ("Q: chunk1", "A: chunk1")
        assert results[1] is None
        assert results[2] == ("Q: chunk2", "A: chunk2")


# ===========================================================================
# Тесты APIQAGenerator
# ===========================================================================


class TestAPIQAGenerator:
    def test_missing_api_key_raises_error(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(OSError, match="не задана. Добавьте её в .env"):
            APIQAGenerator(model="test/model")

    def test_parse_logic(self):
        """Проверка очистки markdown и извлечения ключей из JSON."""
        # Успешный парсинг
        raw = '```json\n{"question": "Q1", "answer": "A1"}\n```'
        assert APIQAGenerator._parse(raw) == ("Q1", "A1")

        # Отсутствие нужных ключей
        bad_json = '{"q": "What?", "a": "Yes"}'
        assert APIQAGenerator._parse(bad_json) is None

        # Сломанный JSON
        broken_json = '{"question": "Q1", "answer": }'
        assert APIQAGenerator._parse(broken_json) is None

    def test_call_api_success_with_rate_limit(self, env_api_key, mock_openai, mock_sleep, mocker):
        """Проверка отправки запроса и расчета времени ожидания (rate limiting)."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"question": "Q?", "answer": "A!"}'
        mock_openai.chat.completions.create.return_value = mock_response

        # 60 rpm = 1 запрос в секунду
        gen = APIQAGenerator(model="test", requests_per_minute=60)

        # Имитируем, что с прошлого запроса прошло только 0.2 секунды
        mocker.patch("time.monotonic", side_effect=[0.0, 0.2, 0.2])

        result = gen._call_api("chunk text")

        assert result == '{"question": "Q?", "answer": "A!"}'
        # Должен уснуть на 0.8 сек (1.0 - 0.2)
        mock_sleep.assert_called_once()
        mock_openai.chat.completions.create.assert_called_once()

    def test_call_api_retries_on_failure(self, env_api_key, mock_openai, mock_sleep):
        """Проверка логики повторных попыток при сетевых ошибках."""
        mock_openai.chat.completions.create.side_effect = Exception("API Error")

        gen = APIQAGenerator(model="test", retry_attempts=3, retry_delay=1.0)

        with pytest.raises(Exception, match="API Error"):
            gen._call_api("Chunk")

        assert mock_openai.chat.completions.create.call_count == 3
        assert mock_sleep.call_count == 2  # Слип вызывается между 1-2 и 2-3 попытками

    def test_generate_flow(self, env_api_key, mock_openai):
        """Проверка полного цикла: вызов API -> парсинг -> возврат кортежа."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"question": "Q1", "answer": "A1"}'
        mock_openai.chat.completions.create.return_value = mock_response

        gen = APIQAGenerator(model="test")
        result = gen.generate("Some text")

        assert result == ("Q1", "A1")

    def test_generate_flow_returns_none_on_crash(self, env_api_key, mock_openai, mock_sleep):
        """При фатальном сбое генератор возвращает None вместо падения скрипта."""
        mock_openai.chat.completions.create.side_effect = Exception("Fatal Error")

        gen = APIQAGenerator(model="test", retry_attempts=1)
        result = gen.generate("Some text")

        assert result is None


# ===========================================================================
# Тесты LocalQAGenerator
# ===========================================================================


class TestLocalQAGenerator:
    def test_generate_success(self):
        """Успешная локальная генерация через пайплайн."""
        mock_pipeline = MagicMock()
        # Пайплайн возвращает список словарей
        mock_pipeline.return_value = [{"generated_text": '{"question": "Q?", "answer": "A!"}'}]

        gen = LocalQAGenerator(pipeline=mock_pipeline, max_new_tokens=100)
        result = gen.generate("Chunk")

        assert result == ("Q?", "A!")
        mock_pipeline.assert_called_once()
        kwargs = mock_pipeline.call_args.kwargs
        assert kwargs["max_new_tokens"] == 100
        assert kwargs["return_full_text"] is False

    def test_generate_pipeline_exception(self):
        """При падении HF пайплайна (OOM, CUDA error) возвращается None."""
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = RuntimeError("CUDA OOM")

        gen = LocalQAGenerator(pipeline=mock_pipeline)
        assert gen.generate("Text") is None

    @pytest.fixture
    def mock_deps_for_manifest(self, mocker):
        """Мокает все тяжелые зависимости Hugging Face и билдера для from_manifest."""
        mock_router = MagicMock()
        mock_builder_cls = mocker.patch("src.tools.benchmark.generator.HFModelBuilder")
        mock_tokenizer_cls = mocker.patch("src.tools.benchmark.generator.AutoTokenizer")
        mock_hf_pipeline = mocker.patch("src.tools.benchmark.generator.hf_pipeline")

        return mock_router, mock_builder_cls, mock_tokenizer_cls, mock_hf_pipeline

    def test_from_manifest_missing_pipeline(self, mock_deps_for_manifest, tmp_path):
        """Проверка валидации: манифест без секции decoder_pipeline."""
        mock_router, _, _, _ = mock_deps_for_manifest
        mock_router.download_manifest.return_value = {"other_pipeline": {}}

        with pytest.raises(KeyError, match="Пайплайн 'decoder_pipeline' не найден"):
            LocalQAGenerator.from_manifest(mock_router, "s3://manifest.json", tmp_path)

    def test_from_manifest_wrong_load_type(self, mock_deps_for_manifest, tmp_path):
        """Проверка валидации: load_type должен быть full_model."""
        mock_router, _, _, _ = mock_deps_for_manifest
        mock_router.download_manifest.return_value = {"decoder_pipeline": {"load_type": "lora"}}

        with pytest.raises(ValueError, match="ожидает load_type=full_model"):
            LocalQAGenerator.from_manifest(mock_router, "s3://manifest.json", tmp_path)

    def test_from_manifest_success(self, mock_deps_for_manifest, tmp_path):
        """Полный флоу успешной сборки генератора из манифеста."""
        mock_router, mock_builder_cls, mock_tokenizer_cls, mock_hf_pipeline = mock_deps_for_manifest

        mock_router.download_manifest.return_value = {
            "decoder_pipeline": {"load_type": "full_model", "model_uri": "s3://models/llama"}
        }
        mock_router.download_from_uri.return_value = tmp_path / "model_weights"

        # Настраиваем мок токенизатора, чтобы у него не было chat_template
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.chat_template = None
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer_instance

        mock_pipe_instance = MagicMock()
        mock_hf_pipeline.return_value = mock_pipe_instance

        # === ВЫПОЛНЕНИЕ ===
        gen = LocalQAGenerator.from_manifest(
            router=mock_router,
            manifest_uri="s3://manifest.json",
            cache_base=tmp_path,
            gen_cfg={"temperature": 0.5},
        )

        # 1. Проверяем работу роутера
        mock_router.download_manifest.assert_called_once()
        mock_router.download_from_uri.assert_called_once_with(
            "s3://models/llama", tmp_path / "decoder_model"
        )

        # 2. Проверяем вызов HFModelBuilder
        mock_builder_cls.assert_called_once()
        builder_kwargs = mock_builder_cls.call_args.kwargs
        assert builder_kwargs["model_name_or_path"] == str(tmp_path / "model_weights")

        # 3. Проверяем добавление chat_template (фоллбэк логика)
        assert mock_tokenizer_instance.chat_template is not None
        assert "{% for message in messages %}" in mock_tokenizer_instance.chat_template

        # 4. Проверяем инициализацию инстанса
        assert isinstance(gen, LocalQAGenerator)
        assert gen.temperature == 0.5
        assert gen._pipeline == mock_pipe_instance
