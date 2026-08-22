from unittest.mock import MagicMock

import pandas as pd
import pytest

# Укажи правильный путь импорта в зависимости от структуры проекта
from src.evaluation.evaluators.decoder import DecoderEvaluator, _cfg_get


# ===========================================================================
# Фикстуры и настройка окружения
# ===========================================================================


@pytest.fixture
def mock_instantiate(mocker):
    """Мокает инстанциацию Hydra для metrics_pipeline."""
    return mocker.patch("src.evaluation.evaluators.decoder.instantiate")


@pytest.fixture
def mock_generator_cls(mocker):
    """Мокает динамический импорт HFTextGenerator."""
    mock_module = MagicMock()
    mock_cls = MagicMock()
    mock_module.HFTextGenerator = mock_cls
    mocker.patch.dict("sys.modules", {"src.pipelines.decoder.inference.generator": mock_module})
    return mock_cls


@pytest.fixture
def mock_experiment_logger():
    return MagicMock()


@pytest.fixture
def base_eval_dataset():
    return [
        {"prompt": "Q1", "response": "A1"},
        {"prompt": "Q2", "response": "A2"},
        {"prompt": "Q3", "response": "A3"},
    ]


# ===========================================================================
# Тесты утилит
# ===========================================================================


def test_cfg_get():
    """Проверка извлечения ключей из словарей и объектов."""
    # Из словаря
    assert _cfg_get({"key": "value"}, "key") == "value"
    assert _cfg_get({"key": "value"}, "missing", "default") == "default"

    # Из объекта (например, DictConfig)
    class DummyConfig:
        key = "obj_value"

    assert _cfg_get(DummyConfig(), "key") == "obj_value"
    assert _cfg_get(DummyConfig(), "missing", "default") == "default"


# ===========================================================================
# Тесты инициализации и подготовки (Setup)
# ===========================================================================


class TestDecoderEvaluatorSetup:
    def test_setup_dataset_success(self, base_eval_dataset):
        """Успешная регистрация датасета для конкретного stage."""
        evaluator = DecoderEvaluator(model_name="test")

        evaluator._setup_dataset("val", base_eval_dataset)

        assert evaluator._env_ready["val"] is True
        assert len(evaluator._eval_datasets["val"]) == 3

    def test_setup_dataset_missing_raises_error(self):
        """Если датасет не передан и не готов, выбрасывается ValueError."""
        evaluator = DecoderEvaluator(model_name="test")

        with pytest.raises(ValueError, match="eval_dataset для stage='val' не передан"):
            evaluator._setup_dataset("val", None)

    def test_setup_generator_explicit_args(self, mock_generator_cls):
        """Разрешение модели и токенизатора из явных аргументов."""
        evaluator = DecoderEvaluator(model_name="test")

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        evaluator._setup_generator(
            trainer=None, pl_module=None, model=mock_model, tokenizer=mock_tokenizer
        )

        mock_generator_cls.assert_called_once_with(
            model=mock_model, tokenizer=mock_tokenizer, generation_kwargs={}
        )
        assert evaluator._generator == mock_generator_cls.return_value

    def test_setup_generator_from_lightning(self, mock_generator_cls):
        """Разрешение модели и токенизатора из объектов PyTorch Lightning."""
        evaluator = DecoderEvaluator(model_name="test")

        mock_trainer = MagicMock()
        mock_trainer.datamodule.tokenizer = "pl_tokenizer"

        mock_pl_module = MagicMock()
        mock_pl_module.model = "pl_model"

        evaluator._setup_generator(
            trainer=mock_trainer, pl_module=mock_pl_module, model=None, tokenizer=None
        )

        mock_generator_cls.assert_called_once_with(
            model="pl_model", tokenizer="pl_tokenizer", generation_kwargs={}
        )

    def test_setup_generator_missing_deps_raises_error(self, mock_generator_cls):
        """Если зависимости не найдены, выбрасывается ошибка."""
        evaluator = DecoderEvaluator(model_name="test")

        with pytest.raises(ValueError, match="Не удалось разрешить model/tokenizer"):
            evaluator._setup_generator(trainer=None, pl_module=None, model=None, tokenizer=None)


# ===========================================================================
# Тесты генерации и сбора статистики
# ===========================================================================


class TestDecoderEvaluatorGeneration:
    def test_generate_chunks_with_stats(self, mocker):
        """Проверка батчирования, таймингов и подсчета токенов."""
        evaluator = DecoderEvaluator(model_name="test", generation_batch_size=2)

        # Мокаем внутренний генератор
        mock_generator = MagicMock()
        # Симулируем генерацию (для батча из 2 вернет 2 ответа, для батча из 1 - 1 ответ)
        mock_generator.generate.side_effect = [
            ["Gen 1", "Gen 2"],  # Ответ на первый батч
            ["Gen 3"],  # Ответ на второй батч
        ]

        # Токенизатор: каждое слово = 1 токен (упрощенная симуляция)
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda text, **kwargs: text.split()
        mock_generator.tokenizer = mock_tokenizer

        evaluator._generator = mock_generator

        # Мокаем время: каждый вызов generate будет занимать 2.0 секунды
        mocker.patch("time.perf_counter", side_effect=[0.0, 2.0, 2.0, 4.0])
        # Мокаем очистку CUDA
        mock_empty_cache = mocker.patch("torch.cuda.empty_cache")
        mocker.patch("torch.cuda.is_available", return_value=True)

        prompts = ["Prompt one", "Prompt two", "Short"]

        generated, stats = evaluator._generate_chunks_with_stats(prompts)

        # 1. Проверяем возвращаемые тексты
        assert generated == ["Gen 1", "Gen 2", "Gen 3"]

        # 2. Проверяем вызовы генератора (разбито на батчи 2 и 1)
        assert mock_generator.generate.call_count == 2
        mock_generator.generate.assert_any_call(["Prompt one", "Prompt two"])
        mock_generator.generate.assert_any_call(["Short"])

        # 3. Проверяем кэш CUDA
        assert mock_empty_cache.call_count == 2

        # 4. Проверяем статистику
        assert len(stats) == 3
        # Первый батч: 2 секунды на 2 промпта = 1.0s/sample
        assert stats[0] == {"latency_s": 1.0, "prompt_tokens": 2, "generated_tokens": 2}
        assert stats[1] == {"latency_s": 1.0, "prompt_tokens": 2, "generated_tokens": 2}
        # Второй батч: 2 секунды на 1 промпт = 2.0s/sample
        assert stats[2] == {"latency_s": 2.0, "prompt_tokens": 1, "generated_tokens": 2}

    def test_token_counting_fallback(self):
        """Если токенизатора нет или он падает, используется len(text.split())."""
        evaluator = DecoderEvaluator(model_name="test", generation_batch_size=1)
        mock_generator = MagicMock()
        mock_generator.generate.return_value = ["A long response here"]
        # Токенизатора нет
        mock_generator.tokenizer = None
        evaluator._generator = mock_generator

        _, stats = evaluator._generate_chunks_with_stats(["Prompt 1"])

        # "Prompt 1" -> 2 слова; "A long response here" -> 4 слова
        assert stats[0]["prompt_tokens"] == 2
        assert stats[0]["generated_tokens"] == 4


# ===========================================================================
# Тесты полного пайплайна оценки (evaluate)
# ===========================================================================


class TestDecoderEvaluatorFullFlow:
    def test_evaluate_empty_batch(self, mock_experiment_logger):
        """Если датасет и fixed_samples пусты, метод возвращает пустой словарь."""
        evaluator = DecoderEvaluator(model_name="test", num_random=0)
        # Настраиваем окружение, но данных нет
        evaluator._env_ready["val"] = True
        evaluator._eval_datasets["val"] = []
        evaluator._generator = MagicMock()

        res = evaluator.evaluate(stage="val", metrics_logger=mock_experiment_logger)

        assert res == {}
        mock_experiment_logger.log_metrics.assert_not_called()

    def test_evaluate_full_pipeline(
        self, base_eval_dataset, mock_experiment_logger, mock_instantiate, mocker
    ):
        """Комплексная проверка: семплинг, генерация, расчет метрик и логирование."""
        # Настраиваем фикстуры
        mock_metrics_pipeline = MagicMock()
        mock_metrics_pipeline.compute_all.return_value = {"rouge1": 0.9}
        mock_instantiate.return_value = mock_metrics_pipeline

        fixed_samples = [{"prompt": "Fixed Q", "target": "Fixed A"}]
        evaluator = DecoderEvaluator(
            model_name="test",
            num_random=1,
            fixed_samples=fixed_samples,
            metrics_cfg={"_target_": "dummy"},
        )

        # Мокаем генерацию (возвращает то же количество ответов, сколько пришло)
        mocker.patch.object(
            evaluator,
            "_generate_chunks_with_stats",
            return_value=(["Gen Fixed", "Gen Random"], [{"stat": 1}, {"stat": 2}]),
        )

        # === ВЫПОЛНЕНИЕ ===
        metrics = evaluator.evaluate(
            stage="val",
            metrics_logger=mock_experiment_logger,
            model="dummy_model",
            tokenizer="dummy_tokenizer",
            eval_dataset=base_eval_dataset,
            contexts=[["context1"], ["context2"]],
        )

        # 1. Проверяем вызов compute_all
        assert metrics == {"rouge1": 0.9}
        mock_metrics_pipeline.compute_all.assert_called_once()
        call_kwargs = mock_metrics_pipeline.compute_all.call_args.kwargs

        # Должен быть 1 фиксированный и 1 случайный сэмпл
        assert len(call_kwargs["prompts"]) == 2
        assert call_kwargs["prompts"][0] == "Fixed Q"
        assert call_kwargs["generated"] == ["Gen Fixed", "Gen Random"]
        assert call_kwargs["contexts"] == [["context1"], ["context2"]]
        assert call_kwargs["extra"]["generation_stats"] == [{"stat": 1}, {"stat": 2}]

        # 2. Проверяем логирование DataFrame и метрик
        mock_experiment_logger.log_metrics.assert_called_once_with(
            metrics={"rouge1": 0.9}, stage="val", step=0
        )

        assert mock_experiment_logger.log_table.call_count == 1
        df_logged: pd.DataFrame = mock_experiment_logger.log_table.call_args.kwargs["df"]

        assert len(df_logged) == 2
        assert list(df_logged["Type"]) == ["Fixed", "Random"]
        assert list(df_logged["Prompt"])[0] == "Fixed Q"
