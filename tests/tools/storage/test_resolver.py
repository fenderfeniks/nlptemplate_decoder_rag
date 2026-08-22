from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

# Укажи правильный путь импорта в зависимости от структуры проекта
from src.tools.storage.resolver import ArtifactResolver


# ===========================================================================
# Фикстуры
# ===========================================================================


@pytest.fixture
def mock_router():
    """Мокает StorageRouter для предотвращения реального скачивания."""
    router = MagicMock()
    return router


@pytest.fixture
def base_cfg():
    """Базовый плоский конфиг, соответствующий ожиданиям _patch_model_path."""
    return OmegaConf.create(
        {
            "model": {
                "builder": {"model_name_or_path": "old_path"},
                "tokenizer": {"tokenizer_name": "old_path"},
                "architecture": {"model_name_or_path": "old_path"},
                "modifiers": {"finetuning": {}},
            },
            "inference": {"bm25": {"index_path": "old_index"}},
        }
    )


@pytest.fixture
def resolver(mock_router, tmp_path):
    """Инициализирует ArtifactResolver с временной директорией кэша."""
    return ArtifactResolver(router=mock_router, cache_base_dir=tmp_path)


# ===========================================================================
# Тесты утилит и парсинга манифеста
# ===========================================================================


class TestResolverUtilities:
    def test_get_model_name_success(self, resolver, mock_router):
        """Успешное извлечение mlflow_model_name."""
        mock_router.download_manifest.return_value = {
            "decoder_pipeline": {"mlflow_model_name": "Llama-3-8B-Instruct"}
        }

        name = resolver.get_model_name("s3://manifest.json", "decoder_pipeline")
        assert name == "Llama-3-8B-Instruct"

    def test_get_model_name_missing_pipeline(self, resolver, mock_router):
        """Ошибка KeyError при отсутствии пайплайна в манифесте."""
        mock_router.download_manifest.return_value = {"other_pipeline": {}}

        with pytest.raises(KeyError, match="Пайплайн 'missing' не найден"):
            resolver.get_model_name("s3://manifest.json", "missing")

    def test_get_model_name_fallback(self, resolver, mock_router):
        """Возврат 'unknown', если ключа mlflow_model_name нет."""
        mock_router.download_manifest.return_value = {"decoder_pipeline": {}}

        name = resolver.get_model_name("s3://manifest.json", "decoder_pipeline")
        assert name == "unknown"

    def test_resolve_base_model_uri(self, resolver, mock_router):
        """Проверка резолвинга базового URI модели."""
        # 1. Отсечение префикса HuggingFace
        assert resolver._resolve_base_model_uri("hf://org/model", "cache") == "org/model"

        # 2. Скачивание из S3/Local
        mock_router.download_from_uri.return_value = Path("/downloaded/model")
        res = resolver._resolve_base_model_uri("s3://bucket/model", "cache")
        assert res == str(Path("/downloaded/model"))
        mock_router.download_from_uri.assert_called_once()

        # 3. Возврат как есть, если схема неизвестна (например локальный абсолютный путь)
        assert resolver._resolve_base_model_uri("/var/lib/model", "cache") == "/var/lib/model"


# ===========================================================================
# Тесты основного метода resolve_and_patch
# ===========================================================================


class TestResolveAndPatch:
    def test_missing_pipeline_raises_error(self, resolver, mock_router, base_cfg):
        """Если пайплайна нет в манифесте, код должен выбросить KeyError."""
        mock_router.download_manifest.return_value = {}
        with pytest.raises(KeyError, match="Пайплайн 'rag_pipeline' не найден"):
            resolver.resolve_and_patch(base_cfg, "uri", "rag_pipeline")

    def test_vector_db_server_persistent(self, resolver, mock_router, base_cfg):
        """Серверная БД (Qdrant) не скачивается, возвращается URI строкой."""
        mock_router.download_manifest.return_value = {
            "rag": {"vector_db_uri": "qdrant://localhost:6333/my_col"}
        }

        db_dir, lora_path, benchmark_dir = resolver.resolve_and_patch(base_cfg, "uri", "rag")

        assert db_dir == "qdrant://localhost:6333/my_col"
        mock_router.download_from_uri.assert_not_called()

    def test_vector_db_and_bm25_download(self, resolver, mock_router, base_cfg):
        """Файловая векторная БД и BM25 индекс скачиваются локально."""
        mock_router.download_manifest.return_value = {
            "rag": {
                "vector_db_uri": "s3://faiss_index",
                "bm25_uri": "s3://bm25_index",
                "benchmark_uri": "s3://benchmark",
            }
        }

        # Настраиваем возвращаемые значения для трех скачиваний
        mock_router.download_from_uri.side_effect = [
            Path("/cache/vector_db"),
            Path("/cache/bm25"),
            Path("/cache/benchmark"),
        ]

        db_dir, lora_path, benchmark_dir = resolver.resolve_and_patch(base_cfg, "uri", "rag")

        assert db_dir == Path("/cache/vector_db")
        assert benchmark_dir == Path("/cache/benchmark")
        assert base_cfg.inference.bm25.index_path == str(Path("/cache/bm25/bm25_index.pkl"))
        assert mock_router.download_from_uri.call_count == 3

    def test_load_type_lora(self, resolver, mock_router, base_cfg):
        """В режиме LoRA патчатся пути к базе и скачивается адаптер."""
        mock_router.download_manifest.return_value = {
            "train": {
                "load_type": "lora",
                "base_model_uri": "hf://meta-llama/Llama-3",
                "lora_uri": "s3://adapters/v1",
            }
        }
        mock_router.download_from_uri.return_value = Path("/cache/adapter")

        db_dir, lora_path, benchmark_dir = resolver.resolve_and_patch(base_cfg, "uri", "train")

        # Проверяем пути в конфиге
        assert base_cfg.model.builder.model_name_or_path == "meta-llama/Llama-3"
        assert base_cfg.model.tokenizer.tokenizer_name == "meta-llama/Llama-3"
        assert base_cfg.model.architecture.model_name_or_path == "meta-llama/Llama-3"

        # Проверяем возвращенный адаптер
        assert lora_path == Path("/cache/adapter")

    def test_load_type_full_model_inference(self, resolver, mock_router, base_cfg):
        """При инференсе монолита модель скачивается, а PEFT принудительно отключается."""
        mock_router.download_manifest.return_value = {
            "inference": {"load_type": "full_model", "model_uri": "s3://merged_models/v2"}
        }
        mock_router.download_from_uri.return_value = Path("/cache/merged_v2")

        db_dir, lora_path, benchmark_dir = resolver.resolve_and_patch(
            base_cfg, "uri", "inference", is_training=False
        )

        assert base_cfg.model.builder.model_name_or_path == str(Path("/cache/merged_v2"))
        assert base_cfg.model.modifiers.finetuning.skip_peft is True
        assert lora_path is None

    def test_load_type_full_model_training(self, resolver, mock_router, base_cfg):
        """При дообучении монолита PEFT остается активным (skip_peft не выставляется)."""
        mock_router.download_manifest.return_value = {
            "train": {"load_type": "full_model", "model_uri": "s3://merged_models/v3"}
        }
        mock_router.download_from_uri.return_value = Path("/cache/merged_v3")

        resolver.resolve_and_patch(base_cfg, "uri", "train", is_training=True)

        # Проверяем, что ключ skip_peft не был создан
        assert "skip_peft" not in base_cfg.model.modifiers.finetuning
