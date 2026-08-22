from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.tools.benchmark.loader import BenchmarkLoader


# ===========================================================================
# Фикстуры
# ===========================================================================


@pytest.fixture
def mock_router():
    return MagicMock()


@pytest.fixture
def loader(mock_router, tmp_path):
    return BenchmarkLoader(
        router=mock_router,
        cache_dir=tmp_path / "cache",
        manifest_uri="s3://manifest.json",
        pipeline_name="qa_pipeline",
    )


@pytest.fixture
def mock_hf_dataset(mocker):
    """Мокает тяжелый импорт HFDataset."""
    return mocker.patch("src.tools.benchmark.loader.HFDataset")


# ===========================================================================
# Тесты загрузки манифеста и валидации кэша
# ===========================================================================


class TestBenchmarkLoaderInternals:
    def test_load_manifest_success(self, loader, mock_router):
        """Успешное извлечение секции нужного пайплайна."""
        mock_router.download_manifest.return_value = {
            "qa_pipeline": {"benchmark_uri": "s3://b1"},
            "other": {"benchmark_uri": "s3://b2"},
        }
        manifest = loader._load_manifest()
        assert manifest == {"benchmark_uri": "s3://b1"}

    def test_load_manifest_exception_returns_empty(self, loader, mock_router):
        """Если роутер падает, возвращается пустой словарь (согласно текущей логике)."""
        mock_router.download_manifest.side_effect = Exception("Network Error")
        assert loader._load_manifest() == {}

    def test_cache_is_valid_missing_file(self, loader):
        """Кэш не валиден, если файла нет."""
        assert loader._cache_is_valid(expected_size=10) is False

    def test_cache_is_valid_no_expected_size(self, loader):
        """Кэш всегда валиден, если expected_size = None и файл существует."""
        loader.cache_dir.mkdir(parents=True, exist_ok=True)
        loader._local_path().touch()

        assert loader._cache_is_valid(expected_size=None) is True

    def test_cache_is_valid_size_match_and_mismatch(self, loader):
        """Сверка количества строк в кэше с expected_size."""
        loader.cache_dir.mkdir(parents=True, exist_ok=True)
        local_file = loader._local_path()

        # Создаем файл с 2 непустыми строками и 1 пустой
        with open(local_file, "w", encoding="utf-8") as f:
            f.write("line1\nline2\n  \n")

        assert loader._cache_is_valid(expected_size=2) is True
        assert loader._cache_is_valid(expected_size=3) is False


# ===========================================================================
# Тесты бизнес-логики (резолвинг и загрузка)
# ===========================================================================


class TestBenchmarkLoaderResolving:
    def test_resolve_local_path_no_uri(self, loader, mock_router):
        """Если в манифесте нет benchmark_uri, возвращаем None (BenchmarkExclusion отключен)."""
        mock_router.download_manifest.return_value = {"qa_pipeline": {}}
        assert loader.resolve_local_path() is None

    def test_resolve_local_path_cache_hit(self, loader, mock_router, mocker):
        """Если кэш валиден, скачивание не запускается."""
        mock_router.download_manifest.return_value = {
            "qa_pipeline": {"benchmark_uri": "s3://b1", "benchmark_size": 100}
        }
        mocker.patch.object(loader, "_cache_is_valid", return_value=True)

        path = loader.resolve_local_path()

        assert path == loader._local_path()
        mock_router.download_file_from_uri.assert_not_called()

    def test_resolve_local_path_cache_miss(self, loader, mock_router, mocker):
        """Если кэш невалиден (или отсутствует), запускается скачивание."""
        mock_router.download_manifest.return_value = {
            "qa_pipeline": {"benchmark_uri": "s3://b1", "benchmark_size": 100}
        }
        mocker.patch.object(loader, "_cache_is_valid", return_value=False)
        mock_router.download_file_from_uri.return_value = Path("/downloaded/bench.jsonl")

        path = loader.resolve_local_path()

        assert path == Path("/downloaded/bench.jsonl")
        mock_router.download_file_from_uri.assert_called_once_with(
            uri="s3://b1", local_path=loader._local_path()
        )

    def test_load_as_dataset_success(self, loader, mocker, mock_hf_dataset):
        """Проверка парсинга JSONL и создания HF Dataset."""
        loader.cache_dir.mkdir(parents=True, exist_ok=True)
        local_file = loader._local_path()

        # Пишем один нормальный JSON, один битый, одну пустую строку
        with open(local_file, "w", encoding="utf-8") as f:
            f.write('{"q": 1}\n\n{broken_json\n{"q": 2}\n')

        mocker.patch.object(loader, "resolve_local_path", return_value=local_file)

        result = loader.load_as_dataset()

        # Убеждаемся, что парсер проигнорировал битый JSON и передал в HFDataset два валидных объекта
        mock_hf_dataset.from_list.assert_called_once_with([{"q": 1}, {"q": 2}])
        assert result == mock_hf_dataset.from_list.return_value

    def test_load_as_dataset_empty_file_returns_none(self, loader, mocker):
        """Если файл пуст или состоит только из битых строк, возвращается None."""
        loader.cache_dir.mkdir(parents=True, exist_ok=True)
        local_file = loader._local_path()
        local_file.touch()

        mocker.patch.object(loader, "resolve_local_path", return_value=local_file)

        assert loader.load_as_dataset() is None
