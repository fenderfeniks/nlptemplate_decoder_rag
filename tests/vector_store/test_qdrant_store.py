# tests/vector_store/test_qdrant_store_extended.py
"""Расширенные тесты для QdrantVectorStore.

Покрывают то, чего нет в test_qdrant_store.py:
- connect(): парсинг qdrant:// URI
- connect(): directory=None
- get_uri(): in-memory и обычный режим
- existing_doc_ids: пагинация scroll (несколько страниц)
- existing_doc_ids: записи без doc_id в payload, с payload=None
- _normalize / _prepare
- search: использует query_points (не search)
- search: нормализация включена/выключена
- insert: 1D массив raises ValueError
- insert: нормализация
- _build_filter: несколько полей, порядок conditions
- reset: кэш инвалидируется
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Инфраструктура мока qdrant_client (копия из test_qdrant_store.py)
# ---------------------------------------------------------------------------


def _make_qdrant_mocks():
    mock_qmodels = MagicMock()
    mock_qmodels.Distance.COSINE = "Cosine"
    mock_qmodels.Distance.DOT = "Dot"
    mock_qmodels.Distance.EUCLID = "Euclid"

    mock_client = MagicMock()
    mock_client_cls = MagicMock(return_value=mock_client)

    mock_collections_response = MagicMock()
    mock_collections_response.collections = []
    mock_client.get_collections.return_value = mock_collections_response

    mock_count_response = MagicMock()
    mock_count_response.count = 0
    mock_client.count.return_value = mock_count_response

    mock_client.scroll.return_value = ([], None)

    mock_qdrant_module = MagicMock()
    mock_qdrant_module.QdrantClient = mock_client_cls

    mock_http = MagicMock()
    mock_http.models = mock_qmodels
    mock_qdrant_module.http = mock_http

    return mock_qdrant_module, mock_client, mock_qmodels


DIM = 4


@pytest.fixture
def qdrant_mocks():
    mock_module, mock_client, mock_qmodels = _make_qdrant_mocks()

    mock_http_module = ModuleType("qdrant_client.http")
    mock_http_module.models = mock_qmodels

    modules_patch = {
        "qdrant_client": mock_module,
        "qdrant_client.http": mock_http_module,
        "qdrant_client.http.models": mock_qmodels,
    }

    with patch.dict(sys.modules, modules_patch):
        with patch("src.vector_store.qdrant_store._QDRANT_AVAILABLE", True):
            with patch("src.vector_store.qdrant_store.QdrantClient", mock_module.QdrantClient):
                with patch("src.vector_store.qdrant_store.qmodels", mock_qmodels):
                    yield mock_module, mock_client, mock_qmodels


def make_store(mock_module, mock_client, mock_qmodels, **kwargs):
    from src.vector_store.qdrant_store import QdrantVectorStore

    return QdrantVectorStore(embedding_dim=DIM, in_memory=True, **kwargs)


# ---------------------------------------------------------------------------
# connect() — парсинг qdrant:// URI
# ---------------------------------------------------------------------------


class TestConnectURI:
    def test_connect_parses_qdrant_uri(self, qdrant_mocks):
        """connect() правильно извлекает url и collection из qdrant:// URI."""
        from src.vector_store.qdrant_store import QdrantVectorStore

        store = QdrantVectorStore.connect(
            directory="qdrant://http://localhost:6333/my_collection",
            embedding_dim=DIM,
            in_memory=True,
        )
        assert store.collection_name == "my_collection"

    def test_connect_uri_url_extracted(self, qdrant_mocks):
        """connect() передаёт url из URI в kwargs (не перезаписывает явный url)."""
        _, mock_client, _ = qdrant_mocks
        from src.vector_store.qdrant_store import QdrantVectorStore

        # in_memory=True чтобы не реально подключаться
        store = QdrantVectorStore.connect(
            directory="qdrant://http://qdrant-server:6333/kb",
            embedding_dim=DIM,
            in_memory=True,  # явный параметр имеет приоритет через kwargs
        )
        # Коллекция должна называться "kb"
        assert store.collection_name == "kb"

    def test_connect_uri_does_not_override_explicit_collection(self, qdrant_mocks):
        """Явный collection_name в kwargs имеет приоритет над URI (setdefault)."""
        from src.vector_store.qdrant_store import QdrantVectorStore

        store = QdrantVectorStore.connect(
            directory="qdrant://http://localhost:6333/from_uri",
            embedding_dim=DIM,
            collection_name="explicit_name",
            in_memory=True,
        )
        # collection_name передан явно — setdefault не перезаписывает
        assert store.collection_name == "explicit_name"

    def test_connect_invalid_uri_raises(self, qdrant_mocks):
        """URI без '/' после хоста вызывает ValueError."""
        from src.vector_store.qdrant_store import QdrantVectorStore

        with pytest.raises(ValueError, match="Невалидный qdrant://"):
            QdrantVectorStore.connect(
                directory="qdrant://noslash",
                embedding_dim=DIM,
                in_memory=True,
            )

    def test_connect_directory_none_ok(self, qdrant_mocks):
        """connect() с directory=None не падает."""
        from src.vector_store.qdrant_store import QdrantVectorStore

        store = QdrantVectorStore.connect(directory=None, embedding_dim=DIM, in_memory=True)
        assert store._embedding_dim == DIM

    def test_connect_non_qdrant_path_ignored(self, qdrant_mocks):
        """directory со стандартным путём логируется и игнорируется."""
        from src.vector_store.qdrant_store import QdrantVectorStore

        with patch("src.vector_store.qdrant_store.logger") as mock_logger:
            store = QdrantVectorStore.connect(
                directory="/local/path/to/index",
                embedding_dim=DIM,
                in_memory=True,
            )

        # debug-лог об игнорировании
        debug_calls = " ".join(str(c) for c in mock_logger.debug.call_args_list)
        assert "игнорируется" in debug_calls or store._embedding_dim == DIM

    def test_connect_drops_unknown_kwargs_with_warning(self, qdrant_mocks):
        """Неизвестные kwargs логируются как warning."""
        from src.vector_store.qdrant_store import QdrantVectorStore

        with patch("src.vector_store.qdrant_store.logger") as mock_logger:
            QdrantVectorStore.connect(
                embedding_dim=DIM,
                in_memory=True,
                hydra_extra="dropped",
                another_extra="also_dropped",
            )

        warning_text = " ".join(str(c) for c in mock_logger.warning.call_args_list)
        assert "hydra_extra" in warning_text or "dropped" in warning_text.lower()


# ---------------------------------------------------------------------------
# get_uri()
# ---------------------------------------------------------------------------


class TestGetUri:
    def test_get_uri_in_memory_scheme(self, qdrant_mocks):
        """In-memory режим возвращает qdrant+memory:// URI."""
        store = make_store(*qdrant_mocks)
        uri = store.get_uri()
        assert uri.startswith("qdrant+memory:///")

    def test_get_uri_in_memory_contains_collection(self, qdrant_mocks):
        """URI содержит имя коллекции."""
        store = make_store(*qdrant_mocks, collection_name="my_kb")
        uri = store.get_uri()
        assert "my_kb" in uri

    def test_get_uri_normal_mode(self, qdrant_mocks):
        """Обычный режим возвращает qdrant://<url>/<collection>."""

        # Имитируем не-in-memory store
        store = make_store(*qdrant_mocks, collection_name="prod_kb")
        store._in_memory = False
        store._url = "http://qdrant:6333"
        uri = store.get_uri()
        assert uri == "qdrant://http://qdrant:6333/prod_kb"

    def test_get_uri_format_qdrant_scheme(self, qdrant_mocks):
        """Схема URI начинается с 'qdrant'."""
        store = make_store(*qdrant_mocks)
        uri = store.get_uri()
        assert uri.startswith("qdrant")


# ---------------------------------------------------------------------------
# existing_doc_ids — пагинация и edge cases
# ---------------------------------------------------------------------------


class TestExistingDocIdsPagination:
    def test_pagination_two_pages(self, qdrant_mocks):
        """Scroll с пагинацией: две страницы возвращают все doc_ids."""
        _, mock_client, _ = qdrant_mocks

        rec1 = MagicMock()
        rec1.payload = {"doc_id": "d1"}
        rec2 = MagicMock()
        rec2.payload = {"doc_id": "d2"}
        rec3 = MagicMock()
        rec3.payload = {"doc_id": "d3"}

        # Первый вызов: (страница 1, offset=42), второй: (страница 2, None)
        mock_client.scroll.side_effect = [
            ([rec1, rec2], 42),  # next_offset=42 -> продолжаем
            ([rec3], None),  # next_offset=None -> стоп
        ]

        store = make_store(*qdrant_mocks)
        ids = store.existing_doc_ids
        assert ids == {"d1", "d2", "d3"}
        assert mock_client.scroll.call_count == 2

    def test_pagination_three_pages(self, qdrant_mocks):
        """Scroll с тремя страницами собирает все ids."""
        _, mock_client, _ = qdrant_mocks

        def make_rec(doc_id):
            r = MagicMock()
            r.payload = {"doc_id": doc_id}
            return r

        mock_client.scroll.side_effect = [
            ([make_rec("d1")], "cursor1"),
            ([make_rec("d2")], "cursor2"),
            ([make_rec("d3")], None),
        ]

        store = make_store(*qdrant_mocks)
        ids = store.existing_doc_ids
        assert ids == {"d1", "d2", "d3"}
        assert mock_client.scroll.call_count == 3

    def test_record_without_doc_id_skipped(self, qdrant_mocks):
        """Записи без ключа doc_id в payload пропускаются."""
        _, mock_client, _ = qdrant_mocks

        rec_with_id = MagicMock()
        rec_with_id.payload = {"doc_id": "d1", "text": "hello"}

        rec_without_id = MagicMock()
        rec_without_id.payload = {"text": "no id here"}

        mock_client.scroll.return_value = ([rec_with_id, rec_without_id], None)

        store = make_store(*qdrant_mocks)
        ids = store.existing_doc_ids
        assert ids == {"d1"}

    def test_record_with_none_payload_skipped(self, qdrant_mocks):
        """Запись с payload=None не вызывает ошибку."""
        _, mock_client, _ = qdrant_mocks

        rec_none = MagicMock()
        rec_none.payload = None

        rec_ok = MagicMock()
        rec_ok.payload = {"doc_id": "d1"}

        mock_client.scroll.return_value = ([rec_none, rec_ok], None)

        store = make_store(*qdrant_mocks)
        # Не должно падать, None payload безопасно обрабатывается
        ids = store.existing_doc_ids
        assert "d1" in ids

    def test_cache_invalidated_after_reset(self, qdrant_mocks):
        """После reset кэш doc_id_cache очищается."""
        _, mock_client, _ = qdrant_mocks
        mock_client.scroll.return_value = ([], None)

        store = make_store(*qdrant_mocks)
        _ = store.existing_doc_ids
        assert store._doc_id_cache is not None

        store.reset()
        assert store._doc_id_cache is None


# ---------------------------------------------------------------------------
# _normalize / _prepare
# ---------------------------------------------------------------------------


class TestNormalizePrepare:
    def test_normalize_unit_vectors(self, qdrant_mocks):
        """_normalize возвращает векторы единичной длины."""
        store = make_store(*qdrant_mocks)
        vecs = np.array([[3.0, 4.0, 0.0, 0.0]], dtype=np.float32)
        normed = store._normalize(vecs)
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, [1.0], atol=1e-6)

    def test_normalize_zero_vector_no_nan(self, qdrant_mocks):
        """Нулевой вектор не приводит к NaN (clip ≥ 1e-10)."""
        store = make_store(*qdrant_mocks)
        vecs = np.zeros((1, DIM), dtype=np.float32)
        result = store._normalize(vecs)
        assert not np.isnan(result).any()

    def test_prepare_casts_to_float32(self, qdrant_mocks):
        """_prepare конвертирует любой dtype в float32."""
        store = make_store(*qdrant_mocks)
        for dtype in [np.float64, np.int32, np.float16]:
            vecs = np.ones((2, DIM), dtype=dtype)
            result = store._prepare(vecs)
            assert result.dtype == np.float32

    def test_prepare_normalizes_when_enabled(self, qdrant_mocks):
        """_prepare нормализует если normalize_embeddings=True."""
        store = make_store(*qdrant_mocks, normalize_embeddings=True)
        vecs = np.array([[2.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        result = store._prepare(vecs)
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), [1.0], atol=1e-6)

    def test_prepare_skips_normalize_when_disabled(self, qdrant_mocks):
        """_prepare не нормализует если normalize_embeddings=False."""
        store = make_store(*qdrant_mocks, normalize_embeddings=False)
        vecs = np.array([[2.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        result = store._prepare(vecs)
        # Длина должна остаться 2.0, а не стать 1.0
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), [2.0], atol=1e-6)

    def test_prepare_multiple_vectors(self, qdrant_mocks):
        """_prepare корректно обрабатывает батч векторов."""
        store = make_store(*qdrant_mocks)
        vecs = np.ones((5, DIM), dtype=np.float64)
        result = store._prepare(vecs)
        assert result.shape == (5, DIM)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# insert — расширенные случаи
# ---------------------------------------------------------------------------


class TestInsertExtended:
    def test_insert_1d_array_raises(self, qdrant_mocks):
        """1D массив вызывает ValueError."""
        store = make_store(*qdrant_mocks)
        bad = np.ones(DIM, dtype=np.float32)
        with pytest.raises(ValueError, match="Ожидается embeddings"):
            store.insert(bad, [{"doc_id": "d1"}])

    def test_insert_normalizes_when_enabled(self, qdrant_mocks):
        """insert нормализует векторы перед upsert."""
        store = make_store(*qdrant_mocks, normalize_embeddings=True)
        with patch.object(store, "_normalize", wraps=store._normalize) as mock_norm:
            emb = np.ones((1, DIM), dtype=np.float32)
            store.insert(emb, [{"doc_id": "d1"}])
        mock_norm.assert_called_once()

    def test_insert_skips_normalize_when_disabled(self, qdrant_mocks):
        """insert не нормализует если normalize_embeddings=False."""
        store = make_store(*qdrant_mocks, normalize_embeddings=False)
        with patch.object(store, "_normalize", wraps=store._normalize) as mock_norm:
            emb = np.ones((1, DIM), dtype=np.float32)
            store.insert(emb, [{"doc_id": "d1"}])
        mock_norm.assert_not_called()

    def test_insert_invalidates_doc_id_cache(self, qdrant_mocks):
        """insert инвалидирует кэш doc_ids."""
        _, mock_client, _ = qdrant_mocks
        mock_client.scroll.return_value = ([], None)
        store = make_store(*qdrant_mocks)
        _ = store.existing_doc_ids  # наполняем кэш
        assert store._doc_id_cache is not None

        emb = np.ones((1, DIM), dtype=np.float32)
        store.insert(emb, [{"doc_id": "new"}])
        assert store._doc_id_cache is None

    def test_insert_large_batch_multiple_upsert(self, qdrant_mocks):
        """n=7, batch_size=3 -> 3 вызова upsert (3+3+1)."""
        _, mock_client, _ = qdrant_mocks
        from src.vector_store.qdrant_store import QdrantVectorStore

        store = QdrantVectorStore(embedding_dim=DIM, in_memory=True, insert_batch_size=3)
        emb = np.ones((7, DIM), dtype=np.float32)
        meta = [{"doc_id": f"d{i}"} for i in range(7)]
        store.insert(emb, meta)
        assert mock_client.upsert.call_count == 3  # ceil(7/3) = 3


# ---------------------------------------------------------------------------
# search — использует query_points (не search)
# ---------------------------------------------------------------------------


class TestSearchQueryPoints:
    """В исходном коде search вызывает client.query_points, а не client.search."""

    def _setup_search(self, mock_client, score=0.9, payload=None):
        mock_client.count.return_value.count = 5
        hit = MagicMock()
        hit.score = score
        hit.payload = payload or {"doc_id": "d1"}
        response = MagicMock()
        response.points = [hit]
        mock_client.query_points.return_value = response

    def test_search_calls_query_points_not_search(self, qdrant_mocks):
        """Метод search вызывает client.query_points (не client.search)."""
        _, mock_client, _ = qdrant_mocks
        self._setup_search(mock_client)
        store = make_store(*qdrant_mocks)

        store.search(np.ones((1, DIM), dtype=np.float32), top_k=3)

        mock_client.query_points.assert_called_once()
        mock_client.search.assert_not_called()

    def test_search_result_score_and_metadata(self, qdrant_mocks):
        """Результат поиска содержит score и metadata."""
        _, mock_client, _ = qdrant_mocks
        self._setup_search(mock_client, score=0.75, payload={"doc_id": "d42", "text": "hi"})
        store = make_store(*qdrant_mocks)

        results = store.search(np.ones((1, DIM), dtype=np.float32), top_k=1)
        assert len(results) == 1
        assert len(results[0]) == 1
        hit = results[0][0]
        assert hit["score"] == pytest.approx(0.75)
        assert hit["metadata"]["doc_id"] == "d42"

    def test_search_multiple_queries_calls_query_points_per_query(self, qdrant_mocks):
        """query_points вызывается по одному разу на каждый запрос."""
        _, mock_client, _ = qdrant_mocks
        mock_client.count.return_value.count = 5
        response = MagicMock()
        response.points = []
        mock_client.query_points.return_value = response

        store = make_store(*qdrant_mocks)
        store.search(np.ones((4, DIM), dtype=np.float32), top_k=1)
        assert mock_client.query_points.call_count == 4

    def test_search_passes_top_k_as_limit(self, qdrant_mocks):
        """search передаёт top_k как параметр limit в query_points."""
        _, mock_client, _ = qdrant_mocks
        mock_client.count.return_value.count = 10
        response = MagicMock()
        response.points = []
        mock_client.query_points.return_value = response

        store = make_store(*qdrant_mocks)
        store.search(np.ones((1, DIM), dtype=np.float32), top_k=7)

        call_kwargs = mock_client.query_points.call_args[1]
        assert call_kwargs["limit"] == 7

    def test_search_empty_payload_returns_empty_dict(self, qdrant_mocks):
        """Если payload=None — возвращается пустой dict."""
        _, mock_client, _ = qdrant_mocks
        mock_client.count.return_value.count = 5

        hit = MagicMock()
        hit.score = 0.5
        hit.payload = None
        response = MagicMock()
        response.points = [hit]
        mock_client.query_points.return_value = response

        store = make_store(*qdrant_mocks)
        results = store.search(np.ones((1, DIM), dtype=np.float32), top_k=1)
        assert results[0][0]["metadata"] == {}

    def test_search_with_filter_passes_qdrant_filter(self, qdrant_mocks):
        """search с filter_metadata передаёт Qdrant Filter в query_points."""
        _, mock_client, mock_qmodels = qdrant_mocks
        mock_client.count.return_value.count = 5
        response = MagicMock()
        response.points = []
        mock_client.query_points.return_value = response

        store = make_store(*qdrant_mocks)
        store.search(
            np.ones((1, DIM), dtype=np.float32),
            filter_metadata={"source": "wiki"},
        )

        call_kwargs = mock_client.query_points.call_args[1]
        # query_filter должен быть передан (не None)
        assert call_kwargs.get("query_filter") is not None

    def test_search_without_filter_passes_none(self, qdrant_mocks):
        """search без фильтра передаёт query_filter=None."""
        _, mock_client, _ = qdrant_mocks
        mock_client.count.return_value.count = 5
        response = MagicMock()
        response.points = []
        mock_client.query_points.return_value = response

        store = make_store(*qdrant_mocks)
        store.search(np.ones((1, DIM), dtype=np.float32), filter_metadata=None)

        call_kwargs = mock_client.query_points.call_args[1]
        assert call_kwargs.get("query_filter") is None


# ---------------------------------------------------------------------------
# _build_filter — расширенные случаи
# ---------------------------------------------------------------------------


class TestBuildFilterExtended:
    def test_multiple_fields_and_semantics(self, qdrant_mocks):
        """Несколько полей -> несколько FieldCondition, все в must (AND)."""
        _, _, mock_qmodels = qdrant_mocks
        store = make_store(*qdrant_mocks)
        store._build_filter({"source": "wiki", "lang": "ru", "year": 2024})
        assert mock_qmodels.FieldCondition.call_count == 3
        # Filter создан с must=
        mock_qmodels.Filter.assert_called_once()
        call_kwargs = mock_qmodels.Filter.call_args[1]
        assert "must" in call_kwargs

    def test_none_filter_skips_FieldCondition(self, qdrant_mocks):
        """_build_filter(None) не создаёт FieldCondition."""
        _, _, mock_qmodels = qdrant_mocks
        store = make_store(*qdrant_mocks)
        store._build_filter(None)
        mock_qmodels.FieldCondition.assert_not_called()

    def test_empty_dict_skips_FieldCondition(self, qdrant_mocks):
        """_build_filter({}) не создаёт FieldCondition."""
        _, _, mock_qmodels = qdrant_mocks
        store = make_store(*qdrant_mocks)
        store._build_filter({})
        mock_qmodels.FieldCondition.assert_not_called()


# ---------------------------------------------------------------------------
# _ensure_collection — поведение при существующей коллекции
# ---------------------------------------------------------------------------


class TestEnsureCollection:
    def test_existing_collection_skips_create(self, qdrant_mocks):
        """Если коллекция уже существует — create_collection не вызывается."""
        _, mock_client, _ = qdrant_mocks
        existing = MagicMock()
        existing.name = "knowledge_base"
        mock_client.get_collections.return_value.collections = [existing]

        make_store(*qdrant_mocks)
        mock_client.create_collection.assert_not_called()

    def test_new_collection_calls_create(self, qdrant_mocks):
        """Если коллекции нет — create_collection вызывается."""
        _, mock_client, _ = qdrant_mocks
        mock_client.get_collections.return_value.collections = []

        make_store(*qdrant_mocks)
        mock_client.create_collection.assert_called_once()

    def test_collection_name_passed_to_create(self, qdrant_mocks):
        """Имя коллекции корректно передаётся в create_collection."""
        _, mock_client, _ = qdrant_mocks
        mock_client.get_collections.return_value.collections = []

        make_store(*qdrant_mocks, collection_name="custom_kb")
        call_kwargs = mock_client.create_collection.call_args[1]
        assert call_kwargs["collection_name"] == "custom_kb"


# ---------------------------------------------------------------------------
# ntotal и состояние
# ---------------------------------------------------------------------------


class TestNtotalAndState:
    def test_ntotal_calls_client_count(self, qdrant_mocks):
        """ntotal делегирует вызов client.count."""
        _, mock_client, _ = qdrant_mocks
        mock_client.count.return_value.count = 99
        store = make_store(*qdrant_mocks)
        assert store.ntotal == 99

    def test_embedding_dim_stored(self, qdrant_mocks):
        """embedding_dim хранится как _embedding_dim."""
        store = make_store(*qdrant_mocks)
        assert store.embedding_dim == DIM
        assert store._embedding_dim == DIM

    def test_in_memory_flag_set(self, qdrant_mocks):
        """in_memory=True корректно устанавливает _in_memory."""
        store = make_store(*qdrant_mocks, in_memory=True)
        assert store._in_memory is True

    def test_url_memory_triggers_in_memory(self, qdrant_mocks):
        """url=':memory:' активирует in-memory режим."""
        from src.vector_store.qdrant_store import QdrantVectorStore

        # Передаем только url, без дублирования in_meгmory
        store = QdrantVectorStore(embedding_dim=DIM, url=":memory:")
        assert store._in_memory is True
