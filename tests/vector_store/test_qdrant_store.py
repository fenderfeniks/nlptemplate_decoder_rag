# tests/vector_store/test_qdrant_store.py
"""Тесты для QdrantVectorStore.

qdrant_client не устанавливается — мокируем целиком через patch.dict(sys.modules).
Все тесты работают без запущенного Qdrant-сервера.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Инфраструктура мока qdrant_client
# ---------------------------------------------------------------------------


def _make_qdrant_mocks():
    """Возвращает (mock_module, mock_client_instance, mock_qmodels)."""
    mock_qmodels = MagicMock()

    # Distance enum-заглушки
    mock_qmodels.Distance.COSINE = "Cosine"
    mock_qmodels.Distance.DOT = "Dot"
    mock_qmodels.Distance.EUCLID = "Euclid"

    mock_client = MagicMock()
    mock_client_cls = MagicMock(return_value=mock_client)

    # get_collections → пустой список по умолчанию
    mock_collections_response = MagicMock()
    mock_collections_response.collections = []
    mock_client.get_collections.return_value = mock_collections_response

    # count → 0 по умолчанию
    mock_count_response = MagicMock()
    mock_count_response.count = 0
    mock_client.count.return_value = mock_count_response

    # scroll → пустой по умолчанию
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
    """Патчит sys.modules для qdrant_client и возвращает (module, client, qmodels)."""
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
    """Создаёт QdrantVectorStore в контексте патча."""
    from src.vector_store.qdrant_store import QdrantVectorStore

    return QdrantVectorStore(
        embedding_dim=DIM,
        in_memory=True,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# __init__ / _ensure_collection
# ---------------------------------------------------------------------------


class TestInit:
    def test_client_created(self, qdrant_mocks):
        _, mock_client, mock_qmodels = qdrant_mocks
        store = make_store(*qdrant_mocks)
        # Клиент создан
        assert store._embedding_dim == DIM

    def test_collection_created_when_not_exists(self, qdrant_mocks):
        _, mock_client, _ = qdrant_mocks
        mock_client.create_collection.assert_called_once()

    def test_collection_not_recreated_when_exists(self, qdrant_mocks):
        _, mock_client, _ = qdrant_mocks
        # Коллекция уже существует
        existing = MagicMock()
        existing.name = "knowledge_base"
        mock_client.get_collections.return_value.collections = [existing]
        mock_client.create_collection.assert_not_called()

    def test_invalid_distance_raises(self, qdrant_mocks):
        from src.vector_store.qdrant_store import QdrantVectorStore

        with pytest.raises(ValueError, match="Неизвестная метрика"):
            QdrantVectorStore(embedding_dim=DIM, in_memory=True, distance="L2_bad")

    def test_recreate_deletes_and_recreates(self, qdrant_mocks):
        _, mock_client, _ = qdrant_mocks
        existing = MagicMock()
        existing.name = "knowledge_base"
        mock_client.get_collections.return_value.collections = [existing]

        from src.vector_store.qdrant_store import QdrantVectorStore

        QdrantVectorStore(embedding_dim=DIM, in_memory=True, recreate_collection=True)
        mock_client.delete_collection.assert_called_once_with("knowledge_base")
        mock_client.create_collection.assert_called_once()

    def test_import_error_without_qdrant(self):
        with patch("src.vector_store.qdrant_store._QDRANT_AVAILABLE", False):
            from src.vector_store.qdrant_store import QdrantVectorStore

            with pytest.raises(ImportError, match="qdrant-client"):
                QdrantVectorStore(embedding_dim=DIM)


# ---------------------------------------------------------------------------
# Свойства состояния
# ---------------------------------------------------------------------------


class TestProperties:
    def test_embedding_dim(self, qdrant_mocks):
        store = make_store(*qdrant_mocks)
        assert store.embedding_dim == DIM

    def test_ntotal_delegates_to_client(self, qdrant_mocks):
        _, mock_client, _ = qdrant_mocks
        mock_client.count.return_value.count = 42
        store = make_store(*qdrant_mocks)
        assert store.ntotal == 42

    def test_existing_doc_ids_empty(self, qdrant_mocks):
        _, mock_client, _ = qdrant_mocks
        mock_client.scroll.return_value = ([], None)
        store = make_store(*qdrant_mocks)
        assert store.existing_doc_ids == set()

    def test_existing_doc_ids_from_scroll(self, qdrant_mocks):
        _, mock_client, _ = qdrant_mocks

        rec1 = MagicMock()
        rec1.payload = {"doc_id": "d1"}
        rec2 = MagicMock()
        rec2.payload = {"doc_id": "d2"}
        mock_client.scroll.return_value = ([rec1, rec2], None)

        store = make_store(*qdrant_mocks)
        assert store.existing_doc_ids == {"d1", "d2"}

    def test_existing_doc_ids_cached(self, qdrant_mocks):
        _, mock_client, _ = qdrant_mocks
        mock_client.scroll.return_value = ([], None)

        store = make_store(*qdrant_mocks)
        _ = store.existing_doc_ids
        _ = store.existing_doc_ids
        # scroll вызывался только один раз (кэш)
        assert mock_client.scroll.call_count == 1

    def test_existing_doc_ids_cache_invalidated_after_insert(self, qdrant_mocks):
        _, mock_client, _ = qdrant_mocks
        mock_client.scroll.return_value = ([], None)

        store = make_store(*qdrant_mocks)
        _ = store.existing_doc_ids
        assert store._doc_id_cache is not None

        emb = np.ones((1, DIM), dtype=np.float32)
        store.insert(emb, [{"doc_id": "d1"}])
        assert store._doc_id_cache is None


# ---------------------------------------------------------------------------
# insert
# ---------------------------------------------------------------------------


class TestInsert:
    def test_insert_calls_upsert(self, qdrant_mocks):
        _, mock_client, _ = qdrant_mocks
        store = make_store(*qdrant_mocks)
        emb = np.ones((2, DIM), dtype=np.float32)
        store.insert(emb, [{"doc_id": "d1"}, {"doc_id": "d2"}])
        mock_client.upsert.assert_called_once()

    def test_insert_wrong_dim_raises(self, qdrant_mocks):
        store = make_store(*qdrant_mocks)
        bad = np.ones((2, DIM + 1), dtype=np.float32)
        with pytest.raises(ValueError, match="Ожидается embeddings"):
            store.insert(bad, [{}, {}])

    def test_insert_length_mismatch_raises(self, qdrant_mocks):
        store = make_store(*qdrant_mocks)
        emb = np.ones((3, DIM), dtype=np.float32)
        with pytest.raises(ValueError, match="Несоответствие длин"):
            store.insert(emb, [{}])

    def test_insert_batching(self, qdrant_mocks):
        """Вставка батчами: 5 векторов с batch_size=2 → 3 вызова upsert."""
        _, mock_client, _ = qdrant_mocks
        from src.vector_store.qdrant_store import QdrantVectorStore

        store = QdrantVectorStore(embedding_dim=DIM, in_memory=True, insert_batch_size=2)
        emb = np.ones((5, DIM), dtype=np.float32)
        meta = [{"doc_id": f"d{i}"} for i in range(5)]
        store.insert(emb, meta)
        assert mock_client.upsert.call_count == 3

    def test_insert_points_have_unique_ids(self, qdrant_mocks):
        """Каждая точка получает уникальный UUID."""
        _, mock_client, mock_qmodels = qdrant_mocks
        store = make_store(*qdrant_mocks)
        emb = np.ones((2, DIM), dtype=np.float32)
        store.insert(emb, [{"doc_id": "d1"}, {"doc_id": "d2"}])

        # Извлекаем id из call_args_list конструктора PointStruct
        ids = [c.kwargs["id"] for c in mock_qmodels.PointStruct.call_args_list]
        assert len(ids) == 2
        assert len(set(ids)) == 2


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_empty_store_returns_empty_lists(self, qdrant_mocks):
        _, mock_client, _ = qdrant_mocks
        mock_client.count.return_value.count = 0
        store = make_store(*qdrant_mocks)
        result = store.search(np.ones((1, DIM), dtype=np.float32))
        assert result == [[]]

    def test_search_calls_client_search(self, qdrant_mocks):
        _, mock_client, _ = qdrant_mocks
        mock_client.count.return_value.count = 5

        hit = MagicMock()
        hit.score = 0.9
        hit.payload = {"doc_id": "d1"}
        mock_client.search.return_value = [hit]

        store = make_store(*qdrant_mocks)
        result = store.search(np.ones((1, DIM), dtype=np.float32), top_k=3)

        mock_client.search.assert_called_once()
        assert len(result) == 1
        assert result[0][0]["score"] == pytest.approx(0.9)
        assert result[0][0]["metadata"] == {"doc_id": "d1"}

    def test_search_multiple_queries(self, qdrant_mocks):
        _, mock_client, _ = qdrant_mocks
        mock_client.count.return_value.count = 5

        hit = MagicMock()
        hit.score = 0.8
        hit.payload = {}
        mock_client.search.return_value = [hit]

        store = make_store(*qdrant_mocks)
        queries = np.ones((3, DIM), dtype=np.float32)
        result = store.search(queries, top_k=1)

        assert len(result) == 3
        assert mock_client.search.call_count == 3

    def test_search_with_filter_builds_qdrant_filter(self, qdrant_mocks):
        _, mock_client, mock_qmodels = qdrant_mocks
        mock_client.count.return_value.count = 5
        mock_client.search.return_value = []

        store = make_store(*qdrant_mocks)
        store.search(
            np.ones((1, DIM), dtype=np.float32),
            filter_metadata={"source": "wiki"},
        )

        # _build_filter должен создать FieldCondition
        mock_qmodels.FieldCondition.assert_called_once_with(
            key="source",
            match=mock_qmodels.MatchValue(value="wiki"),
        )

    def test_search_none_filter_passes_none_to_client(self, qdrant_mocks):
        _, mock_client, _ = qdrant_mocks
        mock_client.count.return_value.count = 5
        mock_client.search.return_value = []

        store = make_store(*qdrant_mocks)
        store.search(np.ones((1, DIM), dtype=np.float32), filter_metadata=None)

        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs["query_filter"] is None


# ---------------------------------------------------------------------------
# save / reset
# ---------------------------------------------------------------------------


class TestSaveReset:
    def test_save_is_noop(self, qdrant_mocks):
        """save() не должен вызывать никаких клиентских методов."""
        _, mock_client, _ = qdrant_mocks
        store = make_store(*qdrant_mocks)
        store.save("/some/dir")
        mock_client.upsert.assert_not_called()

    def test_reset_deletes_and_recreates_collection(self, qdrant_mocks):
        _, mock_client, _ = qdrant_mocks
        store = make_store(*qdrant_mocks)
        store.reset()
        mock_client.delete_collection.assert_called_once_with("knowledge_base")
        # create_collection: один раз при __init__, один раз после reset
        assert mock_client.create_collection.call_count == 2

    def test_reset_invalidates_cache(self, qdrant_mocks):
        _, mock_client, _ = qdrant_mocks
        mock_client.scroll.return_value = ([], None)
        store = make_store(*qdrant_mocks)
        _ = store.existing_doc_ids
        store.reset()
        assert store._doc_id_cache is None


# ---------------------------------------------------------------------------
# connect (фабричный метод)
# ---------------------------------------------------------------------------


class TestConnect:
    def test_connect_filters_unknown_kwargs(self, qdrant_mocks):
        """Лишние kwargs из Hydra не вызывают TypeError."""
        from src.vector_store.qdrant_store import QdrantVectorStore

        # Должен пройти без ошибки
        store = QdrantVectorStore.connect(
            directory="/ignored",
            embedding_dim=DIM,
            in_memory=True,
            unknown_hydra_key="should_be_dropped",
        )
        assert store is not None

    def test_connect_directory_ignored(self, qdrant_mocks):
        """directory не вызывает ошибку и игнорируется."""
        from src.vector_store.qdrant_store import QdrantVectorStore

        store = QdrantVectorStore.connect(
            directory="/tmp/whatever",
            embedding_dim=DIM,
            in_memory=True,
        )
        assert store._embedding_dim == DIM

    def test_connect_logs_warning_when_empty(self, qdrant_mocks):
        """При пустой коллекции логируется warning."""
        _, mock_client, _ = qdrant_mocks
        mock_client.count.return_value.count = 0

        from src.vector_store.qdrant_store import QdrantVectorStore

        with patch("src.vector_store.qdrant_store.logger") as mock_logger:
            QdrantVectorStore.connect(embedding_dim=DIM, in_memory=True)

        warning_args = " ".join(str(c) for c in mock_logger.warning.call_args_list)
        assert (
            "пуст" in warning_args or "empty" in warning_args.lower() or mock_logger.warning.called
        )


# ---------------------------------------------------------------------------
# _build_filter
# ---------------------------------------------------------------------------


class TestBuildFilter:
    def test_none_returns_none(self, qdrant_mocks):
        store = make_store(*qdrant_mocks)
        assert store._build_filter(None) is None

    def test_empty_dict_returns_none(self, qdrant_mocks):
        store = make_store(*qdrant_mocks)
        assert store._build_filter({}) is None

    def test_single_field_builds_filter(self, qdrant_mocks):
        _, _, mock_qmodels = qdrant_mocks
        store = make_store(*qdrant_mocks)
        store._build_filter({"source": "wiki"})
        mock_qmodels.FieldCondition.assert_called_once()
        mock_qmodels.Filter.assert_called_once()

    def test_multiple_fields_multiple_conditions(self, qdrant_mocks):
        _, _, mock_qmodels = qdrant_mocks
        store = make_store(*qdrant_mocks)
        store._build_filter({"source": "wiki", "lang": "ru"})
        assert mock_qmodels.FieldCondition.call_count == 2
