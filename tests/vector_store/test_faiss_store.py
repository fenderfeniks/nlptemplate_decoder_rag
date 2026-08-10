# tests/vector_store/test_faiss_store.py
"""Тесты для FAISSVectorStore.

faiss и numpy — реальные зависимости (они нужны в окружении).
Тесты используют небольшие векторы (dim=4) чтобы быть быстрыми.
Тяжёлые операции IO мокируются там где нужно.
"""

from __future__ import annotations

import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


faiss = pytest.importorskip("faiss", reason="faiss не установлен")

from src.vector_store.faiss_store import FAISSVectorStore  # noqa


DIM = 4


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------


@pytest.fixture
def store():
    return FAISSVectorStore(embedding_dim=DIM, index_type="flat")


@pytest.fixture
def embeddings():
    rng = np.random.default_rng(42)
    return rng.random((3, DIM)).astype(np.float32)


@pytest.fixture
def metadata():
    return [
        {"doc_id": "d1", "source": "wiki"},
        {"doc_id": "d2", "source": "arxiv"},
        {"doc_id": "d3", "source": "wiki"},
    ]


def make_store(**kwargs) -> FAISSVectorStore:
    return FAISSVectorStore(embedding_dim=DIM, **kwargs)


# ---------------------------------------------------------------------------
# __init__ / _build_index
# ---------------------------------------------------------------------------


class TestInit:
    def test_flat_index_created(self, store):
        assert store.index_type == "flat"
        assert store.embedding_dim == DIM
        assert store.ntotal == 0

    def test_hnsw_index_created(self):
        s = make_store(index_type="hnsw")
        assert s.index_type == "hnsw"
        assert s.ntotal == 0

    def test_invalid_index_type_raises(self):
        with pytest.raises(ValueError, match="Неизвестный тип индекса"):
            make_store(index_type="ivf")

    def test_metadata_empty_on_init(self, store):
        assert store._metadata == []

    def test_doc_id_cache_none_on_init(self, store):
        assert store._doc_id_cache is None

    def test_index_type_case_insensitive(self):
        s = FAISSVectorStore(embedding_dim=DIM, index_type="FLAT")
        assert s.index_type == "flat"


# ---------------------------------------------------------------------------
# insert
# ---------------------------------------------------------------------------


class TestInsert:
    def test_insert_adds_vectors(self, store, embeddings, metadata):
        store.insert(embeddings, metadata)
        assert store.ntotal == 3
        assert len(store._metadata) == 3

    def test_insert_wrong_dim_raises(self, store, metadata):
        bad = np.ones((3, DIM + 1), dtype=np.float32)
        with pytest.raises(ValueError, match="Ожидается embeddings"):
            store.insert(bad, metadata[:3])

    def test_insert_1d_raises(self, store, metadata):
        bad = np.ones(DIM, dtype=np.float32)
        with pytest.raises(ValueError, match="Ожидается embeddings"):
            store.insert(bad, metadata[:1])

    def test_insert_length_mismatch_raises(self, store, embeddings):
        with pytest.raises(ValueError, match="Несоответствие длин"):
            store.insert(embeddings, [{"doc_id": "only_one"}])

    def test_insert_invalidates_cache(self, store, embeddings, metadata):
        # Наполняем кэш
        _ = store.existing_doc_ids
        assert store._doc_id_cache is not None
        store.insert(embeddings, metadata)
        assert store._doc_id_cache is None

    def test_insert_atomic_rollback_on_index_error(self, store, embeddings, metadata):
        """При ошибке index.add метаданные откатываются."""
        store.index.add = MagicMock(side_effect=RuntimeError("faiss boom"))
        with pytest.raises(RuntimeError, match="faiss boom"):
            store.insert(embeddings, metadata)
        assert store.ntotal == 0
        assert store._metadata == []

    def test_insert_normalizes_vectors(self, embeddings, metadata):
        s = make_store(normalize_embeddings=True)
        # Патчим _normalize чтобы убедиться что он вызывается
        with patch.object(s, "_normalize", wraps=s._normalize) as mock_norm:
            s.insert(embeddings, metadata)
        mock_norm.assert_called_once()

    def test_insert_skip_normalize_when_disabled(self, embeddings, metadata):
        s = make_store(normalize_embeddings=False)
        with patch.object(s, "_normalize", wraps=s._normalize) as mock_norm:
            s.insert(embeddings, metadata)
        mock_norm.assert_not_called()

    def test_insert_accumulates_multiple_calls(self, store, embeddings, metadata):
        store.insert(embeddings, metadata)
        store.insert(embeddings, metadata)
        assert store.ntotal == 6
        assert len(store._metadata) == 6


# ---------------------------------------------------------------------------
# insert_batched
# ---------------------------------------------------------------------------


class TestInsertBatched:
    def test_batched_inserts_all(self, embeddings, metadata):
        s = FAISSVectorStore(embedding_dim=DIM, insert_batch_size=2)
        s.insert_batched(embeddings, metadata)
        assert s.ntotal == 3

    def test_batched_single_batch(self, store, embeddings, metadata):
        store.insert_batched(embeddings, metadata, desc="Test")
        assert store.ntotal == 3


# ---------------------------------------------------------------------------
# existing_doc_ids
# ---------------------------------------------------------------------------


class TestExistingDocIds:
    def test_empty_store_returns_empty_set(self, store):
        assert store.existing_doc_ids == set()

    def test_returns_doc_ids_after_insert(self, store, embeddings, metadata):
        store.insert(embeddings, metadata)
        assert store.existing_doc_ids == {"d1", "d2", "d3"}

    def test_cache_populated_after_first_access(self, store, embeddings, metadata):
        store.insert(embeddings, metadata)
        assert store._doc_id_cache is None
        _ = store.existing_doc_ids
        assert store._doc_id_cache is not None

    def test_cache_reused_on_second_access(self, store, embeddings, metadata):
        store.insert(embeddings, metadata)
        ids1 = store.existing_doc_ids
        ids2 = store.existing_doc_ids
        assert ids1 is ids2  # тот же объект

    def test_metadata_without_doc_id_skipped(self, store, embeddings):
        meta = [{"text": "no id"}, {"doc_id": "d2", "x": 1}, {"text": "also no id"}]
        store.insert(embeddings, meta)
        assert store.existing_doc_ids == {"d2"}


# ---------------------------------------------------------------------------
# search — без фильтра
# ---------------------------------------------------------------------------


class TestSearchNoFilter:
    def test_empty_index_returns_empty_lists(self, store, embeddings):
        result = store.search(embeddings[:1])
        assert result == [[]]

    def test_search_returns_top_k(self, store, embeddings, metadata):
        store.insert(embeddings, metadata)
        results = store.search(embeddings[:1], top_k=2)
        assert len(results) == 1
        assert len(results[0]) == 2

    def test_search_result_structure(self, store, embeddings, metadata):
        store.insert(embeddings, metadata)
        results = store.search(embeddings[:1], top_k=1)
        hit = results[0][0]
        assert "score" in hit
        assert "metadata" in hit
        assert isinstance(hit["score"], float)
        assert isinstance(hit["metadata"], dict)

    def test_search_sorted_by_score_descending(self, store, embeddings, metadata):
        store.insert(embeddings, metadata)
        results = store.search(embeddings[:1], top_k=3)
        scores = [h["score"] for h in results[0]]
        assert scores == sorted(scores, reverse=True)

    def test_search_multiple_queries(self, store, embeddings, metadata):
        store.insert(embeddings, metadata)
        results = store.search(embeddings, top_k=1)
        assert len(results) == 3

    def test_top_k_clamped_to_ntotal(self, store, embeddings, metadata):
        """top_k > ntotal не вызывает ошибку."""
        store.insert(embeddings, metadata)
        results = store.search(embeddings[:1], top_k=100)
        assert len(results[0]) == 3  # ntotal=3


# ---------------------------------------------------------------------------
# search — с фильтром
# ---------------------------------------------------------------------------


class TestSearchWithFilter:
    def test_filter_by_source(self, store, embeddings, metadata):
        store.insert(embeddings, metadata)
        results = store.search(embeddings[:1], top_k=5, filter_metadata={"source": "wiki"})
        # Все результаты должны иметь source=wiki
        for hit in results[0]:
            assert hit["metadata"]["source"] == "wiki"

    def test_filter_no_match_returns_empty(self, store, embeddings, metadata):
        store.insert(embeddings, metadata)
        results = store.search(embeddings[:1], top_k=5, filter_metadata={"source": "nonexistent"})
        assert results[0] == []

    def test_filter_none_behaves_as_no_filter(self, store, embeddings, metadata):
        store.insert(embeddings, metadata)
        r_none = store.search(embeddings[:1], top_k=3, filter_metadata=None)
        r_no = store.search(embeddings[:1], top_k=3)
        assert len(r_none[0]) == len(r_no[0])

    def test_filter_empty_dict_behaves_as_no_filter(self, store, embeddings, metadata):
        store.insert(embeddings, metadata)
        results = store.search(embeddings[:1], top_k=3, filter_metadata={})
        assert len(results[0]) == 3


# ---------------------------------------------------------------------------
# _normalize / _prepare
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_normalize_produces_unit_vectors(self, store):
        vecs = np.array([[3.0, 4.0, 0.0, 0.0]], dtype=np.float32)
        normed = store._normalize(vecs)
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, [1.0], atol=1e-6)

    def test_normalize_zero_vector_no_nan(self, store):
        """Нулевой вектор → не NaN (clip предотвращает деление на 0)."""
        vecs = np.zeros((1, DIM), dtype=np.float32)
        result = store._normalize(vecs)
        assert not np.isnan(result).any()

    def test_prepare_casts_to_float32(self, store):
        vecs = np.ones((2, DIM), dtype=np.float64)
        result = store._prepare(vecs)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# _check_consistency
# ---------------------------------------------------------------------------


class TestCheckConsistency:
    def test_consistent_state_does_not_raise(self, store, embeddings, metadata):
        store.insert(embeddings, metadata)
        store._check_consistency()  # не должен упасть

    def test_inconsistent_state_raises(self, store, embeddings, metadata):
        store.insert(embeddings, metadata)
        store._metadata.append({"doc_id": "extra"})  # ломаем консистентность
        with pytest.raises(RuntimeError, match="консистентность"):
            store._check_consistency()


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_index(self, store, embeddings, metadata):
        store.insert(embeddings, metadata)
        store.reset()
        assert store.ntotal == 0

    def test_reset_clears_metadata(self, store, embeddings, metadata):
        store.insert(embeddings, metadata)
        store.reset()
        assert store._metadata == []

    def test_reset_invalidates_cache(self, store, embeddings, metadata):
        store.insert(embeddings, metadata)
        _ = store.existing_doc_ids
        store.reset()
        assert store._doc_id_cache is None


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_creates_files(self, store, embeddings, metadata, tmp_path):
        store.insert(embeddings, metadata)
        store.save(tmp_path)
        assert (tmp_path / "index.faiss").exists()
        assert (tmp_path / "metadata.json").exists()

    def test_save_load_roundtrip(self, store, embeddings, metadata, tmp_path):
        store.insert(embeddings, metadata)
        store.save(tmp_path)

        loaded = FAISSVectorStore.load(tmp_path, embedding_dim=DIM)
        assert loaded.ntotal == 3
        assert len(loaded._metadata) == 3

    def test_load_metadata_correct(self, store, embeddings, metadata, tmp_path):
        store.insert(embeddings, metadata)
        store.save(tmp_path)

        loaded = FAISSVectorStore.load(tmp_path, embedding_dim=DIM)
        assert loaded._metadata == metadata

    def test_load_missing_index_raises(self, tmp_path):
        (tmp_path / "metadata.json").write_text("[]")
        with pytest.raises(FileNotFoundError, match="index.faiss"):
            FAISSVectorStore.load(tmp_path, embedding_dim=DIM)

    def test_load_missing_metadata_raises(self, store, embeddings, metadata, tmp_path):
        store.insert(embeddings, metadata)
        store.save(tmp_path)
        (tmp_path / "metadata.json").unlink()
        with pytest.raises(FileNotFoundError, match="metadata"):
            FAISSVectorStore.load(tmp_path, embedding_dim=DIM)

    def test_load_legacy_pkl_fallback(self, store, embeddings, metadata, tmp_path):
        """Если metadata.json нет но есть metadata.pkl — загружается legacy."""
        store.insert(embeddings, metadata)
        store.save(tmp_path)
        (tmp_path / "metadata.json").unlink()
        (tmp_path / "metadata.pkl").write_bytes(pickle.dumps(metadata))

        with patch("src.vector_store.faiss_store.logger") as mock_logger:
            loaded = FAISSVectorStore.load(tmp_path, embedding_dim=DIM)

        assert loaded._metadata == metadata
        # Должно быть предупреждение о legacy pkl
        warning_messages = " ".join(str(c.args) for c in mock_logger.warning.call_args_list)
        assert "pkl" in warning_messages.lower() or "legacy" in warning_messages.lower()

    def test_load_filters_unknown_kwargs(self, store, embeddings, metadata, tmp_path):
        """Лишние kwargs из Hydra-конфига игнорируются без ошибки."""
        store.insert(embeddings, metadata)
        store.save(tmp_path)
        loaded = FAISSVectorStore.load(
            tmp_path,
            embedding_dim=DIM,
            unknown_key="should_be_ignored",
        )
        assert loaded.ntotal == 3

    def test_save_creates_directory_if_missing(self, store, embeddings, metadata, tmp_path):
        target = tmp_path / "nested" / "dir"
        store.insert(embeddings, metadata)
        store.save(target)
        assert (target / "index.faiss").exists()

    def test_loaded_store_can_search(self, store, embeddings, metadata, tmp_path):
        """После load поиск работает корректно."""
        store.insert(embeddings, metadata)
        store.save(tmp_path)

        loaded = FAISSVectorStore.load(tmp_path, embedding_dim=DIM)
        results = loaded.search(embeddings[:1], top_k=1)
        assert len(results[0]) == 1
        assert "score" in results[0][0]
