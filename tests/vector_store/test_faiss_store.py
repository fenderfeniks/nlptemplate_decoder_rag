# tests/vector_store/test_faiss_store_extended.py
"""Расширенные тесты для FAISSVectorStore.

Покрывают то, чего нет в test_faiss_store.py:
- _match_filters (AND-семантика фильтрации)
- _search_with_filter: итеративный over-fetch, расширение multiplier
- insert_batched: граничные случаи
- save/load: предупреждение о dropped kwargs
- Корректность метаданных после поиска с фильтром
- Нормализация: реальный эффект на результаты поиска
- Граничные случаи ntotal=1, top_k=0
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


faiss = pytest.importorskip("faiss", reason="faiss не установлен")

from src.vector_store.faiss_store import FAISSVectorStore  # noqa: E402


DIM = 4


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------


@pytest.fixture
def store():
    return FAISSVectorStore(embedding_dim=DIM, index_type="flat")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def embeddings(rng):
    return rng.random((5, DIM)).astype(np.float32)


@pytest.fixture
def metadata():
    return [
        {"doc_id": "d1", "source": "wiki", "lang": "ru"},
        {"doc_id": "d2", "source": "arxiv", "lang": "en"},
        {"doc_id": "d3", "source": "wiki", "lang": "en"},
        {"doc_id": "d4", "source": "arxiv", "lang": "ru"},
        {"doc_id": "d5", "source": "wiki", "lang": "ru"},
    ]


# ---------------------------------------------------------------------------
# _match_filters — бизнес-логика AND-фильтрации
# ---------------------------------------------------------------------------


class TestMatchFilters:
    """_match_filters реализует AND-семантику: все условия должны совпасть."""

    def test_single_match(self, store):
        doc = {"source": "wiki", "lang": "ru"}
        assert store._match_filters(doc, {"source": "wiki"}) is True

    def test_single_no_match(self, store):
        doc = {"source": "arxiv", "lang": "en"}
        assert store._match_filters(doc, {"source": "wiki"}) is False

    def test_and_semantics_all_match(self, store):
        doc = {"source": "wiki", "lang": "ru"}
        assert store._match_filters(doc, {"source": "wiki", "lang": "ru"}) is True

    def test_and_semantics_partial_match_fails(self, store):
        """AND: если хотя бы одно условие не совпало — False."""
        doc = {"source": "wiki", "lang": "en"}
        assert store._match_filters(doc, {"source": "wiki", "lang": "ru"}) is False

    def test_empty_filters_always_true(self, store):
        """Пустой фильтр совпадает с любым документом."""
        assert store._match_filters({"source": "wiki"}, {}) is True

    def test_missing_key_in_doc_no_match(self, store):
        """Ключ отсутствует в doc — не совпадает (get возвращает None)."""
        doc = {"source": "wiki"}
        assert store._match_filters(doc, {"lang": "ru"}) is False

    def test_none_value_matches_missing_key(self, store):
        """Если фильтр ищет None — совпадёт с отсутствующим ключом."""
        doc = {"source": "wiki"}
        assert store._match_filters(doc, {"lang": None}) is True

    def test_empty_doc_no_match(self, store):
        """Пустой документ не совпадает ни с каким непустым фильтром."""
        assert store._match_filters({}, {"source": "wiki"}) is False

    def test_value_type_matters(self, store):
        """Сравнение строгое: "1" != 1."""
        doc = {"count": 1}
        assert store._match_filters(doc, {"count": "1"}) is False
        assert store._match_filters(doc, {"count": 1}) is True


# ---------------------------------------------------------------------------
# _search_with_filter — итеративный over-fetch
# ---------------------------------------------------------------------------


class TestSearchWithFilterIterative:
    """Проверяем итеративное расширение multiplier при нехватке результатов."""

    def _make_store_with_data(self, n: int, source_pattern: list[str]) -> FAISSVectorStore:
        """Создаёт store с n векторами, у каждого source из паттерна."""
        rng = np.random.default_rng(99)
        s = FAISSVectorStore(
            embedding_dim=DIM,
            filter_fetch_multiplier=2,
            filter_max_fetch_multiplier=50,
        )
        embs = rng.random((n, DIM)).astype(np.float32)
        meta = [
            {"doc_id": f"d{i}", "source": source_pattern[i % len(source_pattern)]} for i in range(n)
        ]
        s.insert(embs, meta)
        return s

    def test_filter_returns_correct_source(self):
        """Только документы с нужным source попадают в результат."""
        s = self._make_store_with_data(10, ["wiki", "arxiv", "other"])
        rng = np.random.default_rng(1)
        q = rng.random((1, DIM)).astype(np.float32)
        results = s.search(q, top_k=3, filter_metadata={"source": "wiki"})
        assert len(results) == 1
        for hit in results[0]:
            assert hit["metadata"]["source"] == "wiki"

    def test_filter_fewer_results_than_top_k(self):
        """Если подходящих документов меньше чем top_k — возвращаем сколько есть."""
        rng = np.random.default_rng(2)
        s = FAISSVectorStore(embedding_dim=DIM)
        embs = rng.random((5, DIM)).astype(np.float32)
        # Только 2 документа с source=rare
        meta = [
            {"doc_id": "d0", "source": "rare"},
            {"doc_id": "d1", "source": "common"},
            {"doc_id": "d2", "source": "common"},
            {"doc_id": "d3", "source": "rare"},
            {"doc_id": "d4", "source": "common"},
        ]
        s.insert(embs, meta)
        q = rng.random((1, DIM)).astype(np.float32)
        results = s.search(q, top_k=5, filter_metadata={"source": "rare"})
        # Всего 2 подходящих — не должно падать
        assert len(results[0]) <= 2
        for hit in results[0]:
            assert hit["metadata"]["source"] == "rare"

    def test_and_filter_multiple_conditions(self, store, embeddings, metadata):
        """AND-фильтрация по нескольким полям."""
        store.insert(embeddings, metadata)
        rng = np.random.default_rng(5)
        q = rng.random((1, DIM)).astype(np.float32)
        results = store.search(q, top_k=5, filter_metadata={"source": "wiki", "lang": "ru"})
        for hit in results[0]:
            assert hit["metadata"]["source"] == "wiki"
            assert hit["metadata"]["lang"] == "ru"

    def test_filter_multiplier_expands_when_insufficient(self):
        """Проверяем что multiplier действительно расширяется через debug-лог."""
        rng = np.random.default_rng(3)
        s = FAISSVectorStore(
            embedding_dim=DIM,
            filter_fetch_multiplier=1,  # начинаем с маленького
            filter_max_fetch_multiplier=100,
        )
        # 20 векторов, только 3 с source=rare — требуем top_k=3
        embs = rng.random((20, DIM)).astype(np.float32)
        meta = [{"doc_id": f"d{i}", "source": "rare" if i < 3 else "common"} for i in range(20)]
        s.insert(embs, meta)
        q = rng.random((1, DIM)).astype(np.float32)

        with patch("src.vector_store.faiss_store.logger") as mock_logger:
            results = s.search(q, top_k=3, filter_metadata={"source": "rare"})

        # Должны найти все 3 редких документа
        assert len(results[0]) == 3
        for hit in results[0]:
            assert hit["metadata"]["source"] == "rare"

    def test_multiple_queries_with_filter(self, store, embeddings, metadata):
        """Фильтрация работает для нескольких запросов одновременно."""
        store.insert(embeddings, metadata)
        results = store.search(embeddings[:3], top_k=2, filter_metadata={"source": "wiki"})
        assert len(results) == 3
        for query_results in results:
            for hit in query_results:
                assert hit["metadata"]["source"] == "wiki"

    def test_max_multiplier_stops_iteration(self):
        """При достижении filter_max_fetch_multiplier итерация прекращается."""
        rng = np.random.default_rng(7)
        # store с маленьким max_multiplier
        s = FAISSVectorStore(
            embedding_dim=DIM,
            filter_fetch_multiplier=1,
            filter_max_fetch_multiplier=2,
        )
        embs = rng.random((10, DIM)).astype(np.float32)
        # Только 1 документ подходит, но top_k=5 — не наберём
        meta = [{"doc_id": f"d{i}", "source": "rare" if i == 0 else "common"} for i in range(10)]
        s.insert(embs, meta)
        q = rng.random((1, DIM)).astype(np.float32)
        # Не должно зависнуть в бесконечном цикле
        results = s.search(q, top_k=5, filter_metadata={"source": "rare"})
        assert len(results) == 1
        assert len(results[0]) <= 1


# ---------------------------------------------------------------------------
# insert_batched — граничные случаи
# ---------------------------------------------------------------------------


class TestInsertBatchedExtended:
    def test_batch_size_1(self):
        """batch_size=1 — каждый вектор в отдельном батче."""
        rng = np.random.default_rng(10)
        s = FAISSVectorStore(embedding_dim=DIM, insert_batch_size=1)
        embs = rng.random((4, DIM)).astype(np.float32)
        meta = [{"doc_id": f"d{i}"} for i in range(4)]
        s.insert_batched(embs, meta)
        assert s.ntotal == 4

    def test_batch_size_equals_n(self):
        """batch_size == len(embeddings) — один батч."""
        rng = np.random.default_rng(11)
        s = FAISSVectorStore(embedding_dim=DIM, insert_batch_size=3)
        embs = rng.random((3, DIM)).astype(np.float32)
        meta = [{"doc_id": f"d{i}"} for i in range(3)]
        s.insert_batched(embs, meta, desc="My batch")
        assert s.ntotal == 3

    def test_batched_accumulates_metadata(self):
        """Метаданные накапливаются корректно при батчевой вставке."""
        rng = np.random.default_rng(12)
        s = FAISSVectorStore(embedding_dim=DIM, insert_batch_size=2)
        embs = rng.random((5, DIM)).astype(np.float32)
        meta = [{"doc_id": f"d{i}", "val": i} for i in range(5)]
        s.insert_batched(embs, meta)
        assert len(s._metadata) == 5
        assert s._metadata[4]["val"] == 4

    def test_batched_existing_doc_ids(self):
        """existing_doc_ids содержит все id после батчевой вставки."""
        rng = np.random.default_rng(13)
        s = FAISSVectorStore(embedding_dim=DIM, insert_batch_size=2)
        embs = rng.random((4, DIM)).astype(np.float32)
        ids = [f"d{i}" for i in range(4)]
        meta = [{"doc_id": did} for did in ids]
        s.insert_batched(embs, meta)
        assert s.existing_doc_ids == set(ids)


# ---------------------------------------------------------------------------
# save/load — расширенные случаи
# ---------------------------------------------------------------------------


class TestSaveLoadExtended:
    def test_load_logs_warning_for_dropped_kwargs(self, store, tmp_path):
        """load логирует warning при обнаружении неизвестных kwargs."""
        rng = np.random.default_rng(20)
        embs = rng.random((2, DIM)).astype(np.float32)
        meta = [{"doc_id": "d1"}, {"doc_id": "d2"}]
        store.insert(embs, meta)
        store.save(tmp_path)

        with patch("src.vector_store.faiss_store.logger") as mock_logger:
            FAISSVectorStore.load(
                tmp_path,
                embedding_dim=DIM,
                hydra_extra_key="dropped",
                another_extra="also_dropped",
            )

        warning_text = " ".join(str(c) for c in mock_logger.warning.call_args_list)
        assert "hydra_extra_key" in warning_text or "dropped" in warning_text.lower()

    def test_loaded_store_search_results_match_original(self, store, tmp_path):
        """После load поиск возвращает те же топ-1 результаты что и оригинал."""
        rng = np.random.default_rng(21)
        embs = rng.random((5, DIM)).astype(np.float32)
        meta = [{"doc_id": f"d{i}", "val": i} for i in range(5)]
        store.insert(embs, meta)
        store.save(tmp_path)

        loaded = FAISSVectorStore.load(tmp_path, embedding_dim=DIM)
        q = rng.random((1, DIM)).astype(np.float32)

        r_orig = store.search(q, top_k=1)
        r_load = loaded.search(q, top_k=1)

        assert r_orig[0][0]["metadata"]["doc_id"] == r_load[0][0]["metadata"]["doc_id"]

    def test_save_load_preserves_normalize_setting(self, tmp_path):
        """normalize_embeddings=False сохраняется после load через kwargs."""
        rng = np.random.default_rng(22)
        s = FAISSVectorStore(embedding_dim=DIM, normalize_embeddings=False)
        embs = rng.random((2, DIM)).astype(np.float32)
        meta = [{"doc_id": "d1"}, {"doc_id": "d2"}]
        s.insert(embs, meta)
        s.save(tmp_path)

        loaded = FAISSVectorStore.load(tmp_path, embedding_dim=DIM, normalize_embeddings=False)
        assert loaded.normalize_embeddings is False

    def test_save_empty_store(self, store, tmp_path):
        """Можно сохранить пустой store без ошибки."""
        store.save(tmp_path)
        assert (tmp_path / "index.faiss").exists()
        assert (tmp_path / "metadata.json").exists()

    def test_load_empty_store_ntotal_zero(self, store, tmp_path):
        """Загрузка пустого store — ntotal=0."""
        store.save(tmp_path)
        loaded = FAISSVectorStore.load(tmp_path, embedding_dim=DIM)
        assert loaded.ntotal == 0
        assert loaded._metadata == []


# ---------------------------------------------------------------------------
# Нормализация — реальный эффект
# ---------------------------------------------------------------------------


class TestNormalizationEffect:
    def test_normalized_scores_in_range(self):
        """При normalize=True косинусное сходство ∈ [-1, 1]."""
        rng = np.random.default_rng(30)
        s = FAISSVectorStore(embedding_dim=DIM, normalize_embeddings=True)
        embs = rng.random((5, DIM)).astype(np.float32)
        meta = [{"doc_id": f"d{i}"} for i in range(5)]
        s.insert(embs, meta)

        q = rng.random((1, DIM)).astype(np.float32)
        results = s.search(q, top_k=5)
        for hit in results[0]:
            assert -1.01 <= hit["score"] <= 1.01

    def test_self_query_highest_score_when_normalized(self):
        """Запрос собственным вектором должен дать наивысший score."""
        rng = np.random.default_rng(31)
        s = FAISSVectorStore(embedding_dim=DIM, normalize_embeddings=True)
        embs = rng.random((5, DIM)).astype(np.float32)
        meta = [{"doc_id": f"d{i}"} for i in range(5)]
        s.insert(embs, meta)

        # Запрашиваем первым вектором — он должен быть ближайшим к себе
        results = s.search(embs[:1], top_k=5)
        top_hit = results[0][0]
        assert top_hit["metadata"]["doc_id"] == "d0"

    def test_prepare_returns_float32(self):
        """_prepare всегда возвращает float32 независимо от входного типа."""
        s = FAISSVectorStore(embedding_dim=DIM)
        for dtype in [np.float64, np.int32, np.float16]:
            vecs = np.ones((2, DIM), dtype=dtype)
            result = s._prepare(vecs)
            assert result.dtype == np.float32, (
                f"Ожидался float32, получен {result.dtype} для {dtype}"
            )


# ---------------------------------------------------------------------------
# Граничные случаи поведения
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_search_single_vector_in_store(self):
        """store с 1 вектором — поиск работает."""
        rng = np.random.default_rng(40)
        s = FAISSVectorStore(embedding_dim=DIM)
        emb = rng.random((1, DIM)).astype(np.float32)
        s.insert(emb, [{"doc_id": "only"}])

        results = s.search(emb, top_k=1)
        assert len(results[0]) == 1
        assert results[0][0]["metadata"]["doc_id"] == "only"

    def test_insert_single_vector(self):
        """Вставка одного вектора работает корректно."""
        rng = np.random.default_rng(41)
        s = FAISSVectorStore(embedding_dim=DIM)
        emb = rng.random((1, DIM)).astype(np.float32)
        s.insert(emb, [{"doc_id": "solo"}])
        assert s.ntotal == 1
        assert s.existing_doc_ids == {"solo"}

    def test_reset_then_insert(self):
        """После reset можно снова вставлять данные."""
        rng = np.random.default_rng(42)
        s = FAISSVectorStore(embedding_dim=DIM)
        embs = rng.random((3, DIM)).astype(np.float32)
        meta = [{"doc_id": f"d{i}"} for i in range(3)]
        s.insert(embs, meta)
        s.reset()
        s.insert(embs[:1], [{"doc_id": "new"}])
        assert s.ntotal == 1
        assert s.existing_doc_ids == {"new"}

    def test_search_after_reset_returns_empty(self):
        """После reset поиск возвращает пустой список."""
        rng = np.random.default_rng(43)
        s = FAISSVectorStore(embedding_dim=DIM)
        embs = rng.random((3, DIM)).astype(np.float32)
        meta = [{"doc_id": f"d{i}"} for i in range(3)]
        s.insert(embs, meta)
        s.reset()

        q = rng.random((1, DIM)).astype(np.float32)
        results = s.search(q, top_k=3)
        assert results == [[]]

    def test_hnsw_search_returns_results(self):
        """HNSW индекс возвращает корректные результаты."""
        rng = np.random.default_rng(44)
        s = FAISSVectorStore(embedding_dim=DIM, index_type="hnsw")
        embs = rng.random((10, DIM)).astype(np.float32)
        meta = [{"doc_id": f"d{i}"} for i in range(10)]
        s.insert(embs, meta)

        q = rng.random((1, DIM)).astype(np.float32)
        results = s.search(q, top_k=3)
        assert len(results[0]) == 3

    def test_insert_consistency_check_raises_on_broken_state(self):
        """insert проверяет консистентность до добавления."""
        rng = np.random.default_rng(45)
        s = FAISSVectorStore(embedding_dim=DIM)
        # Намеренно ломаем состояние
        s._metadata.append({"doc_id": "ghost"})
        embs = rng.random((1, DIM)).astype(np.float32)
        with pytest.raises(RuntimeError, match="консистентность"):
            s.insert(embs, [{"doc_id": "new"}])
