from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.vector_store.faiss_store import FAISSVectorStore


@pytest.fixture
def store():
    """Возвращает чистый инстанс FAISSVectorStore с flat-индексом[cite: 28]."""
    return FAISSVectorStore(embedding_dim=4, index_type="flat", normalize_embeddings=False)


class TestFAISSVectorStore:
    def test_invalid_index_type(self):
        """Ошибка при инициализации с неизвестным типом индекса[cite: 28]."""
        with pytest.raises(ValueError, match="Неизвестный тип индекса"):
            FAISSVectorStore(embedding_dim=4, index_type="unknown")

    def test_hnsw_initialization(self):
        """Проверка успешной инициализации HNSW индекса[cite: 28]."""
        store_hnsw = FAISSVectorStore(embedding_dim=4, index_type="hnsw")
        assert store_hnsw.index_type == "hnsw"
        assert store_hnsw.embedding_dim == 4

    def test_insert_shape_mismatch(self, store):
        """Защита от матриц с неправильной размерностью[cite: 28]."""
        embs = np.random.rand(2, 5)  # dim=5, а ожидается 4
        meta = [{"doc_id": "1"}, {"doc_id": "2"}]
        with pytest.raises(ValueError, match="Ожидается embeddings.shape"):
            store.insert(embs, meta)

    def test_insert_length_mismatch(self, store):
        """Защита от несовпадения длины векторов и метаданных[cite: 28]."""
        embs = np.random.rand(2, 4)
        meta = [{"doc_id": "1"}]  # 1 мета, 2 вектора
        with pytest.raises(ValueError, match="Несоответствие длин"):
            store.insert(embs, meta)

    def test_insert_rollback_on_error(self, store):
        """Проверка атомарности: метаданные откатываются, если index.add падает[cite: 28]."""
        embs = np.random.rand(2, 4).astype(np.float32)
        meta = [{"doc_id": "1"}, {"doc_id": "2"}]

        # Мокаем метод FAISS add, чтобы он вызвал исключение
        with patch.object(store.index, "add", side_effect=Exception("FAISS crash")):
            with pytest.raises(Exception, match="FAISS crash"):
                store.insert(embs, meta)

        # Метаданные должны были откатиться (длина 0)
        assert len(store._metadata) == 0
        assert store.ntotal == 0

    def test_search_no_filter(self, store):
        """Простой поиск без фильтров[cite: 28]."""
        embs = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        meta = [{"doc_id": "1", "val": "A"}, {"doc_id": "2", "val": "B"}]
        store.insert(embs, meta)

        query = np.array([[1, 0, 0, 0]], dtype=np.float32)
        res = store.search(query, top_k=1)

        assert len(res) == 1
        assert len(res[0]) == 1
        assert res[0][0]["metadata"]["doc_id"] == "1"

    def test_search_with_filter(self, store):
        """Итеративный over-fetch поиск с фильтрацией[cite: 28]."""
        # Первый вектор идеально подходит по косинусному сходству, но не подходит по фильтру
        # Второй вектор подходит по фильтру, но немного дальше
        embs = np.array([[1, 0, 0, 0], [0.9, 0.1, 0, 0]], dtype=np.float32)
        meta = [{"doc_id": "1", "type": "article"}, {"doc_id": "2", "type": "news"}]
        store.insert(embs, meta)

        query = np.array([[1, 0, 0, 0]], dtype=np.float32)

        res = store.search(query, top_k=1, filter_metadata={"type": "news"})

        assert len(res) == 1
        assert len(res[0]) == 1
        assert res[0][0]["metadata"]["doc_id"] == "2"

    def test_persistence(self, tmp_path: Path, store):
        """Сохранение и загрузка состояния базы (индекс + pickle метаданных)[cite: 28]."""
        embs = np.array([[1, 0, 0, 0]], dtype=np.float32)
        meta = [{"doc_id": "123"}]
        store.insert(embs, meta)

        store.save(tmp_path)

        assert (tmp_path / "index.faiss").exists()
        assert (tmp_path / "metadata.pkl").exists()

        store2 = FAISSVectorStore.load(tmp_path, embedding_dim=4)
        assert store2.ntotal == 1
        assert store2.existing_doc_ids == {"123"}
