# tests/vector_store/test_lsh.py
"""Тесты для LSHIndex.

Стратегия:
- Тесты с реальным datasketch запускаются напрямую (библиотека лёгкая).
- Тесты no-op пути патчат _DATASKETCH_AVAILABLE=False.
- Тесты save/load используют tmp_path, pickle мокируется где нужно.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest


# Пропускаем весь модуль если datasketch не установлен —
# тесты реального LSH всё равно упадут.
datasketch = pytest.importorskip("datasketch", reason="datasketch не установлен")

from src.vector_store.lsh import LSHIndex


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------


@pytest.fixture
def lsh():
    """LSHIndex с мягкими параметрами для скорости."""
    return LSHIndex(threshold=0.5, num_perm=32, ngram_size=3)


LONG_TEXT = "the quick brown fox jumps over the lazy dog near the river bank"
NEAR_DUP = "the quick brown fox jumps over the lazy dog close to the river"
DIFFERENT = "machine learning is a subfield of artificial intelligence research"
SHORT_TEXT = "hi"


# ---------------------------------------------------------------------------
# Инициализация
# ---------------------------------------------------------------------------


class TestLSHIndexInit:
    def test_is_available_when_datasketch_installed(self, lsh):
        """is_available=True при установленном datasketch."""
        assert lsh.is_available is True

    def test_parameters_stored(self, lsh):
        """Параметры сохраняются как атрибуты."""
        assert lsh.threshold == 0.5
        assert lsh.num_perm == 32
        assert lsh.ngram_size == 3

    def test_lsh_object_created(self, lsh):
        """_lsh не None после инициализации."""
        assert lsh._lsh is not None

    def test_not_available_without_datasketch(self):
        """is_available=False если datasketch недоступен."""
        with patch("src.vector_store.lsh._DATASKETCH_AVAILABLE", False):
            idx = LSHIndex()
            assert idx.is_available is False
            assert idx._lsh is None


# ---------------------------------------------------------------------------
# is_duplicate + register — основная логика
# ---------------------------------------------------------------------------


class TestIsDuplicateAndRegister:
    def test_empty_index_not_duplicate(self, lsh):
        """Пустой индекс — ничего не является дублем."""
        assert lsh.is_duplicate(LONG_TEXT) is False

    def test_registered_text_is_duplicate(self, lsh):
        """После register тот же текст распознаётся как дубль."""
        lsh.register("doc_1", LONG_TEXT)
        assert lsh.is_duplicate(LONG_TEXT) is True

    def test_near_duplicate_detected(self, lsh):
        """Близкий текст с Jaccard > threshold → дубль."""
        lsh.register("doc_1", LONG_TEXT)
        assert lsh.is_duplicate(NEAR_DUP) is True

    def test_different_text_not_duplicate(self, lsh):
        """Сильно отличающийся текст → не дубль."""
        lsh.register("doc_1", LONG_TEXT)
        assert lsh.is_duplicate(DIFFERENT) is False

    def test_is_duplicate_pure_predicate(self, lsh):
        """is_duplicate не изменяет индекс (повторный вызов — тот же результат)."""
        result1 = lsh.is_duplicate(LONG_TEXT)
        result2 = lsh.is_duplicate(LONG_TEXT)
        assert result1 == result2 == False  # noqa: E712

    def test_multiple_docs_registered(self, lsh):
        """Несколько документов — каждый детектируется."""
        lsh.register("doc_1", LONG_TEXT)
        lsh.register("doc_2", DIFFERENT)
        assert lsh.is_duplicate(LONG_TEXT) is True
        assert lsh.is_duplicate(DIFFERENT) is True


# ---------------------------------------------------------------------------
# no-op при недоступном datasketch
# ---------------------------------------------------------------------------


class TestNoOpWhenUnavailable:
    def _make_unavailable(self):
        with patch("src.vector_store.lsh._DATASKETCH_AVAILABLE", False):
            idx = LSHIndex()
        return idx

    def test_is_duplicate_returns_false(self):
        idx = self._make_unavailable()
        assert idx.is_duplicate(LONG_TEXT) is False

    def test_register_does_not_raise(self):
        idx = self._make_unavailable()
        idx.register("doc_1", LONG_TEXT)  # должен молча пройти

    def test_reset_does_not_raise(self):
        idx = self._make_unavailable()
        idx.reset()

    def test_save_does_not_raise(self, tmp_path):
        idx = self._make_unavailable()
        idx.save(tmp_path / "lsh.pkl")  # не должен упасть

    def test_load_does_not_raise(self, tmp_path):
        idx = self._make_unavailable()
        idx.load(tmp_path / "nonexistent.pkl")  # не должен упасть


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_index(self, lsh):
        """После reset ранее зарегистрированные документы не детектируются."""
        lsh.register("doc_1", LONG_TEXT)
        assert lsh.is_duplicate(LONG_TEXT) is True

        lsh.reset()
        assert lsh.is_duplicate(LONG_TEXT) is False

    def test_reset_preserves_parameters(self, lsh):
        """reset сохраняет threshold и num_perm."""
        lsh.reset()
        assert lsh.threshold == 0.5
        assert lsh.num_perm == 32

    def test_reset_lsh_is_new_object(self, lsh):
        """После reset _lsh — новый объект MinHashLSH."""
        old_lsh = lsh._lsh
        lsh.reset()
        assert lsh._lsh is not old_lsh


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_creates_file(self, lsh, tmp_path):
        """save создаёт файл на диске."""
        path = tmp_path / "lsh.pkl"
        lsh.register("doc_1", LONG_TEXT)
        lsh.save(path)
        assert path.exists()

    def test_save_load_roundtrip(self, lsh, tmp_path):
        """Загруженный индекс находит те же дубли что были до сохранения."""
        path = tmp_path / "lsh.pkl"
        lsh.register("doc_1", LONG_TEXT)
        lsh.save(path)

        lsh2 = LSHIndex(threshold=0.5, num_perm=32, ngram_size=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            lsh2.load(path)

        assert lsh2.is_duplicate(LONG_TEXT) is True

    def test_load_issues_pickle_warning(self, lsh, tmp_path):
        """load выбрасывает UserWarning о pickle."""
        path = tmp_path / "lsh.pkl"
        lsh.save(path)

        lsh2 = LSHIndex(threshold=0.5, num_perm=32, ngram_size=3)
        with pytest.warns(UserWarning, match="pickle"):
            lsh2.load(path)

    def test_load_missing_file_logs_warning(self, lsh, tmp_path):
        """load при отсутствии файла не падает, логирует warning."""
        with patch("src.vector_store.lsh.logger") as mock_logger:
            lsh.load(tmp_path / "missing.pkl")
        mock_logger.warning.assert_called_once()
        assert lsh._lsh is not None  # объект остаётся рабочим

    def test_save_is_valid_pickle(self, lsh, tmp_path):
        """Сохранённый файл — валидный pickle."""
        path = tmp_path / "lsh.pkl"
        lsh.save(path)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        assert obj is not None


# ---------------------------------------------------------------------------
# _compute_minhash
# ---------------------------------------------------------------------------


class TestComputeMinHash:
    def test_returns_minhash_object(self, lsh):
        """_compute_minhash возвращает MinHash при нормальном тексте."""
        from datasketch import MinHash

        result = lsh._compute_minhash(LONG_TEXT)
        assert isinstance(result, MinHash)

    def test_short_text_returns_minhash(self, lsh):
        """Короткий текст (< ngram_size слов) тоже возвращает MinHash."""
        from datasketch import MinHash

        result = lsh._compute_minhash(SHORT_TEXT)
        assert isinstance(result, MinHash)

    def test_empty_string_returns_minhash(self, lsh):
        """Пустая строка не вызывает исключений."""
        from datasketch import MinHash

        result = lsh._compute_minhash("")
        assert isinstance(result, MinHash)

    def test_returns_none_without_datasketch(self, lsh):
        """Возвращает None если _DATASKETCH_AVAILABLE=False."""
        with patch("src.vector_store.lsh._DATASKETCH_AVAILABLE", False):
            result = lsh._compute_minhash(LONG_TEXT)
        assert result is None

    def test_same_text_same_hashvalues(self, lsh):
        """Один и тот же текст даёт одинаковые хэш-значения."""
        m1 = lsh._compute_minhash(LONG_TEXT)
        m2 = lsh._compute_minhash(LONG_TEXT)
        import numpy as np

        np.testing.assert_array_equal(m1.hashvalues, m2.hashvalues)

    def test_different_texts_different_hashvalues(self, lsh):
        """Разные тексты дают разные хэш-значения."""
        import numpy as np

        m1 = lsh._compute_minhash(LONG_TEXT)
        m2 = lsh._compute_minhash(DIFFERENT)
        assert not np.array_equal(m1.hashvalues, m2.hashvalues)
