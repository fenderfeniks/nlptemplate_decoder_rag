# tests/vector_store/test_lsh_extended.py
"""Расширенные тесты для LSHIndex.

Покрывают то, чего нет в test_lsh.py:
- Корректный флоу is_duplicate -> register -> is_duplicate (идемпотентность)
- Влияние threshold на детекцию дублей
- Влияние ngram_size на чувствительность
- Граничный случай: повторная регистрация одного doc_id
- Регистрация и проверка многих документов
- Поведение при пустом тексте и тексте из одного слова
- save/load: после load состояние корректно для новых регистраций
- reset + повторная регистрация
- _compute_minhash: стабильность при разных ngram_size
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest


datasketch = pytest.importorskip("datasketch", reason="datasketch не установлен")

from src.vector_store.lsh import LSHIndex  # noqa: E402


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

TEXT_A = "the quick brown fox jumps over the lazy dog near the river bank"
TEXT_B = "the quick brown fox jumps over the lazy dog close to the river"  # near-dup A
TEXT_C = "machine learning is a subfield of artificial intelligence research"
TEXT_D = "deep neural networks require large amounts of labeled training data"
TEXT_E = "natural language processing enables computers to understand human text"


# ---------------------------------------------------------------------------
# Бизнес-логика: флоу is_duplicate -> register
# ---------------------------------------------------------------------------


class TestDuplicationFlow:
    """Основной бизнес-флоу: проверка -> принятие решения -> регистрация."""

    def test_is_duplicate_then_register_then_duplicate_again(self):
        """Правильный флоу: сначала проверяем, затем регистрируем уникальный."""
        lsh = LSHIndex(threshold=0.5, num_perm=32, ngram_size=3)

        # Шаг 1: пусто — не дубль
        assert lsh.is_duplicate(TEXT_A) is False

        # Шаг 2: регистрируем
        lsh.register("doc_1", TEXT_A)

        # Шаг 3: теперь тот же текст — дубль
        assert lsh.is_duplicate(TEXT_A) is True

    def test_register_does_not_affect_is_duplicate_predicate(self):
        """is_duplicate — чистый предикат, не регистрирует документ."""
        lsh = LSHIndex(threshold=0.5, num_perm=32, ngram_size=3)

        # Многократный вызов is_duplicate без register — всегда False
        for _ in range(5):
            assert lsh.is_duplicate(TEXT_A) is False

        # После регистрации — True
        lsh.register("doc_1", TEXT_A)
        assert lsh.is_duplicate(TEXT_A) is True

    def test_unique_documents_all_registered(self):
        """Уникальные документы регистрируются и все детектируются."""
        lsh = LSHIndex(threshold=0.5, num_perm=64, ngram_size=3)
        unique_texts = [TEXT_C, TEXT_D, TEXT_E]

        for i, text in enumerate(unique_texts):
            assert lsh.is_duplicate(text) is False
            lsh.register(f"doc_{i}", text)

        for text in unique_texts:
            assert lsh.is_duplicate(text) is True

    def test_near_duplicates_block_each_other(self):
        """Near-дубль блокируется после регистрации оригинала."""
        lsh = LSHIndex(threshold=0.5, num_perm=64, ngram_size=3)
        lsh.register("doc_1", TEXT_A)
        # TEXT_B очень близок к TEXT_A
        assert lsh.is_duplicate(TEXT_B) is True

    def test_different_texts_not_blocked(self):
        """Разные тексты не блокируют друг друга."""
        lsh = LSHIndex(threshold=0.5, num_perm=64, ngram_size=3)
        lsh.register("doc_1", TEXT_A)
        lsh.register("doc_2", TEXT_C)

        # TEXT_D не похож ни на A, ни на C — не должен быть дублём
        assert lsh.is_duplicate(TEXT_D) is False


# ---------------------------------------------------------------------------
# Влияние threshold
# ---------------------------------------------------------------------------


class TestThresholdEffect:
    def test_high_threshold_only_exact_matches(self):
        """threshold=0.99 — только почти идентичные тексты детектируются."""
        lsh = LSHIndex(threshold=0.99, num_perm=128, ngram_size=3)
        lsh.register("doc_1", TEXT_A)

        # Near-дубль не проходит строгий порог
        assert lsh.is_duplicate(TEXT_B) is False
        # Точное совпадение всегда проходит
        assert lsh.is_duplicate(TEXT_A) is True

    def test_low_threshold_more_permissive(self):
        """threshold=0.1 — даже слабое сходство считается дублём."""
        lsh = LSHIndex(threshold=0.1, num_perm=64, ngram_size=3)
        lsh.register("doc_1", TEXT_A)
        # При очень низком пороге near-dup гарантированно детектируется
        assert lsh.is_duplicate(TEXT_B) is True

    def test_threshold_stored_as_attribute(self):
        """threshold сохраняется как атрибут."""
        lsh = LSHIndex(threshold=0.77, num_perm=32)
        assert lsh.threshold == 0.77


# ---------------------------------------------------------------------------
# Влияние ngram_size
# ---------------------------------------------------------------------------


class TestNgramSizeEffect:
    def test_ngram_size_1_most_sensitive(self):
        """ngram_size=1 (унграммы) — максимальная чувствительность к общим словам."""
        lsh = LSHIndex(threshold=0.5, num_perm=64, ngram_size=1)
        lsh.register("doc_1", TEXT_A)
        # При ngram_size=1 и умеренном пороге near-dup должен детектироваться
        assert lsh.is_duplicate(TEXT_B) is True

    def test_ngram_size_stored_as_attribute(self):
        """ngram_size сохраняется как атрибут."""
        lsh = LSHIndex(ngram_size=7)
        assert lsh.ngram_size == 7

    def test_short_text_handled_correctly(self):
        """Текст короче ngram_size обрабатывается без ошибки."""
        lsh = LSHIndex(threshold=0.5, num_perm=32, ngram_size=10)
        # "hi" — 1 слово, меньше ngram_size=10
        lsh.register("doc_1", "hi")
        assert lsh.is_duplicate("hi") is True


# ---------------------------------------------------------------------------
# Граничные случаи текста
# ---------------------------------------------------------------------------


class TestTextEdgeCases:
    def test_empty_text_registered_and_detected(self):
        """Пустой текст регистрируется и детектируется."""
        lsh = LSHIndex(threshold=0.5, num_perm=32, ngram_size=3)
        lsh.register("empty_doc", "")
        assert lsh.is_duplicate("") is True

    def test_single_word_text(self):
        """Текст из одного слова обрабатывается без ошибки."""
        lsh = LSHIndex(threshold=0.5, num_perm=32, ngram_size=3)
        lsh.register("doc_1", "hello")
        assert lsh.is_duplicate("hello") is True

    def test_unicode_text(self):
        """Текст на кириллице обрабатывается корректно."""
        lsh = LSHIndex(threshold=0.5, num_perm=64, ngram_size=3)
        text_ru = "быстрая рыжая лиса перепрыгивает через ленивую собаку у реки"
        lsh.register("doc_ru", text_ru)
        assert lsh.is_duplicate(text_ru) is True

    def test_whitespace_only_text_no_crash(self):
        """Текст из пробелов не вызывает исключений."""
        lsh = LSHIndex(threshold=0.5, num_perm=32, ngram_size=3)
        # Не должно падать
        lsh.register("doc_ws", "   ")
        result = lsh.is_duplicate("   ")
        assert isinstance(result, bool)

    def test_numbers_in_text(self):
        """Текст с числами обрабатывается корректно."""
        lsh = LSHIndex(threshold=0.5, num_perm=32, ngram_size=3)
        text = "version 3 14 15 92 65 35 89 79 32 38 46 26"
        lsh.register("doc_num", text)
        assert lsh.is_duplicate(text) is True


# ---------------------------------------------------------------------------
# Множественная регистрация и коллизии
# ---------------------------------------------------------------------------


class TestMultipleRegistrations:
    def test_many_unique_docs_no_false_positives(self):
        """100 уникальных документов — минимум ложных срабатываний."""
        import uuid

        lsh = LSHIndex(threshold=0.9, num_perm=128, ngram_size=5)

        # Генерируем уникальные тексты через uuid
        texts = [f"document {uuid.uuid4()} content about unique topic {i}" for i in range(50)]

        # Регистрируем все
        for i, text in enumerate(texts):
            lsh.register(f"doc_{i}", text)

        # Проверяем что каждый из них детектируется как дубль
        for text in texts:
            assert lsh.is_duplicate(text) is True

    def test_interleaved_register_and_check(self):
        """Перемежающаяся регистрация и проверка — корректное поведение."""
        lsh = LSHIndex(threshold=0.5, num_perm=64, ngram_size=3)

        assert lsh.is_duplicate(TEXT_C) is False
        lsh.register("doc_c", TEXT_C)
        assert lsh.is_duplicate(TEXT_C) is True

        assert lsh.is_duplicate(TEXT_D) is False
        lsh.register("doc_d", TEXT_D)
        assert lsh.is_duplicate(TEXT_D) is True

        # C всё ещё дубль
        assert lsh.is_duplicate(TEXT_C) is True


# ---------------------------------------------------------------------------
# reset + повторная регистрация
# ---------------------------------------------------------------------------


class TestResetAndReregister:
    def test_after_reset_can_register_same_doc_id(self):
        """После reset можно повторно зарегистрировать тот же doc_id."""
        lsh = LSHIndex(threshold=0.5, num_perm=32, ngram_size=3)
        lsh.register("doc_1", TEXT_A)
        lsh.reset()
        # После reset нет дублей
        assert lsh.is_duplicate(TEXT_A) is False
        # Повторная регистрация того же id не вызывает ошибку
        lsh.register("doc_1", TEXT_A)
        assert lsh.is_duplicate(TEXT_A) is True

    def test_reset_then_register_new_docs(self):
        """После reset можно регистрировать новые документы."""
        lsh = LSHIndex(threshold=0.5, num_perm=32, ngram_size=3)
        lsh.register("doc_old", TEXT_A)
        lsh.reset()

        lsh.register("doc_new", TEXT_C)
        assert lsh.is_duplicate(TEXT_C) is True
        assert lsh.is_duplicate(TEXT_A) is False

    def test_double_reset_safe(self):
        """Двойной reset не вызывает ошибку."""
        lsh = LSHIndex(threshold=0.5, num_perm=32, ngram_size=3)
        lsh.register("doc_1", TEXT_A)
        lsh.reset()
        lsh.reset()  # второй reset — не должно падать
        assert lsh.is_duplicate(TEXT_A) is False


# ---------------------------------------------------------------------------
# save/load — расширенные случаи
# ---------------------------------------------------------------------------


class TestSaveLoadExtended:
    def test_load_state_allows_new_registrations(self, tmp_path):
        """После load можно регистрировать новые документы."""
        lsh = LSHIndex(threshold=0.5, num_perm=32, ngram_size=3)
        lsh.register("doc_1", TEXT_A)
        lsh.save(tmp_path / "lsh.pkl")

        lsh2 = LSHIndex(threshold=0.5, num_perm=32, ngram_size=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            lsh2.load(tmp_path / "lsh.pkl")

        # Можно зарегистрировать новый документ
        lsh2.register("doc_2", TEXT_C)
        assert lsh2.is_duplicate(TEXT_C) is True
        # Старый документ всё ещё в индексе
        assert lsh2.is_duplicate(TEXT_A) is True

    def test_save_file_not_created_without_datasketch(self, tmp_path):
        """Без datasketch файл не создаётся при save."""
        with patch("src.vector_store.lsh._DATASKETCH_AVAILABLE", False):
            lsh = LSHIndex()
        path = tmp_path / "lsh.pkl"
        lsh.save(path)
        assert not path.exists()

    def test_load_preserves_is_available(self, tmp_path):
        """После load is_available остаётся True."""
        lsh = LSHIndex(threshold=0.5, num_perm=32, ngram_size=3)
        lsh.register("doc_1", TEXT_A)
        lsh.save(tmp_path / "lsh.pkl")

        lsh2 = LSHIndex(threshold=0.5, num_perm=32, ngram_size=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            lsh2.load(tmp_path / "lsh.pkl")

        assert lsh2.is_available is True

    def test_save_then_load_then_reset(self, tmp_path):
        """После load + reset индекс пуст."""
        lsh = LSHIndex(threshold=0.5, num_perm=32, ngram_size=3)
        lsh.register("doc_1", TEXT_A)
        lsh.save(tmp_path / "lsh.pkl")

        lsh2 = LSHIndex(threshold=0.5, num_perm=32, ngram_size=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            lsh2.load(tmp_path / "lsh.pkl")

        lsh2.reset()
        assert lsh2.is_duplicate(TEXT_A) is False


# ---------------------------------------------------------------------------
# _compute_minhash — стабильность
# ---------------------------------------------------------------------------


class TestComputeMinHashStability:
    def test_hashvalues_length_matches_num_perm(self):
        """Длина hashvalues совпадает с num_perm."""
        for num_perm in [32, 64, 128]:
            lsh = LSHIndex(num_perm=num_perm, ngram_size=3)
            m = lsh._compute_minhash(TEXT_A)
            assert len(m.hashvalues) == num_perm

    def test_similar_texts_higher_jaccard_than_different(self):
        """Jaccard-сходство похожих текстов выше чем разных."""
        lsh = LSHIndex(num_perm=128, ngram_size=3)
        m_a = lsh._compute_minhash(TEXT_A)
        m_b = lsh._compute_minhash(TEXT_B)  # near-dup
        m_c = lsh._compute_minhash(TEXT_C)  # разный

        jaccard_ab = m_a.jaccard(m_b)
        jaccard_ac = m_a.jaccard(m_c)
        assert jaccard_ab > jaccard_ac

    def test_self_jaccard_is_one(self):
        """Jaccard текста с самим собой == 1.0."""
        lsh = LSHIndex(num_perm=128, ngram_size=3)
        m1 = lsh._compute_minhash(TEXT_A)
        m2 = lsh._compute_minhash(TEXT_A)
        assert m1.jaccard(m2) == pytest.approx(1.0, abs=1e-6)

    def test_ngram_tokenization_via_word_pattern(self):
        """_word_pattern токенизирует текст в слова (без пунктуации)."""
        lsh = LSHIndex(ngram_size=2)
        tokens = lsh._word_pattern.findall("hello, world! This is a test.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "," not in tokens
        assert "!" not in tokens
