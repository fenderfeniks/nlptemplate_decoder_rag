from pathlib import Path
from unittest.mock import patch

import pytest

from src.vector_store.lsh import LSHIndex


class TestLSHIndex:
    def test_lsh_unavailable_fallback(self):
        """Если datasketch не установлен, LSH должен работать как no-op."""
        # Искусственно отключаем флаг доступности datasketch
        with patch("src.vector_store.lsh._DATASKETCH_AVAILABLE", False):
            lsh = LSHIndex()

            assert lsh.is_available is False
            assert lsh.is_duplicate("любой текст") is False

            # Вызов register не должен вызывать ошибок
            lsh.register("doc1", "любой текст")

    def test_lsh_duplicate_detection(self):
        """Проверка регистрации и детекции near-duplicate текстов."""
        lsh = LSHIndex(threshold=0.5, ngram_size=2)
        if not lsh.is_available:
            pytest.skip("Пакет datasketch не установлен")

        text1 = "мама мыла раму очень долго"
        text2 = "мама мыла раму очень быстро"
        text3 = "совершенно другой текст здесь"

        # Изначально дубликатов нет
        assert lsh.is_duplicate(text1) is False

        # Регистрируем первый текст
        lsh.register("doc1", text1)

        # Теперь text1 и очень похожий text2 должны быть дубликатами
        assert lsh.is_duplicate(text1) is True
        assert lsh.is_duplicate(text2) is True

        # Совершенно другой текст не должен быть дубликатом
        assert lsh.is_duplicate(text3) is False

    def test_lsh_reset(self):
        """Проверка очистки индекса."""
        lsh = LSHIndex()
        if not lsh.is_available:
            pytest.skip("Пакет datasketch не установлен")

        lsh.register("doc1", "тестовый текст")
        assert lsh.is_duplicate("тестовый текст") is True

        lsh.reset()
        assert lsh.is_duplicate("тестовый текст") is False

    def test_persistence(self, tmp_path: Path):
        """Проверка сохранения и загрузки LSH через pickle."""
        lsh = LSHIndex()
        if not lsh.is_available:
            pytest.skip("Пакет datasketch не установлен")

        lsh.register("doc1", "сохраняемый текст")
        file_path = tmp_path / "lsh.pkl"

        lsh.save(file_path)
        assert file_path.exists()

        lsh2 = LSHIndex()
        lsh2.load(file_path)

        assert lsh2.is_duplicate("сохраняемый текст") is True
