# tests/pipelines/base/core/data/test_cleaners.py
import pytest

from src.pipelines.base.core.data.cleaners import (
    RegexCleaner,
    TextCleaningPipeline,
)


class TestRegexCleaner:
    def test_regex_cleaning(self):
        """Проверка замены по паттерну."""
        cleaner = RegexCleaner(pattern=r"<[^>]+>", replacement="")
        assert cleaner.clean("Привет <b>мир</b>!") == "Привет мир!"

    def test_regex_cleaning_default_replacement(self):
        """Проверка дефолтной замены (на пустую строку)."""
        cleaner = RegexCleaner(pattern=r"\d+")
        assert cleaner.clean("Текст 123 и 45") == "Текст  и "

    def test_non_string_input_ignored(self):
        """Если на вход приходит не строка (например, None), она возвращается как есть."""
        cleaner = RegexCleaner(pattern=r"test")
        assert cleaner.clean(None) is None
        assert cleaner.clean(123) == 123


class TestTextCleaningPipeline:
    def test_pipeline_sequential_cleaning(self):
        """Проверка, что клинеры применяются строго по очереди."""
        # Первый клинер меняет 1 на A, второй A на B
        cleaner1 = RegexCleaner(pattern=r"1", replacement="A")
        cleaner2 = RegexCleaner(pattern=r"A", replacement="B")
        pipeline = TextCleaningPipeline(cleaners=[cleaner1, cleaner2])
        
        assert pipeline("123") == "B23"
        
    def test_pipeline_non_string_input_ignored(self):
        """Пайплайн должен безопасно пробрасывать нестроковые типы."""
        pipeline = TextCleaningPipeline(cleaners=[RegexCleaner(r"a", "b")])
        assert pipeline(None) is None
        assert pipeline(["list", "of", "strings"]) == ["list", "of", "strings"]