# tests/decoder_pipeline/core/data/test_cleaners.py
from src.decoder_pipeline.core.data.cleaners import RegexCleaner, TextCleaningPipeline


class TestRegexCleaner:
    def test_removes_html_tags(self):
        cleaner = RegexCleaner(pattern=r"<.*?>", replacement="")
        assert cleaner.clean("<div>Hello</div>") == "Hello"

    def test_replaces_pattern(self):
        cleaner = RegexCleaner(pattern=r"\d+", replacement="NUM")
        assert cleaner.clean("Заказ 123 готов") == "Заказ NUM готов"


class TestTextCleaningPipeline:
    def test_pipeline_executes_sequentially(self):
        cleaner1 = RegexCleaner(pattern=r"bad", replacement="good")
        cleaner2 = RegexCleaner(pattern=r"!", replacement=".")
        
        pipeline = TextCleaningPipeline([cleaner1, cleaner2])
        assert pipeline("This is bad!") == "This is good."