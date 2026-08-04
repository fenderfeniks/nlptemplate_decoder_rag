# tests/pipelines/rag/core/data/transforms/test_metadata.py
import pytest
from datasets import Dataset

from src.pipelines.rag.core.data.transforms.metadata import MetadataInjectorTransform


class TestMetadataInjectorTransform:
    def test_invalid_template(self):
        with pytest.raises(ValueError, match="должен содержать плейсхолдеры"):
            MetadataInjectorTransform(template="Неверный шаблон {meta_string}")

    def test_missing_columns_skipped(self):
        ds = Dataset.from_dict({"text": ["текст"]})
        transform = MetadataInjectorTransform(metadata_column="missing", num_proc=None)
        assert transform(ds) is ds
        
        ds2 = Dataset.from_dict({"metadata": [{"a": 1}]})
        transform2 = MetadataInjectorTransform(text_column="missing", num_proc=None)
        assert transform2(ds2) is ds2

    def test_metadata_injection(self):
        """Успешная инъекция метаданных по шаблону."""
        ds = Dataset.from_dict({
            "text": ["Основной текст статьи."],
            "metadata": [{"title": "RAG systems", "year": 2024, "empty_key": None, "blank_key": "   "}]
        })
        
        template = "[META]\n{meta_string}\n[/META]\n{text}"
        transform = MetadataInjectorTransform(template=template, num_proc=None)
        result = transform(ds)
        
        expected_meta = "Title: RAG systems\nYear: 2024"
        expected_full = f"[META]\n{expected_meta}\n[/META]\nОсновной текст статьи."
        
        assert result["text"][0] == expected_full

    def test_empty_metadata_skipped(self):
        """Если метаданные пустые, текст остается без изменений."""
        ds = Dataset.from_dict({
            "text": ["Текст 1", "Текст 2"],
            "metadata": [{}, None]
        })
        transform = MetadataInjectorTransform(num_proc=None)
        result = transform(ds)
        
        assert result["text"] == ["Текст 1", "Текст 2"]