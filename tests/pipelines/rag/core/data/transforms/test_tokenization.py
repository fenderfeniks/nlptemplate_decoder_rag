# tests/pipelines/rag/core/data/transforms/test_tokenization.py
import pytest
from datasets import Dataset

from src.pipelines.rag.core.data.transforms.tokenization import RAGTokenizationTransform


class DummyTokenizer:
    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return {
            "input_ids": [[1] * len(t) for t in texts],
            "attention_mask": [[1] * len(t) for t in texts]
        }


@pytest.fixture
def dummy_tokenizer():
    return DummyTokenizer()


class TestRAGTokenizationTransform:
    def test_invalid_init(self, dummy_tokenizer):
        with pytest.raises(ValueError, match="Неизвестный режим токенизации"):
            RAGTokenizationTransform(tokenizer=dummy_tokenizer, mode="invalid")
        with pytest.raises(ValueError, match="max_length должен быть положительным"):
            RAGTokenizationTransform(tokenizer=dummy_tokenizer, max_length=0)

    def test_missing_required_column_skipped(self, dummy_tokenizer):
        ds = Dataset.from_dict({"wrong_column": ["text"]})
        transform = RAGTokenizationTransform(tokenizer=dummy_tokenizer, mode="indexing", num_proc=None)
        assert transform(ds) is ds

    # --- Прямое тестирование методов токенизации (для 100% покрытия) ---

    def test_tokenize_indexing_direct(self, dummy_tokenizer):
        """Проверка внутренней функции индексации."""
        transform = RAGTokenizationTransform(tokenizer=dummy_tokenizer, mode="indexing")
        batch = {"text": ["abc", "defg"]}
        res = transform._tokenize_indexing(batch)
        assert res["input_ids"] == [[1, 1, 1], [1, 1, 1, 1]]

    def test_tokenize_contrastive_direct_no_negatives(self, dummy_tokenizer):
        """Проверка contrastive без негативных документов."""
        transform = RAGTokenizationTransform(tokenizer=dummy_tokenizer, mode="contrastive")
        batch = {"query": ["q1"], "positive_doc": ["pos1"]}
        res = transform._tokenize_contrastive(batch)
        assert "query_input_ids" in res
        assert "pos_input_ids" in res
        assert "neg_input_ids" not in res

    def test_tokenize_contrastive_direct_with_negatives(self, dummy_tokenizer):
        """Проверка contrastive с негативными документами и пропусками (None)."""
        transform = RAGTokenizationTransform(
            tokenizer=dummy_tokenizer, 
            mode="contrastive", 
            empty_doc_placeholder="EMPTY"
        )
        batch = {
            "query": ["q1", "q2"],
            "positive_doc": ["pos1", "pos2"],
            "negative_doc": ["neg1", None] # Второй без негатива
        }
        res = transform._tokenize_contrastive(batch)
        assert "neg_input_ids" in res
        assert res["neg_input_ids"][0] == [1, 1, 1, 1]
        assert res["neg_input_ids"][1] is None
        assert res["neg_attention_mask"][1] is None

    # --- Интеграционные тесты пайплайна ---

    def test_indexing_pipeline(self, dummy_tokenizer):
        """Интеграционный тест: __call__ для indexing."""
        ds = Dataset.from_dict({"text": ["abc"], "metadata": [{"id": 1}]})
        transform = RAGTokenizationTransform(tokenizer=dummy_tokenizer, mode="indexing", num_proc=None)
        
        result = transform(ds)
        assert "text" in result.column_names
        assert "metadata" in result.column_names
        assert result["input_ids"] == [[1, 1, 1]]

    def test_contrastive_pipeline(self, dummy_tokenizer):
        """Интеграционный тест: __call__ для contrastive."""
        ds = Dataset.from_dict({"query": ["q1"], "positive_doc": ["pos1"]})
        transform = RAGTokenizationTransform(tokenizer=dummy_tokenizer, mode="contrastive", num_proc=None)
        
        result = transform(ds)
        assert "query_input_ids" in result.column_names
        assert "pos_input_ids" in result.column_names