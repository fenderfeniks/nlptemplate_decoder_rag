import pandas as pd
from omegaconf import OmegaConf

from src.tools.batch_analytics import (
    _build_qa_input,
    _build_sequence_input,
    _build_similarity_input,
    _build_token_input,
)


class TestBatchAnalyticsBuilders:
    def test_build_sequence_input_multiclass(self):
        df = pd.DataFrame({"text_col": ["Текст 1", "Текст 2"]})
        cfg = OmegaConf.create({"data": {"text_column": "text_col"}})

        result = _build_sequence_input(df, cfg)
        assert result == ["Текст 1", "Текст 2"]

    def test_build_sequence_input_nli(self):
        df = pd.DataFrame(
            {"text_col": ["Premise 1", "Premise 2"], "pair_col": ["Hypothesis 1", "Hypothesis 2"]}
        )
        cfg = OmegaConf.create(
            {"data": {"text_column": "text_col", "text_pair_column": "pair_col"}}
        )

        result = _build_sequence_input(df, cfg)
        assert result == [("Premise 1", "Hypothesis 1"), ("Premise 2", "Hypothesis 2")]

    def test_build_similarity_input(self):
        df = pd.DataFrame({"enc_col": ["Документ 1", "Документ 2"]})
        cfg = OmegaConf.create({"data": {"encoding_column": "enc_col"}})

        result = _build_similarity_input(df, cfg)
        assert result == ["Документ 1", "Документ 2"]

    def test_build_token_input(self):
        df = pd.DataFrame({"toks": [["A", "B"], ["C"]]})
        cfg = OmegaConf.create({"data": {"tokens_column": "toks"}})

        result = _build_token_input(df, cfg)
        assert result == [["A", "B"], ["C"]]

    def test_build_qa_input(self):
        df = pd.DataFrame({"q": ["Вопрос 1", "Вопрос 2"], "c": ["Контекст 1", "Контекст 2"]})
        cfg = OmegaConf.create({"data": {"question_column": "q", "context_column": "c"}})

        result = _build_qa_input(df, cfg)
        assert result == [("Вопрос 1", "Контекст 1"), ("Вопрос 2", "Контекст 2")]
