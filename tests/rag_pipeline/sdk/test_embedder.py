from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.rag_pipeline.sdk.embedder import RAGInferenceEmbedder


class TestRAGInferenceEmbedder:
    @pytest.fixture
    def embedder(self):
        mock_model = MagicMock()
        mock_model.to.return_value.eval.return_value = mock_model

        mock_pooler = MagicMock()
        mock_pooler.return_value = torch.ones(2, 8)
        mock_pooler.to.return_value.eval.return_value = mock_pooler

        mock_tokenizer = MagicMock()
        encoded_mock = MagicMock()
        encoded_mock.__getitem__.side_effect = lambda k: torch.ones(2, 4, dtype=torch.long)
        encoded_mock.to.return_value = encoded_mock
        mock_tokenizer.return_value = encoded_mock

        return RAGInferenceEmbedder(
            model=mock_model,
            pooler=mock_pooler,
            tokenizer=mock_tokenizer,
            device="cpu",
            precision="fp32",
        )

    def test_encode_single_string(self, embedder):
        res = embedder.encode("Один текст")
        assert isinstance(res, np.ndarray)

    def test_encode_list_of_strings(self, embedder):
        res = embedder.encode(["Текст 1", "Текст 2"], batch_size=2)
        assert res.shape == (2, 8)
