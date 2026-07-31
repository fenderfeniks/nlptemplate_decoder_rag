# tests/decoder_pipeline/api/test_utils.py
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, Request

from src.decoder_pipeline.api.rest.dependencies import get_generator, get_prompt_manager
from src.decoder_pipeline.api.rest.limiter import get_real_ip


class TestDependencies:
    def test_get_prompt_manager_success(self):
        mock_request = MagicMock(spec=Request)
        mock_manager = MagicMock()
        mock_request.app.state.prompt_manager = mock_manager

        result = get_prompt_manager(mock_request)
        assert result == mock_manager

    def test_get_prompt_manager_raises_503(self):
        """Если PromptManager не готов, возвращаем 503."""
        mock_request = MagicMock(spec=Request)
        mock_request.app.state.prompt_manager = None

        with pytest.raises(HTTPException) as exc:
            get_prompt_manager(mock_request)
        assert exc.value.status_code == 503

    def test_get_generator_success(self):
        mock_request = MagicMock(spec=Request)
        mock_generator = MagicMock()
        mock_request.app.state.ml_models = {"generator": mock_generator}

        result = get_generator(mock_request)
        assert result == mock_generator

    def test_get_generator_raises_503(self):
        """Если генератор не готов, возвращаем 503."""
        mock_request = MagicMock(spec=Request)
        mock_request.app.state.ml_models = {}

        with pytest.raises(HTTPException) as exc:
            get_generator(mock_request)
        assert exc.value.status_code == 503


class TestLimiterUtils:
    def test_get_real_ip_with_x_forwarded_for(self):
        """Извлечение IP должно приоритизировать первый IP из X-Forwarded-For."""
        mock_request = MagicMock()
        mock_request.headers.get.return_value = "192.168.1.1, 10.0.0.1"

        ip = get_real_ip(mock_request)
        assert ip == "192.168.1.1"

    @patch("src.decoder_pipeline.api.rest.limiter.get_remote_address")
    def test_get_real_ip_fallback(self, mock_remote_address):
        """Если заголовка нет, используем fallback."""
        mock_request = MagicMock()
        mock_request.headers.get.return_value = None
        mock_remote_address.return_value = "127.0.0.1"

        ip = get_real_ip(mock_request)
        assert ip == "127.0.0.1"
