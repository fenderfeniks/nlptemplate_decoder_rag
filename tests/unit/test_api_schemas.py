import pytest
from pydantic import ValidationError

from src.api.schemas import ChatMessage, ChatRequest


def test_chat_request_accepts_valid_payload():
    request = ChatRequest(query="Как настроить логгер?", use_rag=True, max_tokens=100)

    assert request.query == "Как настроить логгер?"
    assert request.use_rag is True
    assert request.max_tokens == 100


def test_chat_request_rejects_missing_query():
    with pytest.raises(ValidationError):
        ChatRequest(use_rag=False)


def test_chat_request_accepts_dialog_history():
    request = ChatRequest(
        query="Продолжи",
        history=[ChatMessage(role="user", content="Расскажи про Hydra")],
    )

    assert len(request.history) == 1
    assert request.history[0].content == "Расскажи про Hydra"
