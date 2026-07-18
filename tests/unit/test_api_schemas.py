import pytest
from pydantic import ValidationError

from src.api.schemas import ChatMessage, ChatRequest


def test_chat_request_accepts_valid_payload():
    body = ChatRequest(query="Как настроить логгер?", use_rag=True, max_tokens=100)

    assert body.query == "Как настроить логгер?"
    assert body.use_rag is True
    assert body.max_tokens == 100


def test_chat_request_rejects_missing_query():
    with pytest.raises(ValidationError):
        ChatRequest(use_rag=False)


def test_chat_request_accepts_dialog_history():
    body = ChatRequest(
        query="Продолжи",
        history=[ChatMessage(role="user", content="Расскажи про Hydra")],
    )

    assert len(body.history) == 1
    assert body.history[0].content == "Расскажи про Hydra"
