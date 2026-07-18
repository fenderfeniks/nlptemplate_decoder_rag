from src.core.models.promts import PromptManager


def test_build_simple_prompt_contains_query():
    prompt = PromptManager.build_simple_prompt("Тестовый запрос")

    assert "Тестовый запрос" in prompt
    assert "Ты — полезный ИИ-ассистент." in prompt


def test_build_rag_prompt_contains_context_and_query():
    prompt = PromptManager.build_rag_prompt(
        query="Кто написал этот код?",
        context="Этот код написала команда MLOps.",
    )

    assert "Кто написал этот код?" in prompt
    assert "Этот код написала команда MLOps." in prompt
    assert "<context>" in prompt
