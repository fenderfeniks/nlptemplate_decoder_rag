from unittest.mock import MagicMock

import pytest


async def test_generate_without_rag_returns_answer(
    async_client, override_ml_deps, mock_generator, mock_retriever
):
    response = await async_client.post(
        "/chat/generate",
        json={"query": "Привет", "use_rag": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Мок-ответ модели."
    assert body["context_used"] is None
    mock_retriever.retrieve_context.assert_not_called()
    mock_generator.generate.assert_called_once()


async def test_generate_with_rag_returns_context(
    async_client, override_ml_deps, mock_generator, mock_retriever
):
    response = await async_client.post(
        "/chat/generate",
        json={"query": "Как перезапустить pod?", "use_rag": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Мок-ответ модели."
    assert body["context_used"] == "Контекст из FAISS."
    mock_retriever.retrieve_context.assert_called_once_with("Как перезапустить pod?")
    mock_generator.generate.assert_called_once()


async def test_generate_with_rag_passes_context_to_prompt_builder(
    async_client, override_ml_deps, mock_generator, mock_retriever
):
    await async_client.post(
        "/chat/generate",
        json={"query": "Вопрос по базе", "use_rag": True},
    )

    prompt_arg = mock_generator.generate.call_args.args[0]
    assert "Контекст из FAISS." in prompt_arg
    assert "Вопрос по базе" in prompt_arg


async def test_generate_passes_max_tokens_to_generator(
    async_client, override_ml_deps, mock_generator
):
    await async_client.post(
        "/chat/generate",
        json={"query": "Тест", "use_rag": False, "max_tokens": 128},
    )

    _, kwargs = mock_generator.generate.call_args
    assert kwargs["max_new_tokens"] == 128
    assert mock_generator.generation_kwargs["max_new_tokens"] == 256


async def test_generate_with_history_prepends_dialog_to_prompt(
    async_client, override_ml_deps, mock_generator
):
    await async_client.post(
        "/chat/generate",
        json={
            "query": "Продолжи",
            "use_rag": False,
            "history": [{"role": "user", "content": "Расскажи про FastAPI"}],
        },
    )

    prompt_arg = mock_generator.generate.call_args.args[0]
    assert "История предыдущего диалога:" in prompt_arg
    assert "User: Расскажи про FastAPI" in prompt_arg
    assert "Продолжи" in prompt_arg


async def test_generate_returns_500_when_generator_fails(
    async_client, override_ml_deps, mock_generator
):
    mock_generator.generate.side_effect = RuntimeError("CUDA OOM")

    response = await async_client.post(
        "/chat/generate",
        json={"query": "Тест", "use_rag": False},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Ошибка генерации ответа."
