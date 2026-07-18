import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_generate_text_no_rag(async_client: AsyncClient, override_ml_deps, mock_generator):
    payload = {"query": "Привет, как дела?", "use_rag": False}

    response = await async_client.post("/chat/generate", json=payload)

    assert response.status_code == 200
    assert response.json()["answer"] == "Мок-ответ модели."
    # Проверяем, что генератор был вызван
    assert mock_generator.generate.called


@pytest.mark.asyncio
async def test_generate_text_with_rag(async_client: AsyncClient, override_ml_deps, mock_retriever):
    payload = {
        "query": "Что такое RAG?",
        "use_rag": True,
        "history": [
            {"role": "user", "content": "Привет"},
            {"role": "assistant", "content": "Привет! Чем могу помочь?"},
        ],
    }

    response = await async_client.post("/chat/generate", json=payload)

    assert response.status_code == 200
    assert response.json()["context_used"] == "Контекст из FAISS."
    assert "answer" in response.json()
    # Проверяем, что ретривер был вызван
    assert mock_retriever.retrieve_context.called


# 3. Тест на валидацию (ожидаем 422)
@pytest.mark.asyncio
async def test_generate_text_invalid_payload(async_client: AsyncClient, override_ml_deps):
    """
    Проверка, что API возвращает 422 при отсутствии обязательного поля 'query'.
    """
    payload = {"use_rag": False}  # Пропущено поле 'query'

    response = await async_client.post("/chat/generate", json=payload)

    assert response.status_code == 422  # Стандартная ошибка валидации FastAPI


# 4. Тест на пустой ответ от RAG (Edge Case)
@pytest.mark.asyncio
async def test_generate_text_rag_empty_context(
    async_client: AsyncClient, override_ml_deps, mock_retriever
):
    """
    Проверка, что система не падает, если RAG вернул пустоту.
    """
    # Переопределяем поведение мока для этого конкретного теста
    mock_retriever.retrieve_context.return_value = ""

    payload = {"query": "Вопрос, на который нет ответа в базе", "use_rag": True}

    response = await async_client.post("/chat/generate", json=payload)

    assert response.status_code == 200
    data = response.json()
    # Проверяем, что контекст пуст, но API отработало штатно
    assert data["context_used"] == ""
    assert "answer" in data
