import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000/chat/generate")
# ИСПРАВЛЕНИЕ: Читаем API-ключ
API_KEY = os.getenv("API_KEY", "")

st.set_page_config(page_title="Корпоративный NLP Ассистент", page_icon="🤖")
st.title("База знаний: Вопрос-Ответ")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Задайте вопрос по корпоративным документам..."):
    # ИСПРАВЛЕНИЕ: Вытаскиваем историю до того, как добавим текущий промпт
    # Берем последние 10 сообщений (5 пар вопрос-ответ), чтобы не перегружать память
    history_payload = [
        {"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages
    ][-10:]

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ИСПРАВЛЕНИЕ: Добавляем history в тело запроса
    payload = {"query": prompt, "history": history_payload, "use_rag": True, "max_tokens": 512}

    # ИСПРАВЛЕНИЕ: Добавляем заголовок авторизации
    headers = {"X-API-Key": API_KEY} if API_KEY else {}

    with st.chat_message("assistant"):
        with st.spinner("Анализирую документы..."):
            try:
                # ИСПРАВЛЕНИЕ: Увеличили таймаут до 120 секунд
                response = requests.post(API_URL, json=payload, headers=headers, timeout=120)
                response.raise_for_status()
                answer = response.json().get("answer", "Пустой ответ")

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                context = response.json().get("context_used")
                if context:
                    with st.expander("Посмотреть найденные документы"):
                        st.write(context)

            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка связи с сервером: {e}")
