import os

import requests
import streamlit as st


# Настраиваем адрес нашего FastAPI сервера
API_URL = os.getenv("API_URL", "http://localhost:8000/chat/generate")

st.set_page_config(page_title="Корпоративный NLP Ассистент", page_icon="🤖")
st.title("База знаний: Вопрос-Ответ")

# Инициализируем историю сообщений в памяти сессии
if "messages" not in st.session_state:
    st.session_state.messages = []

# Отрисовываем предыдущую историю сообщений
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Поле ввода для пользователя
if prompt := st.chat_input("Задайте вопрос по корпоративным документам..."):
    # 1. Добавляем вопрос пользователя в UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Формируем запрос к нашему FastAPI
    payload = {"query": prompt, "use_rag": True, "max_tokens": 512}

    # 3. Отправляем запрос и выводим ответ
    with st.chat_message("assistant"):
        with st.spinner("Анализирую документы..."):
            try:
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                answer = response.json().get("answer", "Пустой ответ")

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Дополнительно можно красиво вывести контекст (найденные документы),
                # если API его вернуло
                context = response.json().get("context_used")
                if context:
                    with st.expander("Посмотреть найденные документы"):
                        st.write(context)

            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка связи с сервером: {e}")
