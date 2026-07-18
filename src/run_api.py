import os

import uvicorn
from dotenv import load_dotenv


# Загружаем локальные секреты (если файла нет, например в K8s, функция просто проигнорирует это)
load_dotenv()

if __name__ == "__main__":
    # Читаем порт из среды. Обязательно конвертируем в int!
    # Если переменной нет, берем 8000 по умолчанию.
    api_port = int(os.getenv("API_PORT", 8000))

    uvicorn.run(
        "src.api.rest.server:app",  # <-- Исправлен путь импорта
        host="0.0.0.0",
        port=api_port,
        reload=False,  # На проде reload строго выключен
    )
