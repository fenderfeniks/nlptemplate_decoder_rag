import os
import uvicorn
from dotenv import load_dotenv

# Загружаем секреты и метаданные проекта
load_dotenv()

if __name__ == "__main__":
    # Читаем порт из .env. Обязательно конвертируем в int! 
    # Если переменной нет, берем 8000 по умолчанию.
    api_port = int(os.getenv("API_PORT", 8000))

    # В Uvicorn можно передать параметр env_file, чтобы воркеры тоже его увидели
    uvicorn.run(
        "src.api.server:app", 
        host="0.0.0.0", 
        port=api_port, 
        reload=False,         # На проде reload должен быть выключен!
        env_file=".env"       # <-- Заставляем Uvicorn прокинуть .env во все дочерние процессы
    )