# run_api.py
import uvicorn

if __name__ == "__main__":
    # Запускаем Uvicorn (ASGI-сервер), который поднимет наше FastAPI приложение
    uvicorn.run(
        "src.api.server:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False # На проде reload должен быть выключен!
    )