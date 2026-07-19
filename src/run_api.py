import os

import uvicorn
from dotenv import load_dotenv


# Загружаем локальные секреты
load_dotenv()

if __name__ == "__main__":
    api_port = int(os.getenv("API_PORT", 8000))

    uvicorn.run(
        "src.api.rest.server:app",
        host="0.0.0.0",
        port=api_port,
        reload=False,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
