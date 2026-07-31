import asyncio
import logging
import os

import hydra
from aiogram import Bot, Dispatcher
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from src.decoder_pipeline.core.prompts.manager import PromptManager
from src.decoder_pipeline.sdk.inference import LLMGenerationClient
from src.tg_bot.handlers.chat import router as chat_router


load_dotenv()
logger = logging.getLogger(__name__)


def _build_orchestrator(cfg: DictConfig):
    from src.application.orchestrator import RAGOrchestrator

    logger.info("Сборка сетевого RAGOrchestrator для локального бота...")

    # URL микросервисов берем из окружения или ставим дефолтные
    llm_url = os.getenv("LLM_API_URL", "http://localhost:8000/v1")
    rag_url = os.getenv("RAG_API_URL", "http://localhost:8001")

    llm_client = LLMGenerationClient(api_base=llm_url)
    prompt_manager = PromptManager(cfg.get("prompts", {}))

    return RAGOrchestrator(
        rag_api_url=rag_url,
        llm_client=llm_client,
        prompt_manager=prompt_manager,
        default_template=cfg.tg_bot.get("rag_template", "rag_qa"),
    )


@hydra.main(config_path="../../configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    bot_token = os.getenv("TG_BOT_TOKEN")
    if not bot_token:
        raise ValueError("TG_BOT_TOKEN не найден в .env")

    bot = Bot(token=bot_token)
    dp = Dispatcher()
    dp.include_router(chat_router)

    dp["cfg"] = cfg
    dp["prompt_manager"] = PromptManager(cfg.get("prompts", {}))

    use_orchestrator = cfg.tg_bot.get("use_orchestrator", False)
    if use_orchestrator:
        dp["orchestrator"] = _build_orchestrator(cfg)
        logger.info("Режим: Сетевой RAGOrchestrator.")
    else:
        dp["orchestrator"] = None
        logger.info("Режим: Оркестратор отключен.")

    async def start_polling() -> None:
        await bot.delete_webhook(drop_pending_updates=True)
        logger.info("Запуск бота в режиме Long Polling...")
        await dp.start_polling(bot)

    asyncio.run(start_polling())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
