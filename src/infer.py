import logging
import os

from dotenv import load_dotenv


# Загружаем окружение до Гидры!
load_dotenv()

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from src.utils.hydra_utils import setup_config  # noqa: E402


logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="main", version_base="1.3")  # <-- Исправлен путь
def infer(cfg: DictConfig) -> None:
    setup_config(cfg)

    logger.info("Загрузка токенизатора...")
    tokenizer = hydra.utils.instantiate(cfg.model.tokenizer).build()

    # 1. Загружаем БАЗОВУЮ архитектуру
    logger.info(f"Загрузка базовой модели: {cfg.model.builder.model_name_or_path}")
    model = hydra.utils.instantiate(cfg.model.builder, tokenizer=tokenizer).build()

    # 2. Безопасная загрузка обученных весов (LoRA или full fine-tune)
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path:
        logger.info(f"Подгрузка кастомных весов из: {ckpt_path}")
        
        # ИСПРАВЛЕНИЕ: Проверяем тип чекпоинта по наличию adapter_config.json
        if os.path.isdir(ckpt_path) and os.path.exists(os.path.join(ckpt_path, "adapter_config.json")):
            logger.info("Обнаружен PEFT/LoRA адаптер. Оборачиваем модель...")
            # Импортируем peft только если он действительно нужен
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, ckpt_path)
        else:
            logger.warning("adapter_config.json не найден. Попытка загрузки как state_dict...")
            import torch
            try:
                # Если передан путь к папке, ищем стандартный бинарник, иначе грузим сам файл
                weight_path = os.path.join(ckpt_path, "pytorch_model.bin") if os.path.isdir(ckpt_path) else ckpt_path
                
                # Грузим веса на CPU, чтобы не получить спайк VRAM перед стартом
                state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict, strict=False)
                logger.info("Веса state_dict успешно загружены.")
            except Exception as e:
                logger.error(f"Не удалось загрузить чекпоинт. Ошибка: {e}")
                raise e

    logger.info("Инициализация генератора текста...")
    generator = hydra.utils.instantiate(cfg.model.generation, model=model, tokenizer=tokenizer)

    query = cfg.get("text", "Напиши пример кода на Python.")
    logger.info(f"Входящий запрос: {query}")

    responses = generator.generate(query)

    print("\n" + "=" * 50)
    print(f"Ответ модели:\n{responses[0]}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    infer()