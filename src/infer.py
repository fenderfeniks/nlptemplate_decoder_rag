import logging
from dotenv import load_dotenv

# Загружаем окружение до Гидры!
load_dotenv()

import hydra
from omegaconf import DictConfig
from src.utils.hydra_utils import setup_config

logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="main", version_base="1.3") # <-- Исправлен путь
def infer(cfg: DictConfig) -> None:
    setup_config(cfg)
    
    logger.info("Загрузка токенизатора...")
    tokenizer = hydra.utils.instantiate(cfg.model.tokenizer).build()
    
    # 1. Загружаем БАЗОВУЮ архитектуру
    logger.info(f"Загрузка базовой модели: {cfg.model.builder.model_name_or_path}")
    model = hydra.utils.instantiate(cfg.model.builder, tokenizer=tokenizer).build()
    
    # 2. Если передан путь к обученным весам (LoRA или full fine-tune)
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path:
        logger.info(f"Подгрузка кастомных весов из: {ckpt_path}")
        # Здесь мы загружаем веса поверх модели. 
        # Если это LoRA, используем PeftModel:
        # from peft import PeftModel
        # model = PeftModel.from_pretrained(model, ckpt_path)
        # 
        # Если это полное обучение, builder должен уметь загружать state_dict.
        # Оставим абстракцию, но помни про этот момент!
        model.load_adapter(ckpt_path) # Пример для PEFT/LoRA
    
    logger.info("Инициализация генератора текста...")
    generator = hydra.utils.instantiate(
        cfg.model.generation, 
        model=model, 
        tokenizer=tokenizer
    )
    
    query = cfg.get("text", "Напиши пример кода на Python.")
    logger.info(f"Входящий запрос: {query}")
    
    responses = generator.generate(query)
    
    print("\n" + "="*50)
    print(f"Ответ модели:\n{responses[0]}")
    print("="*50 + "\n")

if __name__ == "__main__":
    infer()