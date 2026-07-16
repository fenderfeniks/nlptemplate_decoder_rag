import hydra
import logging
from omegaconf import DictConfig
from src.utils.hydra_utils import setup_config

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="main", version_base="1.3")
def infer(cfg: DictConfig) -> None:
    setup_config(cfg)
    
    # 1. Загрузка токенизатора и модели
    logger.info("Загрузка модели для инференса...")
    tokenizer = hydra.utils.instantiate(cfg.model.tokenizer).build()
    
    # Если передали ckpt_path в консоли, билдер должен подхватить эти веса,
    # иначе загрузит базовые из HuggingFace
    ckpt_path = cfg.get("ckpt_path", cfg.model.builder.model_name_or_path)
    cfg.model.builder.model_name_or_path = ckpt_path
    
    model = hydra.utils.instantiate(cfg.model.builder, tokenizer=tokenizer).build()
    
    # 2. Инициализация генератора текста
    generator = hydra.utils.instantiate(
        cfg.model.generation, 
        model=model, 
        tokenizer=tokenizer
    )
    
    # 3. Обработка запроса
    # Мы можем передать text через консоль: python infer.py text="Привет!"
    query = cfg.get("text", "Напиши пример кода на Python.")
    logger.info(f"Входящий запрос: {query}")
    
    # Генерируем ответ
    responses = generator.generate(query)
    
    print("\n" + "="*50)
    print(f"Ответ модели:\n{responses[0]}")
    print("="*50 + "\n")

if __name__ == "__main__":
    infer()