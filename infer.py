import hydra
from omegaconf import DictConfig
from src.core.utils.hydra_utils import setup_config

@hydra.main(config_path="configs", config_name="main", version_base="1.3")
def infer(cfg: DictConfig) -> None:
    setup_config(cfg)
    
    # 1. Загрузка компонентов
    tokenizer = hydra.utils.instantiate(cfg.model.tokenizer).build()
    model = hydra.utils.instantiate(cfg.model.builder, tokenizer=tokenizer).build()
    
    # 2. Инициализация генератора
    generator = hydra.utils.instantiate(
        cfg.model.generation, 
        model=model, 
        tokenizer=tokenizer
    )
    
    # 3. Пример использования
    text = "Пример запроса к модели"
    response = generator.generate(text)
    print(f"Ответ модели: {response}")

if __name__ == "__main__":
    infer()