

@hydra.main(config_path="configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig):
    
    # 1. Создаем БЕЗОПАСНЫЙ токенизатор через нашу фабрику
    tokenizer_builder = instantiate(cfg.model.tokenizer)
    tokenizer = tokenizer_builder.build()
    
    # 2. Передаем готовый токенизатор в слой данных
    datamodule = instantiate(
        cfg.data.datamodule, 
        data_cfg=cfg.data, 
        tokenizer=tokenizer
    )
    
    # 3. Передаем тот же токенизатор в сборщик модели
    model = instantiate(
        cfg.model.builder, 
        model_cfg=cfg.model, 
        tokenizer=tokenizer
    )