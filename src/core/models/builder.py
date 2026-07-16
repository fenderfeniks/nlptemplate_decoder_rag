"""
Фабрика для загрузки архитектур моделей из Hugging Face.
Отвечает за инициализацию базовых весов, квантизацию и синхронизацию словаря.
"""

import logging
import importlib
from typing import Any, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, BitsAndBytesConfig
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

class HFModelBuilder:
    """
    Индустриальный билдер для моделей Hugging Face.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: Optional[PreTrainedTokenizerBase] = None, # <-- Принимаем токенизатор из train.py!
        auto_model_class: str = "transformers.AutoModelForCausalLM",
        cache_dir: Optional[str] = None,
        quantization_config: Optional[Any] = None, # Type Any, так как придет DictConfig
        trust_remote_code: bool = False,
        torch_dtype: str = "auto",
        peft_config: Optional[Any] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = tokenizer
        self.auto_model_class = auto_model_class
        self.cache_dir = cache_dir
        self.quantization_config = quantization_config
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype

    def build(self) -> PreTrainedModel:
        """
        Собирает базовую модель и синхронизирует ее с токенизатором.
        """
        logger.info(f"Загрузка базовой модели: {self.model_name_or_path}")

        # 1. Динамический импорт нужного класса
        module_name, class_name = self.auto_model_class.rsplit(".", 1)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        # 2. Безопасная настройка квантизации (Защита от DictConfig)
        bnb_config = None
        if self.quantization_config is not None:
            logger.info("Применение квантизации BitsAndBytes.")
            
            # Если пришел DictConfig от Гидры, превращаем его в чистый Python dict
            if isinstance(self.quantization_config, DictConfig):
                quant_dict = OmegaConf.to_container(self.quantization_config, resolve=True)
            else:
                quant_dict = self.quantization_config
                
            bnb_config = BitsAndBytesConfig(**quant_dict)

        # 3. Парсинг torch_dtype
        parsed_dtype = getattr(torch, self.torch_dtype) if self.torch_dtype != "auto" else "auto"

        # 4. Загрузка весов
        model = model_class.from_pretrained(
            self.model_name_or_path,
            cache_dir=self.cache_dir,
            quantization_config=bnb_config,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=parsed_dtype,
        )

        # 5. Синхронизация словаря
        # Если в токенизатор были добавлены новые спец-токены (например, pad_token),
        # матрица эмбеддингов модели должна быть расширена, иначе будет краш.
        if self.tokenizer is not None:
            vocab_size = len(self.tokenizer)
            if model.config.vocab_size != vocab_size:
                logger.info(
                    f"Изменение размера матрицы эмбеддингов: "
                    f"{model.config.vocab_size} -> {vocab_size}"
                )
                model.resize_token_embeddings(vocab_size)

        # 6. Обертка PEFT / LoRA
        if self.peft_config is not None:
            logger.info("Инициализация PEFT/LoRA адаптеров.")
            
            # Прячу импорты тяжелых библиотек
            # внутрь функций, чтобы скрипт не падал, если peft не установлен
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            # Если мы загрузили модель в 4-bit, ее нужно "подготовить" для LoRA
            if bnb_config is not None:
                model = prepare_model_for_kbit_training(model)

            # Безопасная распаковка DictConfig от Гидры
            if isinstance(self.peft_config, DictConfig):
                peft_dict = OmegaConf.to_container(self.peft_config, resolve=True)
            else:
                peft_dict = self.peft_config
                
            lora_config = LoraConfig(**peft_dict)
            model = get_peft_model(model, lora_config)
            
            # Выводим в логи, сколько параметров мы реально будем обучать (обычно < 1%)
            trainable_params, all_param = model.get_nb_trainable_parameters()
            logger.info(
                f"LoRA инициализирована: обучаемых параметров: {trainable_params:,d} "
                f"|| всего параметров: {all_param:,d} || "
                f"процент: {100 * trainable_params / all_param:.4f}%"
            )

        return model