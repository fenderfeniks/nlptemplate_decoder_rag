# src/pipelines/base/core/models/modifiers.py
import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedModel, PreTrainedTokenizerBase


logger = logging.getLogger(__name__)


class BaseModelModifier(ABC):
    """Базовый интерфейс для модификации архитектуры модели после загрузки.

    Маркерные атрибуты класса используются в ``HFModelBuilder._build_modifiers``
    для автоматической передачи runtime-аргументов без строкового матча по ``_target_``.
    Добавляя новый модификатор — укажи нужные маркеры, и builder передаст kwargs сам.

    Маркеры:
    - ``_needs_tokenizer = True``   → builder передаёт ``tokenizer=<tokenizer>``
    - ``_needs_lora_path = True``   → builder передаёт ``lora_resume_path=<path|None>``
    """

    _needs_tokenizer: bool = False
    _needs_lora_path: bool = False

    @abstractmethod
    def __call__(self, model: PreTrainedModel) -> PreTrainedModel:
        pass


class EmbeddingResizeModifier(BaseModelModifier):
    """Синхронизирует размер матрицы эмбеддингов с размером словаря токенизатора.

    Применяется когда в токенизатор добавлены новые специальные токены
    (например, для нового языка или задачи), и embedding-матрица модели
    не покрывает весь словарь.

    Новые строки инициализируются средним значением существующих эмбеддингов —
    это устойчивый к масштабу старт, лучше нулей и случайной инициализации.
    """

    _needs_tokenizer = True

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, pad_to_multiple_of: int | None = 8
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, model: PreTrainedModel) -> PreTrainedModel:
        vocab_size = len(self.tokenizer)
        old_vocab_size = model.config.vocab_size

        if old_vocab_size == vocab_size:
            logger.info("Размер словаря совпадает (%d). Ресайз не требуется.", vocab_size)
            return model

        logger.warning(
            "Изменение размера матрицы эмбеддингов: %d → %d. Увеличение потребления VRAM.",
            old_vocab_size,
            vocab_size,
        )

        # pad_to_multiple_of=8 выравнивает размер словаря до кратного 8 —
        # это ускоряет матричные операции на Tensor Core (NVIDIA Ampere+).
        model.resize_token_embeddings(vocab_size, pad_to_multiple_of=self.pad_to_multiple_of)

        if vocab_size > old_vocab_size:
            # Инициализируем новые токены средним по существующим эмбеддингам.
            # Использует no_grad() явно — resize_token_embeddings не всегда
            # гарантирует отсутствие градиентов для новых строк.
            with torch.no_grad():
                input_emb = model.get_input_embeddings()
                w = input_emb.weight.data
                mean_vec = w[:old_vocab_size].mean(dim=0, keepdim=True)
                w[old_vocab_size:] = mean_vec

                output_emb = model.get_output_embeddings()
                if output_emb is not None and output_emb is not input_emb:
                    # Проверяем что lm_head — отдельная матрица (не weight-tied)
                    ow = output_emb.weight.data
                    out_mean = ow[:old_vocab_size].mean(dim=0, keepdim=True)
                    ow[old_vocab_size:] = out_mean

        logger.info(
            "Ресайз завершён. Новый vocab_size модели: %d",
            model.config.vocab_size,
        )
        return model


class PEFTModifier(BaseModelModifier):
    """Подготавливает модель и применяет LoRA-адаптеры через PEFT."""

    _needs_lora_path = True

    def __init__(
        self,
        peft_config: Any,
        lora_resume_path: str | None = None,
        gradient_checkpointing: bool = True,
        is_quantized: bool = True,
    ) -> None:
        """
        Args:
            peft_config: DictConfig или LoraConfig с параметрами адаптера.
            lora_resume_path: Путь к сохранённому PEFT-адаптеру для продолжения
                обучения. ``None`` → инициализируется новый адаптер.
            gradient_checkpointing: Включить gradient checkpointing для экономии VRAM.
            is_quantized: Если True — вызывает ``prepare_model_for_kbit_training``
                перед применением LoRA (обязательно для 4bit/8bit моделей).
        """
        self.peft_config = peft_config
        self.lora_resume_path = lora_resume_path
        self.gradient_checkpointing = gradient_checkpointing
        self.is_quantized = is_quantized

    def __call__(self, model: PreTrainedModel) -> PreTrainedModel:
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

        if self.is_quantized:
            # prepare_model_for_kbit_training замораживает базовые веса,
            # кастит LayerNorm в fp32 и включает checkpointing если нужно
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=self.gradient_checkpointing,
            )
        elif self.gradient_checkpointing:
            # use_reentrant=False обязателен с PyTorch >= 2.1 + PEFT:
            # reentrant-режим несовместим с LoRA hooks и даёт некорректные градиенты
            # для modules_to_save.
            model.gradient_checkpointing_enable({"use_reentrant": False})

        if self.lora_resume_path is not None:
            logger.info("PEFT: загрузка адаптера из '%s'", self.lora_resume_path)
            model = PeftModel.from_pretrained(model, self.lora_resume_path, is_trainable=True)
        else:
            logger.info("PEFT: инициализация нового LoRA-адаптера.")
            if isinstance(self.peft_config, LoraConfig):
                lora_config = self.peft_config
            else:
                peft_dict = (
                    OmegaConf.to_container(self.peft_config, resolve=True)
                    if isinstance(self.peft_config, DictConfig)
                    else dict(self.peft_config)
                )
                lora_config = LoraConfig(**peft_dict)

            model = get_peft_model(model, lora_config)

        trainable, all_param = model.get_nb_trainable_parameters()
        logger.info(
            "LoRA: %d обучаемых из %d параметров (%.4f%%)",
            trainable,
            all_param,
            100 * trainable / all_param,
        )
        return model


class FullFinetuningModifier(BaseModelModifier):
    """Подготавливает модель для полного дообучения (Full Fine-Tuning).

    Размораживает все параметры и включает gradient checkpointing.
    """

    def __init__(self, gradient_checkpointing: bool = True) -> None:
        """
        Args:
            gradient_checkpointing: Включить gradient checkpointing.
                Существенно снижает потребление VRAM при минимальном overhead по времени.
        """
        self.gradient_checkpointing = gradient_checkpointing

    def __call__(self, model: PreTrainedModel) -> PreTrainedModel:
        if self.gradient_checkpointing:
            logger.info("Full FT: активация gradient checkpointing.")
            model.gradient_checkpointing_enable({"use_reentrant": False})

        # Принудительно размораживаем все веса — на случай если базовый класс
        # или предыдущий модификатор их заморозил
        for param in model.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Full FT: %d обучаемых параметров (100%%).", trainable)
        return model
