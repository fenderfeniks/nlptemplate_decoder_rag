# src/sdk/inference.py
import logging
import os
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.core.models.parsers import ResponseCleaner
from src.utils.hf_hub import download_hf_artifact


logger = logging.getLogger(__name__)


class NLPPipeline:
    """
    Универсальный SDK для работы с NLP-моделями (Классификация и Генерация).
    Скрывает внутри инициализацию Hydra, токенизаторов, моделей и постпроцессинг.
    """

    MODELS = {
        "spam": {
            "hf_repo_id": "tvoi_nik/spam-classifier",
            "checkpoint_path": "best.ckpt",
            "task": "classification",
        },
        "script_generator": {
            "hf_repo_id": "tvoi_nik/video-script-llama",
            "checkpoint_path": "adapter_model",  # Указываем папку для LoRA
            "task": "generation",
        },
    }

    def __init__(
        self,
        model_name: str | None = None,
        config_name: str = "main",
        checkpoint_path: str | None = None,
        hf_repo_id: str | None = None,
        task: str = "classification",
        cleaner_kwargs: dict[str, Any] | None = None,
    ):
        self.task = task

        if model_name and model_name in self.MODELS:
            hf_repo_id = self.MODELS[model_name]["hf_repo_id"]
            checkpoint_path = self.MODELS[model_name]["checkpoint_path"]
            self.task = self.MODELS[model_name]["task"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Инициализация NLPPipeline ({self.task}) на устройстве: {self.device}")

        # 1. Загрузка конфигурации Hydra
        config_dir = str(Path(__file__).resolve().parents[2] / "configs")
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            self.cfg = compose(config_name=config_name)
            OmegaConf.resolve(self.cfg)

        self.max_length = self.cfg.data.max_length

        # 2. Инициализация базовой архитектуры
        self.tokenizer = instantiate(self.cfg.model.tokenizer).build()
        model_builder = instantiate(self.cfg.model.builder, tokenizer=self.tokenizer)
        self.model = model_builder.build()

        # 3. Загрузка кастомных весов
        if checkpoint_path:
            if hf_repo_id and not os.path.exists(checkpoint_path):
                # Если передан файл, скачиваем его
                checkpoint_path = download_hf_artifact(
                    repo_id=hf_repo_id, filename=checkpoint_path, local_dir="./models/downloaded"
                )
            self._load_weights(checkpoint_path)

        self.model.to(self.device)
        self.model.eval()

        # 4. Инициализация клинера для генерации
        if self.task == "generation":
            cleaner_kwargs = cleaner_kwargs or {}
            self.cleaner = ResponseCleaner(**cleaner_kwargs)

    def _load_weights(self, path: str) -> None:
        """Умная загрузка весов (State Dict или LoRA)."""
        logger.info(f"Загрузка кастомных весов из: {path}")

        # Проверка на PEFT/LoRA адаптер
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json")):
            logger.info("Обнаружен PEFT/LoRA адаптер. Оборачиваем модель...")
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, path)
        else:
            logger.info("Загрузка обычного state_dict...")
            weight_path = os.path.join(path, "pytorch_model.bin") if os.path.isdir(path) else path
            checkpoint = torch.load(weight_path, map_location=self.device)

            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
                self.model.load_state_dict(clean_state_dict, strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)

    @torch.no_grad()
    def __call__(self, texts: str | list[str], **kwargs) -> list[dict[str, float]] | list[str]:
        """Точка входа. Перенаправляет запрос в зависимости от задачи."""
        if self.task == "classification":
            return self._predict_class(texts)
        elif self.task == "generation":
            return self._generate_text(texts, **kwargs)
        else:
            raise ValueError(f"Неизвестный тип задачи: {self.task}")

    def _predict_class(self, texts: str | list[str]) -> list[dict[str, float]]:
        """Логика классификации (напр., Spam / Not Spam)."""
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)

        if hasattr(outputs, "logits"):
            probs = torch.softmax(outputs.logits, dim=-1)
        else:
            raise ValueError("Модель не вернула logits. Проверьте конфигурацию архитектуры.")

        preds = torch.argmax(probs, dim=-1)

        results = []
        for prob, pred in zip(probs, preds, strict=False):
            results.append(
                {
                    "label_id": pred.item(),
                    "confidence": round(prob[pred].item(), 4),
                    "all_probabilities": [round(p, 4) for p in prob.cpu().tolist()],
                }
            )

        return results

    @torch.inference_mode()
    def _generate_text(self, texts: str | list[str], **kwargs) -> list[str]:
        """Логика генерации текста с поддержкой **kwargs для декодирования."""
        if isinstance(texts, str):
            texts = [texts]

        # 1. Токенизация
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
            self.device
        )

        # 2. Настройка параметров генерации
        gen_kwargs = {"max_new_tokens": 256, "temperature": 0.7, "do_sample": True}
        gen_kwargs.update(kwargs)  # Перезаписываем параметры теми, что передал пользователь

        # 3. Вызов встроенного метода генерации
        generated_ids = self.model.generate(
            **inputs,
            **gen_kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # 4. Обрезка входного промпта
        input_length = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_length:]

        # 5. Декодирование
        decoded_texts = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
        )

        # 6. Очистка ответов через ResponseCleaner
        final_responses = []
        for prompt, raw_response in zip(texts, decoded_texts, strict=False):
            cleaned_text = self.cleaner.clean(raw_text=raw_response, prompt=prompt)
            final_responses.append(cleaned_text)

        return final_responses
