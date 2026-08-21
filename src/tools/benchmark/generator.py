# src/tools/benchmark/generator.py
"""Генератор QA-пар для эталонного бенчмарка RAG.

Архитектурно отделён от LLMJudge: тот оценивает существующие ответы,
этот генерирует новые вопросы+ответы из чанка.

Поддерживает два бэкенда (переключается через конфиг):
- ``api``   — OpenRouter / любой OpenAI-совместимый эндпоинт.
              Дёшево, не грузит GPU во время нерегулярной генерации бенчмарка.
- ``local`` — локальная HF-модель из manifest["decoder_pipeline"].
              Нужен если нет доступа в интернет или требуется воспроизводимость.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Промпт-шаблон
# Требования к вопросу:
# 1. Отвечается ТОЛЬКО из чанка, без внешних знаний
# 2. Не ссылается явно на "документ" / "текст" / "автора"
# 3. Краткий точный ответ (не сочинение)
# ------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a QA dataset creator. Your task is to generate one question-answer pair "
    "from the given text chunk. The question must be answerable ONLY from the chunk — "
    "no external knowledge required. The answer must be a short factual statement "
    "directly supported by the chunk. Do NOT reference 'the text', 'the document', "
    "or 'the author'. Respond ONLY with a JSON object, no markdown:\n"
    '{"question": "<question>", "answer": "<answer>"}'
)

_USER_TMPL = "TEXT CHUNK:\n{chunk_text}"


class BaseQAGenerator(ABC):
    """Абстрактный генератор QA-пар."""

    @abstractmethod
    def generate(self, chunk_text: str) -> tuple[str, str] | None:
        """Генерирует (question, answer) из чанка.

        Returns:
            Кортеж (question, answer) или None при ошибке парсинга/генерации.
        """
        ...

    def generate_batch(
        self, chunk_texts: list[str]
    ) -> list[tuple[str, str] | None]:
        """Генерирует QA для списка чанков. Дефолт — sequential."""
        return [self.generate(text) for text in chunk_texts]


class APIQAGenerator(BaseQAGenerator):
    """Генератор через OpenRouter / OpenAI-совместимый API.

    Параметры конфигурируются через Hydra:
    ``configs/evaluation/benchmark/generator/api.yaml``
    """

    def __init__(
        self,
        model: str,
        api_key_env: str = "OPENROUTER_API_KEY",
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.3,
        max_tokens: int = 256,
        requests_per_minute: int = 60,
        retry_attempts: int = 3,
        retry_delay: float = 5.0,
        system_prompt: str = _SYSTEM_PROMPT,
        user_template: str = _USER_TMPL,
    ) -> None:
        from openai import OpenAI

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise OSError(
                f"Переменная окружения '{api_key_env}' не задана. "
                "Добавьте её в .env или secrets."
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.min_interval = 60.0 / requests_per_minute
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.system_prompt = system_prompt
        self.user_template = user_template
        self._last_request_time: float = 0.0

        logger.info("APIQAGenerator: model=%s, url=%s", model, base_url)

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        wait = self.min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.monotonic()

    def _call_api(self, chunk_text: str) -> str:
        user_msg = self.user_template.format(chunk_text=chunk_text)
        for attempt in range(1, self.retry_attempts + 1):
            try:
                self._rate_limit()
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                logger.warning(
                    "APIQAGenerator: API ошибка (попытка %d/%d): %s",
                    attempt, self.retry_attempts, e,
                )
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_delay * attempt)
                else:
                    raise
        return ""

    @staticmethod
    def _parse(raw: str) -> tuple[str, str] | None:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            data = json.loads(cleaned)
            question = str(data.get("question", "")).strip()
            answer = str(data.get("answer", "")).strip()
            if question and answer:
                return question, answer
        except json.JSONDecodeError:
            logger.warning("APIQAGenerator: не удалось распарсить JSON: %r", raw[:200])
        return None

    def generate(self, chunk_text: str) -> tuple[str, str] | None:
        try:
            raw = self._call_api(chunk_text)
            return self._parse(raw)
        except Exception as e:
            logger.error("APIQAGenerator: сбой генерации: %s", e)
            return None


class LocalQAGenerator(BaseQAGenerator):
    """Генератор через локальную HF-модель (decoder/instruction-tuned).

    Принимает готовый HF text-generation pipeline снаружи.
    Загрузка модели — через фабричный метод from_manifest.

    Инициализация:
        # Через готовый pipeline (инъекция снаружи):
        gen = LocalQAGenerator(pipeline=pipe)

        # Через манифест (самостоятельная загрузка):
        gen = LocalQAGenerator.from_manifest(router, manifest_uri, cache_base, gen_cfg)
    """

    def __init__(
        self,
        pipeline: Any,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        do_sample: bool = True,
        system_prompt: str = _SYSTEM_PROMPT,
        user_template: str = _USER_TMPL,
    ) -> None:
        self._pipeline = pipeline
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.system_prompt = system_prompt
        self.user_template = user_template
        logger.info("LocalQAGenerator: готов.")

    # ------------------------------------------------------------------
    # Фабричный метод — загрузка через манифест
    # ------------------------------------------------------------------

    @classmethod
    def from_manifest(
        cls,
        router,
        manifest_uri: str,
        cache_base: Path,
        gen_cfg: dict | None = None,
    ) -> "LocalQAGenerator":
        """Загружает decoder LLM из manifest["decoder_pipeline"] через HFModelBuilder.

        Args:
            router:       StorageRouter.
            manifest_uri: URI единого манифеста (system.manifest.uri).
            cache_base:   Локальная директория для кэша весов.
            gen_cfg:      Словарь параметров из конфига генератора
                          (torch_dtype, device_map, max_new_tokens, temperature, ...).
                          Если None — используются дефолты.

        Raises:
            KeyError:   Если "decoder_pipeline" не найден в манифесте.
            ValueError: Если load_type != "full_model".

        Returns:
            Готовый LocalQAGenerator с загруженным pipeline.
        """
        import torch
        from transformers import AutoTokenizer
        from transformers import pipeline as hf_pipeline

        from src.pipelines.base.core.models.builder import HFModelBuilder

        cfg = gen_cfg or {}

        logger.info("LocalQAGenerator: загрузка из манифеста '%s'", manifest_uri)
        full_manifest = router.download_manifest(
            manifest_uri, cache_base / "decoder_manifest"
        )

        pipeline_key = "decoder_pipeline"
        if pipeline_key not in full_manifest:
            raise KeyError(
                f"Пайплайн '{pipeline_key}' не найден в манифесте {manifest_uri}. "
                "Запустите prepare_artifacts.py pipeline_name=decoder_pipeline"
            )

        manifest = full_manifest[pipeline_key]
        if manifest.get("load_type") != "full_model":
            raise ValueError(
                f"Decoder-модель ожидает load_type=full_model, "
                f"получено: {manifest.get('load_type')}."
            )

        model_path = router.download_from_uri(
            manifest["model_uri"], cache_base / "decoder_model"
        )
        logger.info("LocalQAGenerator: веса получены из storage: %s", model_path)

        builder = HFModelBuilder(
            model_name_or_path=str(model_path),
            auto_model_class="transformers.AutoModelForCausalLM",
            torch_dtype=cfg.get("torch_dtype", "bfloat16"),
            device_map=cfg.get("device_map", "auto_cuda"),
            trust_remote_code=cfg.get("trust_remote_code", False),
            attn_implementation=None,
        )
        model = builder.build()

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if getattr(tokenizer, "chat_template", None) is None:
            logger.warning("chat_template не найден в конфигурации. Устанавливаем дефолтный ChatML.")
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}"
                "{% endfor %}"
                "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
            )
        # --------------------------

        pipe = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        logger.info("LocalQAGenerator: pipeline готов.")

        return cls(
            pipeline=pipe,
            max_new_tokens=cfg.get("max_new_tokens", 256),
            temperature=cfg.get("temperature", 0.3),
            do_sample=cfg.get("do_sample", True),
            system_prompt=cfg.get("system_prompt") or _SYSTEM_PROMPT,
            user_template=cfg.get("user_template") or _USER_TMPL,
        )

    # ------------------------------------------------------------------
    # Генерация
    # ------------------------------------------------------------------

    def _build_messages(self, chunk_text: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.format(chunk_text=chunk_text)},
        ]

    @staticmethod
    def _parse(raw: str) -> tuple[str, str] | None:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            data = json.loads(cleaned)
            question = str(data.get("question", "")).strip()
            answer = str(data.get("answer", "")).strip()
            if question and answer:
                return question, answer
        except json.JSONDecodeError:
            logger.warning("LocalQAGenerator: не удалось распарсить JSON: %r", raw[:200])
        return None

    def generate(self, chunk_text: str) -> tuple[str, str] | None:
        messages = self._build_messages(chunk_text)
        try:
            outputs = self._pipeline(
                messages,
                max_new_tokens=self.max_new_tokens,
                max_length=None,
                clean_up_tokenization_spaces=False,
                temperature=self.temperature,
                do_sample=self.do_sample,
                return_full_text=False,
            )
            raw = outputs[0]["generated_text"].strip()
            return self._parse(raw)
        except Exception as e:
            logger.error("LocalQAGenerator: сбой генерации: %s", e)
            return None