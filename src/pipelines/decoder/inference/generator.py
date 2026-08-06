# src/pipelines/decoder/inference/generator.py
"""Модуль для текстовой генерации (Inference).

Оборачивает логику декодирования, сэмплирования и очистки токенов.
"""

import logging
from collections.abc import Iterator
from threading import Thread
from typing import Any

import hydra
import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextIteratorStreamer,
)

from src.pipelines.decoder.inference.response_cleaner import ResponseCleaner


logger = logging.getLogger(__name__)


class HFTextGenerator:
    """Обёртка над моделями Hugging Face для батч-генерации и стриминга."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        generation_kwargs: dict[str, Any],
        cleaner_cfg: Any = None,
    ) -> None:
        """
        Args:
            model: Загруженная модель HF.
            tokenizer: Токенизатор для декодирования и паддинга.
            generation_kwargs: Базовые аргументы генерации
                (``temperature``, ``top_p``, ``max_new_tokens`` и т.д.).
                Переопределяются per-call через ``**kwargs`` в ``generate``.
            cleaner_cfg: Hydra DictConfig для инстанцирования ``ResponseCleaner``.
                ``None`` -> создаётся ``ResponseCleaner()`` с дефолтными параметрами.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.generation_kwargs = generation_kwargs

        # padding_side='left' обязателен для батч-генерации:
        # при 'right' паддинге модель "видит" pad-токены перед EOS и генерирует мусор
        if self.tokenizer.padding_side != "left":
            logger.warning(
                "padding_side='%s' -> принудительно устанавливаем 'left' "
                "для корректной батч-генерации.",
                self.tokenizer.padding_side,
            )
            self.tokenizer.padding_side = "left"

        self.cleaner = hydra.utils.instantiate(cleaner_cfg) if cleaner_cfg else ResponseCleaner()
        self.model.eval()

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _merge_kwargs(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Мержит глобальные kwargs из конфига с per-call переопределениями."""
        merged = self.generation_kwargs.copy()
        merged.update(overrides)
        return merged

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Токенизирует список промптов и переносит на устройство модели."""
        device = next(self.model.parameters()).device
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).to(device)

    def _generate_in_thread(self, generation_args: dict[str, Any]) -> None:
        """Запускает ``model.generate`` в фоновом потоке с активным inference_mode.

        ``@torch.inference_mode()`` на вызывающем методе не распространяется
        на дочерние потоки — контекст нужно активировать явно внутри потока.
        """
        with torch.inference_mode():
            self.model.generate(**generation_args)

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(self, texts: str | list[str], **kwargs: Any) -> list[str]:
        """Генерирует ответы для одного промпта или батча.

        Args:
            texts: Один промпт или список промптов.
            **kwargs: Переопределения параметров генерации для этого вызова
                (например, ``max_new_tokens=256``). Не мутируют ``self.generation_kwargs``.

        Returns:
            Список очищенных строк в том же порядке что и входные промпты.
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self._tokenize(texts)
        gen_kwargs = self._merge_kwargs(kwargs)

        generated_ids = self.model.generate(
            **inputs,
            **gen_kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # model.generate возвращает [input_ids + generated_ids] — срезаем промпт
        input_length = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_length:]

        # skip_special_tokens=False — оставляем спецтокены для ResponseCleaner,
        # который удаляет их в один проход вместе с остальными артефактами
        decoded = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

        return [
            self.cleaner.clean(raw_text=raw, prompt=prompt)
            for prompt, raw in zip(texts, decoded)  # noqa
        ]

    def generate_stream(self, text: str, **kwargs: Any) -> Iterator[str]:
        """Потоковая генерация текста (batch_size=1).

        Запускает ``model.generate`` в фоновом потоке с ``TextIteratorStreamer``,
        чтобы генерация не блокировала вызывающий поток.

        Args:
            text: Одиночный промпт.
            **kwargs: Переопределения параметров генерации.

        Yields:
            Строковые фрагменты по мере генерации.

        Raises:
            ValueError: Если передан список вместо строки.
        """
        if not isinstance(text, str):
            raise ValueError(
                "generate_stream поддерживает только одиночные строки (batch_size=1). "
                "Для батча используйте generate()."
            )

        inputs = self._tokenize([text])
        gen_kwargs = self._merge_kwargs(kwargs)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,  # промпт не возвращается в стрим
            skip_special_tokens=True,  # <|eot_id|>, </s> и т.п. фильтруются на лету
        )

        generation_args = {
            **inputs,
            **gen_kwargs,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }

        # inference_mode активируется явно внутри потока — декоратор на вызывающем
        # методе не распространяется на дочерние потоки (см. _generate_in_thread)
        thread = Thread(target=self._generate_in_thread, args=(generation_args,))
        thread.start()

        for new_text in streamer:
            # Не делаем strip() — токенизатор возвращает пробел перед новыми словами,
            # strip() склеил бы слова на границах фрагментов
            if new_text:
                yield new_text
