# src/tools/evaluation/judges/llm_judge.py
"""LLM-as-a-Judge через OpenRouter (OpenAI-compatible API)."""

from __future__ import annotations

import logging
import os
import re
import time
from string import Template

from openai import OpenAI

from src.tools.evaluation.judges.base import BaseJudge
from src.tools.evaluation.schema import EvalInput, EvalResult


logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Дефолтный промпт-шаблон
# Поддерживает: $prompt, $response, $reference
# ------------------------------------------------------------------
_DEFAULT_SYSTEM = "You are an expert evaluator. Be concise and objective."

_DEFAULT_USER_TMPL = """\
### Task
Evaluate the following model response.

### User Prompt
$prompt

### Model Response
$response
${reference_block}
### Instructions
${score_instruction}
${reasoning_instruction}
${verdict_instruction}

Respond strictly in this JSON format (no markdown):
{"score": <float|null>, "verdict": <true|false|null>, "reasoning": <str|null>}
"""

_SCORE_INSTR = "- `score`: float from $min_score to $max_score ($min_score=worst, $max_score=best)."
_VERDICT_INSTR = "- `verdict`: true if the response is acceptable, false otherwise."
_REASONING_INSTR = "- `reasoning`: one sentence explaining your decision."
_REFERENCE_BLOCK = "### Reference Answer\n$reference\n"


class LLMJudge(BaseJudge):
    """Judge на базе LLM через OpenRouter.

    Конфигурируется полностью через Hydra — см. configs/evaluation/judge/openrouter.yaml.

    Что возвращает — управляется флагами:
    - ``return_score=True``     → EvalResult.score
    - ``return_reasoning=True`` → EvalResult.reasoning
    - ``return_verdict=True``   → EvalResult.verdict

    Параметры rate-limiting (``requests_per_minute``, ``retry_attempts``) защищают
    от 429 при батчевой оценке.
    """

    def __init__(
        self,
        model: str,
        api_key_env: str = "OPENROUTER_API_KEY",
        base_url: str = "https://openrouter.ai/api/v1",
        # --- Что возвращать ---
        return_score: bool = True,
        return_reasoning: bool = False,
        return_verdict: bool = False,
        min_score: float = 1.0,
        max_score: float = 5.0,
        # --- Промпт ---
        system_prompt: str = _DEFAULT_SYSTEM,
        user_prompt_template: str | None = None,
        # --- Генерация ---
        temperature: float = 0.0,
        max_tokens: int = 256,
        # --- Rate limiting ---
        requests_per_minute: int = 60,
        retry_attempts: int = 3,
        retry_delay: float = 5.0,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise OSError(
                f"Переменная окружения '{api_key_env}' не задана. Добавьте её в .env файл."
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.return_score = return_score
        self.return_reasoning = return_reasoning
        self.return_verdict = return_verdict
        self.min_score = min_score
        self.max_score = max_score
        self.system_prompt = system_prompt
        self.user_tmpl = user_prompt_template or _DEFAULT_USER_TMPL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.min_interval = 60.0 / requests_per_minute
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self._last_request_time: float = 0.0

        logger.info(
            "LLMJudge: model=%s, score=%s, reasoning=%s, verdict=%s",
            model,
            return_score,
            return_reasoning,
            return_verdict,
        )

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _build_prompt(self, inp: EvalInput) -> str:
        """Рендерит пользовательский промпт через string.Template."""
        reference_block = (
            Template(_REFERENCE_BLOCK).substitute(reference=inp.reference) if inp.reference else ""
        )
        score_instruction = (
            Template(_SCORE_INSTR).substitute(min_score=self.min_score, max_score=self.max_score)
            if self.return_score
            else ""
        )
        return Template(self.user_tmpl).substitute(
            prompt=inp.prompt,
            response=inp.response,
            reference_block=reference_block,
            score_instruction=score_instruction,
            reasoning_instruction=_REASONING_INSTR if self.return_reasoning else "",
            verdict_instruction=_VERDICT_INSTR if self.return_verdict else "",
        )

    def _rate_limit(self) -> None:
        """Простой rate limiter: выдерживает минимальный интервал между запросами."""
        elapsed = time.monotonic() - self._last_request_time
        wait = self.min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.monotonic()

    def _call_api(self, user_prompt: str) -> str:
        """Вызывает API с retry при ошибках."""
        for attempt in range(1, self.retry_attempts + 1):
            try:
                self._rate_limit()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.warning(
                    "LLMJudge API ошибка (попытка %d/%d): %s",
                    attempt,
                    self.retry_attempts,
                    e,
                )
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_delay * attempt)
                else:
                    raise

        return ""  # unreachable

    def _parse_response(self, raw: str) -> tuple[float | None, bool | None, str | None]:
        """Парсит JSON-ответ judge. Устойчив к markdown-обёрткам."""
        import json

        # Убираем ```json ... ``` если модель решила добавить
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("LLMJudge: не удалось распарсить JSON: %r", raw[:200])
            return None, None, None

        score = data.get("score")
        verdict = data.get("verdict")
        reasoning = data.get("reasoning")

        # Нормализация score в [0, 1] относительно заданного диапазона
        if score is not None:
            try:
                score = float(score)
                score = (score - self.min_score) / (self.max_score - self.min_score)
                score = max(0.0, min(1.0, score))
            except (TypeError, ValueError):
                score = None

        if verdict is not None and not isinstance(verdict, bool):
            verdict = str(verdict).lower() in ("true", "1", "yes", "pass")

        return score, verdict, reasoning

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    def evaluate_batch(self, inputs: list[EvalInput]) -> list[EvalResult]:
        results = []
        for inp in inputs:
            user_prompt = self._build_prompt(inp)
            try:
                raw = self._call_api(user_prompt)
                score, verdict, reasoning = self._parse_response(raw)
            except Exception as e:
                logger.error("LLMJudge: сбой для примера '%s...': %s", inp.prompt[:50], e)
                raw = str(e)
                score, verdict, reasoning = None, None, None

            results.append(
                EvalResult(
                    score=score if self.return_score else None,
                    verdict=verdict if self.return_verdict else None,
                    reasoning=reasoning if self.return_reasoning else None,
                    raw=raw,
                    metadata=inp.metadata,
                )
            )
        return results
