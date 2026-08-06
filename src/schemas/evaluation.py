# src/schemas/evaluation.py
from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING


@dataclass
class LLMJudgeConfig:
    """Конфиг для LLM-as-a-Judge через OpenRouter."""

    _target_: str = "src.tools.evaluation.judges.llm_judge.LLMJudge"
    model: str = "google/gemini-flash-1.5"
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"
    return_score: bool = True
    return_reasoning: bool = False
    return_verdict: bool = False
    min_score: float = 1.0
    max_score: float = 5.0
    temperature: float = 0.0
    max_tokens: int = 256
    requests_per_minute: int = 60
    retry_attempts: int = 3
    retry_delay: float = 5.0
    system_prompt: str | None = None
    user_prompt_template: str | None = None


@dataclass
class NLIJudgeConfig:
    """Конфиг для NLI-Judge через локальную RoBERTa-модель."""

    _target_: str = "src.tools.evaluation.judges.nli_judge.NLIJudge"
    manifest_uri: str = "${storage.uri_prefix}manifests/nli_pipeline_manifest.json"
    router: Any = "${storage_router}"
    cache_dir: str = "${paths.data_dir}/nli_cache"
    tokenizer_name: str | None = None
    device: str = "auto"
    batch_size: int = 32
    max_length: int = 512
    entailment_label: str = "entailment"
    verdict_threshold: float = 0.5
    return_score: bool = True
    return_verdict: bool = True
    return_reasoning: bool = False


@dataclass
class EvaluationConfig:
    """Корневой узел evaluation.

    cfg.evaluation.judge инстанциируется в callback через hydra.utils.instantiate().
    Переключение между LLM-judge и NLI-judge — смена конфиг-группы в main.yaml:
        - evaluation/judge: openrouter   (LLMJudgeConfig)
        - evaluation/judge: nli          (NLIJudgeConfig)

    judge намеренно MISSING а не конкретный датакласс — иначе OmegaConf фиксирует
    схему по дефолту и падает при merge с альтернативным конфигом (KeyError на
    полях которых нет в зафиксированном датаклассе).
    """

    judge: Any = MISSING
