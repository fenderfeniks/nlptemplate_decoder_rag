# src/pipelines/base/core/models/builder.py
import importlib
import logging
from typing import Any
import os
import boto3
from urllib.parse import urlparse
from pathlib import Path
import torch
from hydra._internal.utils import _locate
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from transformers import BitsAndBytesConfig, PreTrainedModel


logger = logging.getLogger(__name__)

# Допустимые строковые значения torch_dtype
_TORCH_FLOAT_DTYPES = frozenset(
    {
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "half",
        "float",
        "double",
    }
)


class HFModelBuilder:
    """Индустриальная фабрика для загрузки базовых моделей Hugging Face.

    Поддерживает:
    - Произвольный AutoModel-класс через строку ``auto_model_class``.
    - BitsAndBytes квантизацию (4bit / 8bit) с автоматическим парсингом dtype.
    - Flash Attention 2 с автоматическим fallback на SDPA.
    - Постзагрузочные модификаторы (ресайз эмбеддингов, LoRA, Full FT).
    - RoPE scaling для длинных контекстов.

    Модификаторы обнаруживают нужные runtime-аргументы через маркерные атрибуты
    класса (``_needs_tokenizer``, ``_needs_lora_path``) — без строкового матча
    по ``_target_``. Добавляя новый модификатор — укажи нужные маркеры в классе.
    """

    def __init__(
        self,
        model_name_or_path: str,
        auto_model_class: str = "transformers.AutoModel",
        cache_dir: str | None = None,
        quantization_config: Any | None = None,
        trust_remote_code: bool = False,
        torch_dtype: str = "auto",
        attn_implementation: str | None = "flash_attention_2",
        rope_scaling: dict[str, Any] | None = None,
        device_map: str | dict | None = "auto_cuda",
        # modifiers — DictConfig из model.modifiers (embedding_resize, finetuning, ...).
        # Не список: Hydra мержит подгруппы как dict с ключами по имени файла подгруппы.
        # _build_modifiers() итерирует .items() в порядке defaults из model/default.yaml.
        modifiers: Any | None = None,
    ) -> None:
        """
        Args:
            model_name_or_path: HF Hub id или локальный путь к модели.
            auto_model_class: Полное имя класса для загрузки, например
                ``'transformers.AutoModelForCausalLM'`` или ``'transformers.AutoModel'``.
            cache_dir: Директория кэша HF (по умолчанию ``~/.cache/huggingface``).
            quantization_config: DictConfig или BitsAndBytesConfig для квантизации.
                ``None`` или пустой dict — без квантизации.
            trust_remote_code: Доверять ли коду из репозитория модели.
            torch_dtype: Тип весов — ``'auto'``, ``'float16'``, ``'bfloat16'`` и т.д.
            attn_implementation: ``'flash_attention_2'``, ``'sdpa'``, ``'eager'`` или ``None``.
                При ``'flash_attention_2'`` — автоматический fallback на ``'sdpa'``.
            rope_scaling: Словарь параметров RoPE scaling (опционально).
            device_map: Стратегия размещения модели по девайсам.
                ``'auto_cuda'`` (дефолт) -> ``{"": current_device}`` при наличии CUDA,
                ``None`` при CPU/MPS. ``'auto'`` -> HF авто-шардирование (multi-GPU).
                Передай явный dict для полного контроля.
            modifiers: DictConfig с модификаторами постзагрузки.
        """
        self.model_name_or_path = model_name_or_path
        self.auto_model_class = auto_model_class
        self.cache_dir = cache_dir
        self.quantization_config = quantization_config
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.rope_scaling = rope_scaling
        self.device_map = device_map
        self.modifiers_cfg = modifiers
        # Устанавливается снаружи после резолва пути из MLflow или манифеста.
        self.lora_resume_path: str | None = None
        # True только при full_model инференсе — пропускает PEFTModifier целиком.
        # При обучении всегда False: lora_resume_path=None означает новый адаптер.

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_attn_implementation(requested: str | None) -> str | None:
        """Выбирает реализацию внимания с автоматическим fallback.

        Flash Attention 2 требует:
        - пакет ``flash-attn`` (``pip install flash-attn --no-build-isolation``)
        - CUDA compute capability >= 8.0 (A100, H100, RTX 30xx+)

        Если CUDA недоступна совсем (CPU / MPS) -> ``None``.
        Если железо не тянет FA2 -> ``'sdpa'``.
        """
        if requested != "flash_attention_2":
            return requested

        try:
            import flash_attn  # noqa: F401
        except ImportError:
            logger.warning(
                "flash-attn не установлен -> откат на attn_implementation='sdpa'. "
                "Установите: pip install flash-attn --no-build-isolation"
            )
            return "sdpa"

        if not torch.cuda.is_available():
            logger.warning(
                "CUDA недоступна (CPU или MPS) -> attn_implementation сбрасывается в None."
            )
            return None

        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            logger.warning(
                "GPU compute capability %d.x < 8.0 — Flash Attention 2 не поддерживается "
                "-> откат на 'sdpa'.",
                major,
            )
            return "sdpa"

        logger.info("Flash Attention 2: железо и пакет совместимы.")
        return "flash_attention_2"

    @staticmethod
    def _parse_torch_dtype(torch_dtype: str) -> Any:
        """Резолвит строку dtype в объект ``torch.*``.

        Raises:
            ValueError: Если строка не является допустимым torch dtype.
        """
        if torch_dtype == "auto":
            return "auto"
        if torch_dtype not in _TORCH_FLOAT_DTYPES:
            raise ValueError(
                f"Недопустимое значение torch_dtype: '{torch_dtype}'. "
                f"Допустимые: 'auto', {sorted(_TORCH_FLOAT_DTYPES)}."
            )
        return getattr(torch, torch_dtype)

    @staticmethod
    def _parse_quantization_config(quantization_config: Any) -> Any:
        from transformers import AwqConfig, GPTQConfig

        if quantization_config is None:
            return None

        # Уже готовый объект — возвращаем как есть
        if isinstance(quantization_config, (BitsAndBytesConfig, AwqConfig, GPTQConfig)):
            return quantization_config

        quant_dict = (
            OmegaConf.to_container(quantization_config, resolve=True)
            if isinstance(quantization_config, DictConfig)
            else dict(quantization_config)
        )

        if not quant_dict:
            return None          # none.yaml — пустой dict

        # Определяем тип по _target_
        target = quant_dict.pop("_target_", None)

        if target and "Awq" in target:
            return AwqConfig(**quant_dict)

        if target and "GPTQ" in target:
            return GPTQConfig(**quant_dict)

        # Без _target_ — считаем BnB (обратная совместимость)
        for k, v in quant_dict.items():
            if k.endswith("_dtype") and isinstance(v, str):
                quant_dict[k] = getattr(torch, v)
        return BitsAndBytesConfig(**quant_dict)

    def _resolve_device_map(self, bnb_config: BitsAndBytesConfig | None) -> dict | str | None:
        """Резолвит device_map с учётом квантизации и доступности CUDA.

        Стратегии:
        - ``'auto_cuda'`` (дефолт): пинит модель на текущий CUDA-девайс.
          Безопасно для single-GPU + квантизация. Бросает RuntimeError если
          квантизация запрошена, но CUDA недоступна.
        - ``'auto'``: HF авто-шардирование — для multi-GPU без квантизации.
        - ``None``: без device_map — PyTorch сам решает (CPU/MPS).
        - Explicit dict: полный контроль, передаётся as-is.

        Raises:
            RuntimeError: Если квантизация запрошена, но CUDA недоступна.
        """
        if bnb_config is not None and not torch.cuda.is_available():
            raise RuntimeError("BitsAndBytes квантизация требует CUDA, но CUDA недоступна.")

        if self.device_map == "auto_cuda":
            if torch.cuda.is_available():
                return {"": torch.cuda.current_device()}
            # CPU / MPS — без device_map
            return None

        return self.device_map

    def _build_modifiers(self, tokenizer: Any) -> list:
        """Инстанциирует и возвращает список модификаторов в порядке defaults.

        Runtime-аргументы определяются автоматически через маркерные атрибуты
        класса — без строкового матча по ``_target_``:
        - ``_needs_tokenizer = True`` -> передаёт ``tokenizer``
        - ``_needs_lora_path = True`` -> передаёт ``lora_resume_path``

        Порядок ключей в ``model.modifiers`` соответствует порядку в defaults
        ``model/default.yaml``. Hydra гарантирует сохранение порядка DictConfig.

        Raises:
            ValueError: Если ``EmbeddingResizeModifier`` запрошен без tokenizer.
        """
        if not self.modifiers_cfg:
            return []

        modifiers = []
        for name, modifier_cfg in self.modifiers_cfg.items():
            if modifier_cfg is None:
                logger.debug("Модификатор '%s' отключен (None). Пропускаем.", name)
                continue
            target = modifier_cfg.get("_target_", "")

            # Резолвим класс через _locate — внутренний но стабильный API Hydra,
            # де-факто стандарт для pre-instantiate интроспекции в Hydra-проектах.
            # Позволяет читать маркеры класса до его создания.
            modifier_cls = _locate(target)

            extra_kwargs: dict[str, Any] = {}
            
            if getattr(modifier_cls, "_needs_tokenizer", False):
                if tokenizer is None:
                    raise ValueError(
                        f"Модификатор '{name}' ({target}) требует tokenizer, "
                        "но build() вызван без него."
                    )
                extra_kwargs["tokenizer"] = tokenizer

            if getattr(modifier_cls, "_needs_lora_path", False):
                extra_kwargs["lora_resume_path"] = self.lora_resume_path

            modifier = instantiate(modifier_cfg, **extra_kwargs)
            logger.info("Modifier инициализирован: %s (%s)", name, target)
            modifiers.append(modifier)

        return modifiers

    def _download_model_from_s3(self) -> str:
        """Скачивает веса модели из S3 в локальный кэш."""
        missing = [v for v in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY") if not os.getenv(v)]
        if missing:
            raise EnvironmentError(f"AWS credentials не установлены: {missing}")

        parsed = urlparse(self.model_name_or_path)
        bucket_name = parsed.netloc
        prefix = parsed.path.lstrip("/")

        # Формируем путь в cache_dir
        cache_base = Path(self.cache_dir) if self.cache_dir else Path.home() / ".cache" / "huggingface"
        local_model_dir = cache_base / "s3_models" / bucket_name / prefix

        if local_model_dir.exists() and any(local_model_dir.iterdir()):
            logger.info("S3 модель найдена в кэше: %s", local_model_dir)
            return str(local_model_dir)

        logger.info("Скачиваем модель из S3: %s", self.model_name_or_path)
        local_model_dir.mkdir(parents=True, exist_ok=True)

        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                rel_path = key[len(prefix):].lstrip("/")
                file_dest = local_model_dir / rel_path
                file_dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket_name, key, str(file_dest))

        logger.info("Скачивание модели из S3 завершено: %s", local_model_dir)
        return str(local_model_dir)

    def _validate_quantization(
        self,
        model: PreTrainedModel,
        requested_config: Any,
    ) -> None:
        """Проверяет соответствие запрошенной квантизации реальным весам модели."""
        from transformers import AwqConfig, GPTQConfig

        model_quant_cfg = getattr(model.config, "quantization_config", None)

        # Запросили AWQ — веса должны быть AWQ
        if isinstance(requested_config, AwqConfig):
            if model_quant_cfg is None or model_quant_cfg.get("quant_type") != "awq":
                raise ValueError(
                    f"quantization_config=AwqConfig, но модель '{self.model_name_or_path}' "
                    f"не квантизована в AWQ формат. "
                    f"Запусти scripts/tools/quantize_awq.py или возьми готовую AWQ модель с HF."
                )

        # Запросили BnB — у модели не должно быть своего AWQ/GPTQ конфига
        if isinstance(requested_config, BitsAndBytesConfig):
            if model_quant_cfg is not None:
                quant_type = model_quant_cfg.get("quant_type", "unknown")
                raise ValueError(
                    f"quantization_config=BitsAndBytesConfig, но модель уже квантизована "
                    f"({quant_type}). Используй соответствующий конфиг или none.yaml."
                )
    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    def build(self, tokenizer: Any = None) -> PreTrainedModel:
        """Загружает модель и последовательно применяет все модификаторы.

        Args:
            tokenizer: Инстанс токенизатора. Обязателен если среди модификаторов
                есть модификаторы с ``_needs_tokenizer = True``.

        Returns:
            Готовая ``PreTrainedModel`` (или ``PeftModel`` при наличии LoRA-адаптера).

        Raises:
            ValueError: При невалидном ``torch_dtype`` или отсутствии ``tokenizer``
                для модификатора с ``_needs_tokenizer = True``.
            RuntimeError: Если квантизация запрошена на машине без CUDA.
            ImportError: Если ``auto_model_class`` указывает на недоступный модуль.
        """
        if self.model_name_or_path.startswith("s3://"):
            self.model_name_or_path = self._download_model_from_s3()
            
        logger.info(
            "Загрузка модели: %s (class=%s)",
            self.model_name_or_path,
            self.auto_model_class,
        )

        module_name, class_name = self.auto_model_class.rsplit(".", 1)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        bnb_config = self._parse_quantization_config(self.quantization_config)
        parsed_dtype = self._parse_torch_dtype(self.torch_dtype)
        attn_impl = self._resolve_attn_implementation(self.attn_implementation)
        resolved_device_map = self._resolve_device_map(bnb_config)

        parsed_rope_scaling = None
        if self.rope_scaling is not None:
            parsed_rope_scaling = (
                OmegaConf.to_container(self.rope_scaling, resolve=True)
                if isinstance(self.rope_scaling, DictConfig)
                else dict(self.rope_scaling)
            )

        model_kwargs: dict[str, Any] = {
            "cache_dir": self.cache_dir,
            "quantization_config": bnb_config,
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": parsed_dtype,
            "device_map": resolved_device_map,
        }

        # Не передаём None в attn_implementation — некоторые модели падают на этом
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl

        if parsed_rope_scaling is not None:
            model_kwargs["rope_scaling"] = parsed_rope_scaling

        model = model_class.from_pretrained(self.model_name_or_path, **model_kwargs)
        self._validate_quantization(model, bnb_config) 
        logger.info("Базовая модель загружена.")

        for modifier in self._build_modifiers(tokenizer):
            model = modifier(model)

        return model