# src/core/models/tokenization.py
import logging

from transformers import AutoTokenizer, PreTrainedTokenizerBase


logger = logging.getLogger(__name__)

_VALID_PADDING_SIDES = frozenset({"left", "right"})


class HFTokenizerBuilder:
    """Фабрика для безопасной загрузки и настройки HuggingFace токенизаторов.

    Решает типичные production-проблемы:
    - Отсутствующий ``pad_token`` у Llama / Mistral / Falcon → фикс через ``eos_token``.
    - Явное задание ``padding_side`` (важно для batched generation vs training).
    - Поддержка кастомных chat-template для instruction fine-tuning.
    """

    def __init__(
        self,
        tokenizer_name: str,
        use_fast: bool = True,
        padding_side: str = "right",
        add_eos_token: bool = False,
        chat_template: str | None = None,
        trust_remote_code: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        """
        Args:
            tokenizer_name: HF Hub id или локальный путь к токенизатору,
                например ``'meta-llama/Meta-Llama-3-8B'``.
            use_fast: Использовать Rust-реализацию (``tokenizers`` backend).
                Быстрее в 5–10 раз, стандарт де-факто. ``False`` только если
                модель явно не поддерживает fast tokenizer.
            padding_side: ``'right'`` для обучения (causal LM требует правый padding
                чтобы не смещать позиции токенов); ``'left'`` для батч-инференса
                (генерация начинается с одной позиции для всего батча).
            add_eos_token: Автоматически добавлять EOS в конец каждой последовательности.
            chat_template: Jinja2-шаблон для форматирования диалогов.
                Если ``None`` — используется шаблон из конфига токенизатора.
            trust_remote_code: Разрешить выполнение кода из репозитория токенизатора.
                Нужно для ряда моделей (например, Qwen, InternLM).
            cache_dir: Директория кэша HF. По умолчанию ``~/.cache/huggingface``.

        Raises:
            ValueError: Если ``padding_side`` не ``'left'`` или ``'right'``.
        """
        if padding_side not in _VALID_PADDING_SIDES:
            raise ValueError(
                f"Недопустимое значение padding_side: '{padding_side}'. "
                f"Допустимые: {sorted(_VALID_PADDING_SIDES)}."
            )

        self.tokenizer_name = tokenizer_name
        self.use_fast = use_fast
        self.padding_side = padding_side
        self.add_eos_token = add_eos_token
        self.chat_template = chat_template
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir

    def build(self) -> PreTrainedTokenizerBase:
        """Загружает и конфигурирует токенизатор.

        Returns:
            Инициализированный и пропатченный токенизатор.
        """
        logger.info("Загрузка токенизатора: %s", self.tokenizer_name)

        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            use_fast=self.use_fast,
            add_eos_token=self.add_eos_token,
            trust_remote_code=self.trust_remote_code,
            cache_dir=self.cache_dir,
        )

        # Принудительно задаём padding_side после загрузки — некоторые токенизаторы
        # игнорируют его при from_pretrained если он прописан в конфиге модели
        tokenizer.padding_side = self.padding_side

        # Фикс для Llama / Mistral / Falcon: у них нет pad_token по умолчанию.
        # eos_token как pad_token — стандартная практика для causal LM fine-tuning;
        # лейблы для padding позиций маскируются через -100 в коллаторе.
        if tokenizer.pad_token is None:
            logger.warning(
                "Токенизатор '%s' не имеет pad_token → устанавливаем pad_token = eos_token ('%s').",
                self.tokenizer_name,
                tokenizer.eos_token,
            )
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if self.chat_template is not None:
            tokenizer.chat_template = self.chat_template
            logger.info("Установлен кастомный chat_template.")

        logger.info(
            "Токенизатор загружен: vocab_size=%d, pad='%s', padding_side='%s'",
            len(tokenizer),
            tokenizer.pad_token,
            tokenizer.padding_side,
        )
        return tokenizer
