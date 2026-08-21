# src/pipelines/base/core/data/transforms/deduplication.py
import hashlib
import logging
import re
from typing import Any

from datasets import Dataset as HFDataset

from src.pipelines.base.core.data.transforms.base import BaseDatasetTransform

logger = logging.getLogger(__name__)

# Пытаемся импортировать datasketch, оставляем предупреждение если его нет
try:
    from datasketch import MinHash, MinHashLSH
except ImportError:
    MinHash, MinHashLSH = None, None
    logger.warning(
        "Библиотека datasketch не установлена. "
        "MinHashDeduplicationTransform будет недоступен."
    )

# Схема MinHash фиксируется явно (unsigned 32-bit, совместимость с datasketch < 1.6),
# чтобы быть независимой от дефолта библиотеки.
# Менять только вместе с пересчётом всех сигнатур.
_MINHASH_SCHEME = "legacy"

# Служебные колонки, добавляемые в процессе дедупликации и удаляемые после
_HASH_COLUMN = "_md5_hash"
_MINHASH_COLUMN = "_minhash_signature"


class ExactDeduplicationTransform(BaseDatasetTransform):
    """Отсеивает полные дубликаты текстов на основе точного MD5-хэширования.

    Для decoder-пайплайна поддерживает два режима через параметр ``target_columns``:

    - CPT: дедупликация по колонке ``text``
    - SFT: дедупликация по колонке ``prompt`` (таргет намеренно исключён —
      разные ответы на один промпт дубликатами не считаются)

    Оригиналом считается первый встреченный документ с данным хэшем.
    Служебная колонка ``_md5_hash`` удаляется из датасета после фильтрации.
    """

    def __init__(
        self,
        target_columns: list[str],
        num_proc: int = 4,
        column_separator: str = "\n\n",
    ) -> None:
        """
        Args:
            target_columns: Список колонок для хэширования. Значения склеиваются
                через ``column_separator`` перед вычислением хэша. Если ни одна
                из колонок не найдена в датасете — дедупликация пропускается
                с предупреждением.
            num_proc: Число процессов для параллельного map.
            column_separator: Разделитель при конкатенации нескольких колонок.

        Raises:
            ValueError: Если ``target_columns`` — пустой список.
        """
        if not target_columns:
            raise ValueError("target_columns не может быть пустым списком.")
        self.target_columns = target_columns
        self.num_proc = num_proc
        self.column_separator = column_separator

    def __call__(self, dataset: HFDataset) -> HFDataset:
        active_cols = [c for c in self.target_columns if c in dataset.column_names]
        if not active_cols:
            logger.warning(
                "Ни одна из колонок %s не найдена в датасете — "
                "точная дедупликация пропущена.",
                self.target_columns,
            )
            return dataset

        logger.info(
            "Запуск точной дедупликации (MD5) по колонкам %s...", active_cols
        )
        initial_count = len(dataset)

        def _compute_hash(example: dict[str, Any]) -> dict[str, str]:
            combined = self.column_separator.join(
                str(example[c]) for c in active_cols
            )
            return {_HASH_COLUMN: hashlib.md5(combined.encode("utf-8")).hexdigest()}

        hashed_dataset = dataset.map(
            _compute_hash,
            num_proc=self.num_proc,
            desc="MD5 hashing",
        )

        unique_hashes: set[str] = set()
        unique_indices: list[int] = []
        for idx, hash_val in enumerate(hashed_dataset[_HASH_COLUMN]):
            if hash_val not in unique_hashes:
                unique_hashes.add(hash_val)
                unique_indices.append(idx)

        dedup_dataset = dataset.select(unique_indices)

        logger.info(
            "Точная дедупликация: %d -> %d (удалено %d дубликатов)",
            initial_count,
            len(dedup_dataset),
            initial_count - len(dedup_dataset),
        )

        # Удаляем служебную колонку — она не должна попасть в downstream трансформы
        return dedup_dataset.remove_columns(
            [c for c in [_HASH_COLUMN] if c in dedup_dataset.column_names]
        )


class MinHashDeduplicationTransform(BaseDatasetTransform):
    """Отсеивает нечёткие дубликаты (Near-Duplicates) с помощью MinHash LSH.

    Устойчив к изменению пунктуации, лишним пробелам и замене отдельных слов.
    Схема хэширования фиксирована константой ``_MINHASH_SCHEME`` — не меняйте её
    без пересчёта всех сигнатур.

    Оригиналом считается первый встреченный документ: порядок обхода влияет
    на то, какой из похожих документов будет сохранён.
    Служебная колонка ``_minhash_signature`` удаляется из датасета после фильтрации.

    Note:
        Дефолтный порог 0.9 (против 0.85 в RAG) выбран намеренно: в SFT-данных
        допустимо большее лексическое сходство между промптами разных тем,
        и агрессивная фильтрация приводит к потере вариативности.
    """

    def __init__(
        self,
        target_columns: list[str],
        num_perm: int = 128,
        threshold: float = 0.9,
        ngram_size: int = 5,
        num_proc: int = 4,
        column_separator: str = "\n\n",
    ) -> None:
        """
        Args:
            target_columns: Список колонок для хэширования (аналогично
                ExactDeduplicationTransform).
            num_perm: Количество перестановок хэш-функции (точность vs скорость).
                128 — стандартный баланс для большинства задач.
            threshold: Порог сходства Jaccard [0, 1]. При 0.9 тексты,
                совпадающие на 90%+, считаются дубликатами. Выше чем в RAG —
                см. Note в docstring класса.
            ngram_size: Размер шинглов (n-грамм по словам).
            num_proc: Число процессов для параллельного вычисления сигнатур.
            column_separator: Разделитель при конкатенации нескольких колонок.

        Raises:
            ImportError: Если библиотека datasketch не установлена.
            ValueError: Если ``target_columns`` — пустой список.
        """
        if MinHash is None:
            raise ImportError(
                "Для MinHashDeduplicationTransform установите: pip install datasketch"
            )
        if not target_columns:
            raise ValueError("target_columns не может быть пустым списком.")

        self.target_columns = target_columns
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram_size = ngram_size
        self.num_proc = num_proc
        self.column_separator = column_separator
        self.word_pattern = re.compile(r"(?u)\b\w+\b")

    def __call__(self, dataset: HFDataset) -> HFDataset:
        active_cols = [c for c in self.target_columns if c in dataset.column_names]
        if not active_cols:
            logger.warning(
                "Ни одна из колонок %s не найдена в датасете — "
                "MinHash дедупликация пропущена.",
                self.target_columns,
            )
            return dataset

        logger.info(
            "Запуск MinHash LSH дедупликации по %s "
            "(threshold=%.2f, ngrams=%d, num_perm=%d)...",
            active_cols,
            self.threshold,
            self.ngram_size,
            self.num_perm,
        )
        initial_count = len(dataset)

        def _compute_minhash(example: dict[str, Any]) -> dict[str, list[int]]:
            combined = self.column_separator.join(
                str(example[c]) for c in active_cols
            ).lower()
            tokens = self.word_pattern.findall(combined)

            # Явно задаём scheme при создании — независимость от дефолта библиотеки
            m = MinHash(num_perm=self.num_perm, scheme=_MINHASH_SCHEME)
            if len(tokens) < self.ngram_size:
                # Текст короче n-граммы — хэшируем целиком
                m.update(" ".join(tokens).encode("utf-8"))
            else:
                for i in range(len(tokens) - self.ngram_size + 1):
                    shingle = " ".join(tokens[i : i + self.ngram_size]).encode("utf-8")
                    m.update(shingle)

            # list[int] — единственный тип, который HF Dataset умеет сериализовать
            return {_MINHASH_COLUMN: m.hashvalues.tolist()}

        # 1. Параллельно считаем сигнатуры
        hashed_dataset = dataset.map(
            _compute_minhash,
            num_proc=self.num_proc,
            desc="Computing MinHash signatures",
        )

        # 2. Строим LSH-индекс и выявляем дубликаты.
        #    Логика: первый встреченный документ считается оригиналом,
        #    все последующие похожие на него — дубликатами.
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        duplicates_to_remove: set[int] = set()

        for idx, signature_list in enumerate(hashed_dataset[_MINHASH_COLUMN]):
            if idx in duplicates_to_remove:
                continue

            # Восстанавливаем MinHash с той же схемой, что использовалась при вычислении
            m = MinHash(
                num_perm=self.num_perm,
                hashvalues=signature_list,
                scheme=_MINHASH_SCHEME,
            )
            similar = lsh.query(m)
            if similar:
                # Текущий документ похож на уже проиндексированный — дубликат
                duplicates_to_remove.add(idx)
            else:
                # Уникальный — добавляем в индекс как эталон
                lsh.insert(str(idx), m)

        # 3. Фильтруем датасет
        unique_indices = [i for i in range(initial_count) if i not in duplicates_to_remove]
        dedup_dataset = dataset.select(unique_indices)

        logger.info(
            "MinHash дедупликация: %d -> %d (удалено %d дубликатов)",
            initial_count,
            len(dedup_dataset),
            initial_count - len(dedup_dataset),
        )

        # Удаляем служебную колонку — она не должна попасть в downstream трансформы
        return dedup_dataset.remove_columns(
            [c for c in [_MINHASH_COLUMN] if c in dedup_dataset.column_names]
        )