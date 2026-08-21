# src/vector_store/lsh.py
"""MinHash LSH для нечёткой дедупликации документов.

Полностью независим от бэкенда векторного хранилища — работает одинаково
с FAISS, Qdrant и любым другим. ``KnowledgeBaseIndexer`` использует его
напрямую, не через векторное хранилище.

Требует: ``pip install datasketch``
"""

from __future__ import annotations

import logging
import pickle
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from datasketch import MinHash as MinHashType

try:
    from datasketch import MinHash, MinHashLSH

    _DATASKETCH_AVAILABLE = True
except ImportError:
    MinHash = None  # type: ignore[assignment, misc]
    MinHashLSH = None  # type: ignore[assignment, misc]
    _DATASKETCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class LSHIndex:
    """MinHash LSH индекс для нечёткой дедупликации near-duplicate документов.

    Инкапсулирует MinHashLSH и логику вычисления MinHash — оба ранее
    были размазаны между ``FAISSVectorDB`` и ``KnowledgeBaseIndexer``.

    Если ``datasketch`` не установлен — все методы работают как no-op,
    ``is_available`` возвращает False. Это позволяет использовать
    ``KnowledgeBaseIndexer`` без нечёткой дедупликации без изменения кода.

    Пример::

        lsh = LSHIndex(threshold=0.85, num_perm=128, ngram_size=5)
        if not lsh.is_duplicate("doc_42", text):
            lsh.register("doc_42", text)
    """

    def __init__(
        self,
        threshold: float = 0.85,
        num_perm: int = 128,
        ngram_size: int = 5,
    ) -> None:
        """
        Args:
            threshold: Порог сходства Jaccard [0, 1]. Документы с сходством
                выше порога считаются near-duplicate.
            num_perm: Число перестановок хэш-функции. Больше -> точнее, медленнее.
                128 — стандартный компромисс.
            ngram_size: Размер словесной n-граммы (шингла). Меньше -> чувствительнее
                к небольшим отличиям.
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self._word_pattern = re.compile(r"(?u)\b\w+\b")

        if not _DATASKETCH_AVAILABLE:
            logger.warning(
                "datasketch не установлен — нечёткая дедупликация (MinHashLSH) отключена. "
                "Установите: pip install datasketch"
            )
            self._lsh = None
        else:
            self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
            logger.info(
                "LSHIndex инициализирован (threshold=%.2f, num_perm=%d, ngram_size=%d)",
                threshold,
                num_perm,
                ngram_size,
            )

    @property
    def is_available(self) -> bool:
        """True если datasketch установлен и LSH работает."""
        return self._lsh is not None

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    def is_duplicate(self, text: str) -> bool:
        """Проверяет является ли текст нечётким дубликатом уже зарегистрированного.

        Чистый предикат — не изменяет состояние индекса.
        После принятия решения об уникальности вызови ``register``.

        Args:
            text: Текст для проверки.

        Returns:
            ``True`` если найден near-duplicate, ``False`` иначе или если
            datasketch не установлен.
        """
        if not self.is_available:
            return False
        m = self._compute_minhash(text)
        if m is None:
            return False
        return bool(self._lsh.query(m))

    def register(self, doc_id: str, text: str) -> None:
        """Регистрирует документ в LSH для детекции будущих нечётких дублей.

        Вызывается только после того как документ признан уникальным.
        Отделено от ``is_duplicate`` чтобы не нарушать принцип наименьшего удивления.

        Args:
            doc_id: Уникальный идентификатор документа (ключ в LSH).
            text: Текст документа.
        """
        if not self.is_available:
            return
        m = self._compute_minhash(text)
        if m is not None:
            self._lsh.insert(doc_id, m)

    def reset(self) -> None:
        """Очищает LSH индекс."""
        if not self.is_available:
            return
        self._lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        logger.info("LSHIndex сброшен.")

    # ------------------------------------------------------------------
    # Персистентность
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Сохраняет состояние LSH в файл (pickle).

        Warning:
            Используйте только для доверенных данных.
        """
        if not self.is_available:
            return
        with open(path, "wb") as f:
            pickle.dump(self._lsh, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("LSHIndex сохранён в '%s'.", path)

    def load(self, path: str | Path) -> None:
        """Загружает состояние LSH из файла (pickle).

        Warning:
            Pickle может выполнить произвольный код. Загружайте только
            из доверенных источников.
        """
        if not self.is_available:
            return
        path = Path(path)
        if not path.exists():
            logger.warning("Файл LSH не найден: '%s' — LSH пуст.", path)
            return
        warnings.warn(
            "LSHIndex.load() использует pickle. "
            "Убедитесь что файл получен из доверенного источника.",
            UserWarning,
            stacklevel=2,
        )
        with open(path, "rb") as f:
            self._lsh = pickle.load(f)  # noqa: S301
        logger.info("LSHIndex загружен из '%s'.", path)

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _compute_minhash(self, text: str) -> MinHashType | None:
        """Вычисляет MinHash сигнатуру текста через словесные n-граммы."""
        if not _DATASKETCH_AVAILABLE:
            return None

        tokens = self._word_pattern.findall(text.lower())
        m = MinHash(num_perm=self.num_perm, scheme="legacy")

        if len(tokens) < self.ngram_size:
            # Короткий текст — хэшируем целиком
            m.update(" ".join(tokens).encode("utf-8"))
        else:
            for i in range(len(tokens) - self.ngram_size + 1):
                shingle = " ".join(tokens[i : i + self.ngram_size]).encode("utf-8")
                m.update(shingle)
        return m
