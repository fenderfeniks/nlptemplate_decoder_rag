# src/pipelines/rag/core/data/schemas.py
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class RAGIndexingRecord(BaseModel):
    """Контракт для сырой записи при подготовке векторной базы (индексация).

    Используется при построении FAISS/Qdrant индекса.
    Валидирует наличие непустого текста достаточной длины.
    """

    text: str = Field(description="Сырой текст документа/статьи")
    metadata: Optional[dict] = Field(
        default_factory=dict,
        description="Метаданные документа (URL, title, date и др.)",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Проверяет текст документа на пустоту и минимальную длину.

        Args:
            v: Входная строка текста документа.

        Returns:
            Очищенная от пробелов по краям строка.

        Raises:
            ValueError: Если текст пустой или короче 10 символов.
        """
        v = v.strip()
        if not v:
            raise ValueError("Текст документа не может быть пустым")
        if len(v) < 10:
            raise ValueError(
                "Текст слишком короткий для индексации (минимум 10 символов)"
            )
        return v


class RAGTrainingRecord(BaseModel):
    """Контракт для обучающей записи энкодера (Contrastive Learning / Triplet Loss).

    Поддерживает два режима:
    - Contrastive: query + positive_doc (без negative_doc)
    - Triplet: query + positive_doc + negative_doc (Hard Negative Mining)

    Режим определяется автоматически по наличию negative_doc.
    """

    query: str = Field(description="Поисковый запрос")
    positive_doc: str = Field(description="Релевантный документ для запроса")
    negative_doc: Optional[str] = Field(
        default=None,
        description="Нерелевантный документ (Hard Negative). "
                    "Если передан — активируется Triplet Loss режим",
    )

    @field_validator("query", "positive_doc")
    @classmethod
    def validate_required_content(cls, v: str) -> str:
        """Проверяет обязательные текстовые поля на пустоту и минимальную длину.

        Args:
            v: Входная строка (query или positive_doc).

        Returns:
            Очищенная от пробелов по краям строка.

        Raises:
            ValueError: Если поле пустое или короче 3 символов.
        """
        v = v.strip()
        if not v:
            raise ValueError("Поле не может быть пустым")
        if len(v) < 3:
            raise ValueError("Текст слишком короткий (минимум 3 символа)")
        return v

    @field_validator("negative_doc")
    @classmethod
    def validate_negative_doc(cls, v: Optional[str]) -> Optional[str]:
        """Проверяет negative_doc на пустоту, если он передан.

        Args:
            v: Входная строка нерелевантного документа или None.

        Returns:
            Очищенная от пробелов по краям строка или None.

        Raises:
            ValueError: Если negative_doc передан, но является пустой строкой
                        или короче 3 символов.
        """
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError(
                    "negative_doc передан, но является пустой строкой"
                )
            if len(v) < 3:
                raise ValueError(
                    "negative_doc слишком короткий (минимум 3 символа)"
                )
        return v

    @model_validator(mode="after")
    def validate_training_mode(self) -> "RAGTrainingRecord":
        """Определяет и фиксирует режим обучения по составу полей.

        В Triplet режиме negative_doc не должен совпадать с positive_doc,
        иначе обучение теряет смысл.

        Returns:
            Валидная запись с корректным составом полей.

        Raises:
            ValueError: Если negative_doc идентичен positive_doc.
        """
        if self.negative_doc is not None:
            if self.negative_doc == self.positive_doc:
                raise ValueError(
                    "negative_doc не должен совпадать с positive_doc: "
                    "Hard Negative должен быть семантически отличным документом"
                )
        return self

    @property
    def mode(self) -> str:
        """Возвращает режим обучения текущей записи.

        Returns:
            'triplet' если передан negative_doc, иначе 'contrastive'.
        """
        return "triplet" if self.negative_doc is not None else "contrastive"