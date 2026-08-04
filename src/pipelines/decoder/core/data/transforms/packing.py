# src/pipelines/decoder/core/data/transforms/packing.py
import functools
import logging
import operator
from typing import Any

from datasets import Dataset as HFDataset

from src.pipelines.base.core.data.transforms.base import BaseDatasetTransform

logger = logging.getLogger(__name__)

# Колонки которые упаковываются — все три обязательны для training loop
_PACK_COLUMNS = ["input_ids", "attention_mask", "labels"]


class SequencePackingTransform(BaseDatasetTransform):
    """Упаковывает короткие токенизированные последовательности в длинные блоки.

    Конкатенирует все последовательности в батче в одну длинную цепочку,
    затем нарезает её на блоки фиксированного размера ``packing_chunk_size``.
    Применяется после токенизации; ожидает колонки ``input_ids``,
    ``attention_mask`` и ``labels``.

    .. note:: При ``drop_remainder=True`` хвост, не кратный ``packing_chunk_size``,
        отбрасывается. На больших датасетах потери незначительны, на малых —
        могут быть существенными. Используйте ``drop_remainder=False`` чтобы
        сохранить хвост с паддингом до ``packing_chunk_size``.
    """

    def __init__(
        self,
        packing_chunk_size: int = 2048,
        drop_remainder: bool = True,
        num_proc: int = 4,
        batch_size: int = 1000,
        writer_batch_size: int = 200,
    ) -> None:
        """
        Args:
            packing_chunk_size: Размер упакованного блока в токенах.
                Должен быть положительным числом.
            drop_remainder: Если ``True`` — хвост короче ``packing_chunk_size``
                отбрасывается. Если ``False`` — сохраняется как есть (без паддинга).
            num_proc: Число процессов для параллельного map.
            batch_size: Размер батча для map. Большие батчи дают лучшую упаковку
                за счёт большего пула последовательностей для конкатенации.
            writer_batch_size: Размер батча при записи на диск. Уменьшите при
                нехватке RAM.

        Raises:
            ValueError: Если ``packing_chunk_size`` не является положительным числом.
        """
        if packing_chunk_size <= 0:
            raise ValueError(
                f"packing_chunk_size должен быть положительным числом, "
                f"получено: {packing_chunk_size}"
            )
        self.packing_chunk_size = packing_chunk_size
        self.drop_remainder = drop_remainder
        self.num_proc = num_proc
        self.batch_size = batch_size
        self.writer_batch_size = writer_batch_size

    def __call__(self, dataset: HFDataset) -> HFDataset:
        active_cols = [c for c in _PACK_COLUMNS if c in dataset.column_names]
        if not active_cols:
            logger.warning(
                "Ни одна из колонок %s не найдена в датасете — "
                "упаковка пропущена. Убедитесь, что токенизация выполнена до этого шага.",
                _PACK_COLUMNS,
            )
            return dataset

        missing = [c for c in _PACK_COLUMNS if c not in dataset.column_names]
        if missing:
            logger.warning(
                "Колонки %s отсутствуют в датасете и не будут упакованы. "
                "Это может сломать training loop, ожидающий все три колонки.",
                missing,
            )

        logger.info(
            "Упаковка последовательностей по колонкам %s "
            "(chunk_size=%d, drop_remainder=%s)...",
            active_cols,
            self.packing_chunk_size,
            self.drop_remainder,
        )
        initial_count = len(dataset)

        def _pack_sequences(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
            concatenated = {
                k: functools.reduce(operator.iconcat, examples[k], [])
                for k in active_cols
            }
            total_length = len(concatenated[active_cols[0]])
            if self.drop_remainder:
                total_length = (
                    total_length // self.packing_chunk_size
                ) * self.packing_chunk_size
            return {
                k: [
                    t[i : i + self.packing_chunk_size]
                    for i in range(0, total_length, self.packing_chunk_size)
                ]
                for k, t in concatenated.items()
            }

        packed_dataset = dataset.map(
            _pack_sequences,
            batched=True,
            batch_size=self.batch_size,
            writer_batch_size=self.writer_batch_size,
            num_proc=self.num_proc,
            desc=f"Packing to {self.packing_chunk_size} tokens",
        )

        logger.info(
            "Упаковка завершена: %d записей → %d блоков",
            initial_count,
            len(packed_dataset),
        )
        return packed_dataset