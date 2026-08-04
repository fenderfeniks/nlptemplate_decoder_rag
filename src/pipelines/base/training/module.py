# src/pipelines/base/training/module.py
import logging
from typing import Any

import torch
from hydra.utils import instantiate


logger = logging.getLogger(__name__)


class OptimizerMixin:
    """Миксин: configure_optimizers и on_save_checkpoint для всех NLP-модулей.

    Не наследует ``pl.LightningModule`` намеренно — это чистый миксин.
    Дочерние классы сами наследуют Lightning в правильном порядке:

        class MyModule(OptimizerMixin, pl.LightningModule): ...

    MRO при таком порядке: MyModule → OptimizerMixin → pl.LightningModule.
    ``super().__init__()`` проходит по одному пути без дублирования хуков Lightning.

    Требует от дочернего класса атрибутов:
    - ``self.model``           — nn.Module с параметрами
    - ``self.optimizer_cfg``   — DictConfig или callable
    - ``self.scheduler_cfg``   — DictConfig, callable или None
    - ``self.trainer``         — автоматически доступен через pl.LightningModule
    """

    # ------------------------------------------------------------------
    # Чекпоинтинг
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Оставляет в чекпоинте только веса PEFT-адаптера (если используется).

        Для Full FT сохраняет все веса без изменений.
        """
        try:
            from peft import PeftModel
            from peft.utils import get_peft_model_state_dict
        except ImportError:
            return

        if isinstance(self.model, PeftModel):
            checkpoint["state_dict"] = get_peft_model_state_dict(
                self.model, state_dict=checkpoint["state_dict"]
            )

    # ------------------------------------------------------------------
    # Оптимизация
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any] | torch.optim.Optimizer:
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if not trainable_params:
            logger.warning("Нет параметров для обучения! Проверь конфигурацию модификаторов.")

        if callable(self.optimizer_cfg):
            optimizer = self.optimizer_cfg(trainable_params)
        else:
            optimizer = instantiate(self.optimizer_cfg, params=trainable_params)

        if self.scheduler_cfg is None:
            return optimizer

        if callable(self.scheduler_cfg):
            # self.trainer — стандартный атрибут pl.LightningModule,
            # доступен после attach к Trainer (т.е. всегда к моменту configure_optimizers)
            total_steps = self.trainer.estimated_stepping_batches
            if total_steps == float("inf"):
                raise ValueError(
                    "estimated_stepping_batches=inf: задайте max_steps в конфиге тренера."
                )
            scheduler = self.scheduler_cfg(optimizer=optimizer, num_training_steps=total_steps)
        else:
            scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
