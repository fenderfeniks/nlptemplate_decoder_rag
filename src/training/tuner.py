import gc
import logging

import optuna
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from optuna.integration import PyTorchLightningPruningCallback
from optuna.integration.mlflow import MLflowCallback


logger = logging.getLogger(__name__)


class NLPOptunaTuner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.optuna_cfg = cfg.optuna

    def objective(self, trial: optuna.Trial) -> float:
        """Функция, которую Optuna будет запускать N раз"""

        # 1. Генерируем гиперпараметры (пространство поиска)
        # Подбираем Learning Rate и Weight Decay для AdamW
        lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)

        # 2. Динамически подменяем значения в конфиге Гидры
        self.cfg.model_module.optimizer_cfg.lr = lr
        self.cfg.model_module.optimizer_cfg.weight_decay = weight_decay

        logger.info(
            f"Trial {trial.number} started with lr={lr:.2e}, weight_decay={weight_decay:.2e}"
        )

        # 3. Инициализируем компоненты стандартно, как в eval.py и infer.py
        tokenizer = instantiate(self.cfg.model.tokenizer).build()
        datamodule = instantiate(self.cfg.datamodule, tokenizer=tokenizer)

        base_model = instantiate(self.cfg.model.builder, tokenizer=tokenizer).build()
        model_module = instantiate(self.cfg.model_module, model=base_model)

        # 4. Настраиваем Trainer с интеграцией ранней остановки (Pruning)
        callbacks = []
        if self.optuna_cfg.enable_pruning:
            # Если метрика на 2-й эпохе ужасная, Optuna убьет этот Trial досрочно
            callbacks.append(
                PyTorchLightningPruningCallback(trial, monitor=self.optuna_cfg.metric_name)
            )

        trainer = instantiate(self.cfg.trainer, callbacks=callbacks)

        # 5. Запускаем обучение
        try:
            trainer.fit(model=model_module, datamodule=datamodule)
            val_metric = trainer.callback_metrics.get(self.optuna_cfg.metric_name)

            if val_metric is None:
                # ИСПРАВЛЕНИЕ: Динамический штраф вместо хардкода 0.0
                failure_val = (
                    float("inf") if self.optuna_cfg.direction == "minimize" else float("-inf")
                )
                logger.warning(
                    f"Метрика {self.optuna_cfg.metric_name} не найдена. Возвращаем {failure_val}."
                )
                return failure_val

            return val_metric.item()

        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            # ИСПРАВЛЕНИЕ: Тот же динамический штраф при падении
            return float("inf") if self.optuna_cfg.direction == "minimize" else float("-inf")
        finally:
            del trainer, model_module, base_model, datamodule
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def optimize(self):
        """Запуск процесса подбора с логированием в MLflow"""
        logger.info(f"Создание Optuna Study: {self.optuna_cfg.study_name}")

        # 1. Настраиваем MLflow коллбек для Optuna
        mlflow_kwargs = {
            "tracking_uri": self.cfg.trainer.logger.tracking_uri,
            "metric_name": self.optuna_cfg.metric_name,
            "create_experiment": True,
        }

        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow_kwargs["tracking_uri"],
            metric_name=mlflow_kwargs["metric_name"],
            # Складываем все trials Optuna в отдельный эксперимент, чтобы не смешивать с основными!
            experiment_name=f"{self.cfg.trainer.logger.experiment_name}_HPO",
            nest_trials=True,  # Делает дашборд чистым (один родительский ран, внутри 50 child ранов)
        )

        # 2. Создаем Study
        study = optuna.create_study(
            study_name=self.optuna_cfg.study_name,
            direction=self.optuna_cfg.direction,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
            if self.optuna_cfg.enable_pruning
            else None,
        )

        # 3. Запускаем оптимизацию, передавая MLflowCallback
        study.optimize(
            self.objective, n_trials=self.optuna_cfg.n_trials, callbacks=[mlflow_callback]
        )

        logger.info(f"Лучший результат ({self.optuna_cfg.metric_name}): {study.best_value:.4f}")
