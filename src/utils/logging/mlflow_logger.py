# src/utils/logging/mlflow_logger.py
"""MLflow-реализация ExperimentLogger.

Вся MLflow-специфика проекта сосредоточена здесь.

Два класса:
    MLflowLogger          — standalone-режим (eval.py, DAG, inference, post-training).
    LightningMLflowLogger — режим обучения (callbacks внутри Lightning цикла).

При смене бэкенда — создать новый файл рядом, реализовать ExperimentLogger,
поменять _target_ в конфиге Hydra. Весь остальной код не трогается.
"""

from __future__ import annotations

import gc
import logging
import os
import re
import sys
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.utils.torch_utils import register_safe_globals


logger = logging.getLogger(__name__)


# ===========================================================================
# Приватные утилиты
# ===========================================================================


def _extract_run_id_from_trainer(trainer: Any) -> str | None:
    if not trainer.logger:
        return None
    for attr in ("run_id", "_run_id", "runid"):
        val = getattr(trainer.logger, attr, None)
        if val:
            return val
    try:
        active = mlflow.active_run()
        if active:
            return active.info.run_id
    except Exception:
        pass
    return None


def _ensure_tracking_uri() -> None:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    else:
        logger.warning("MLFLOW_TRACKING_URI не задан. Используется ./mlruns")


def _find_adapter_config(root: Path) -> Path | None:
    for candidate in [root, root / "peft", root / "lora_weights", root / "adapter"]:
        if (candidate / "adapter_config.json").exists():
            return candidate
    matches = list(root.rglob("adapter_config.json"))
    return matches[0].parent if matches else None


def _download_by_run_id(run_id: str, artifact_path: str) -> Path:
    logger.info("Скачивание адаптера: run_id=%s path=%s", run_id, artifact_path)
    return Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path))


def _download_by_registry(model_name: str, alias: str, artifact_path: str) -> Path:
    logger.info("Поиск адаптера в Registry: %s (alias='%s')", model_name, alias)
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
    except MlflowException as e:
        raise MlflowException(
            f"Алиас '{alias}' не найден для модели '{model_name}': {e.message}"
        ) from e
    logger.info("Найдена версия %s (run_id=%s)", mv.version, mv.run_id)
    if mv.run_id:
        return Path(
            mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path=artifact_path)
        )
    if mv.source:
        return Path(mlflow.artifacts.download_artifacts(artifact_uri=mv.source))
    raise MlflowException(
        f"Версия {mv.version} модели '{model_name}' не содержит ни run_id, ни source URI."
    )


def _patch_peft_config_for_hydra(model: Any) -> None:
    if not hasattr(model, "peft_config"):
        return
    for _, peft_cfg in model.peft_config.items():
        for key, value in vars(peft_cfg).items():
            if isinstance(value, (ListConfig, DictConfig)):
                setattr(peft_cfg, key, OmegaConf.to_container(value, resolve=True))


def _build_reg_model_name(mlflow_model_name: str, adapter_type: str = "LoRA") -> str:
    return f"{mlflow_model_name}_{adapter_type}"


def _save_adapter_to_tempdir(model_to_save: Any, tokenizer: Any, tmp_path: Path) -> None:
    logger.info("Сохранение адаптера во временную директорию: %s", tmp_path)
    model_to_save.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    if not (tmp_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"save_pretrained не создал adapter_config.json в {tmp_path}. "
            f"Содержимое: {list(tmp_path.iterdir())}"
        )
    logger.info("Файлы адаптера: %s", [f.name for f in tmp_path.iterdir()])


def _create_model_version(
    client: MlflowClient, run_id: str, artifact_path: str, reg_model_name: str
) -> str:
    model_uri = f"runs:/{run_id}/{artifact_path}"
    try:
        client.create_registered_model(reg_model_name)
    except Exception:
        pass
    try:
        mv = client.create_model_version(name=reg_model_name, source=model_uri, run_id=run_id)
        return mv.version
    except Exception as e:
        logger.warning("create_model_version упал (%s), пробуем register_model...", e)
        return mlflow.register_model(model_uri=model_uri, name=reg_model_name).version


if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

_INFERENCE_GROUP = "inference-core"


def _get_inference_pip_requirements(pyproject_path: str | Path) -> list[str]:
    pyproject_path = Path(pyproject_path)
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    try:
        declared = data["project"]["optional-dependencies"][_INFERENCE_GROUP]
    except KeyError:
        logger.warning("Группа %s не найдена в %s.", _INFERENCE_GROUP, pyproject_path)
        return []
    pinned: list[str] = []
    for req in declared:
        pkg_name = re.split(r"[<>=!~\[;]", req, maxsplit=1)[0].strip()
        try:
            pinned.append(f"{pkg_name}=={pkg_version(pkg_name)}")
        except PackageNotFoundError:
            logger.warning("Пакет '%s' не установлен — пропускаю.", pkg_name)
    return pinned


# ===========================================================================
# MLflowLogger — standalone (eval.py, DAG, inference, post-training)
# ===========================================================================


class MLflowLogger:
    """Standalone MLflow логгер. Реализует ExperimentLogger."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "Default",
        artifact_location: str | None = None,
    ) -> None:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            _ensure_tracking_uri()

        if artifact_location and not artifact_location.startswith("file://"):
            abs_path = Path(artifact_location).resolve().as_posix()
            artifact_location = f"file:///{abs_path}"

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=artifact_location,
            )
        else:
            experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_id=experiment_id)

    # --- метрики и таблицы -------------------------------------------------

    def log_metrics(self, metrics: dict[str, float], stage: str, step: int = 0) -> None:
        if not mlflow.active_run():
            logger.warning("log_metrics: нет активного run.")
            return
        for name, value in metrics.items():
            mlflow.log_metric(f"{stage}_{name}", value, step=step)

    def log_table(
        self,
        df: pd.DataFrame,
        stage: str,
        step: int = 0,
        artifact_suffix: str = "",
    ) -> None:
        if not mlflow.active_run():
            logger.warning("log_table: нет активного run.")
            return
        artifact_file = f"generations/{stage}_step_{step}_results{artifact_suffix}.json"
        mlflow.log_table(data=df, artifact_file=artifact_file)

    # --- run management ----------------------------------------------------

    def get_run_id(self, trainer: Any = None) -> str | None:
        if trainer is not None:
            return _extract_run_id_from_trainer(trainer)
        active = mlflow.active_run()
        return active.info.run_id if active else None

    @contextmanager
    def start_run(self, run_name: str = "") -> Generator[None, None, None]:
        with mlflow.start_run(run_name=run_name):
            yield

    @contextmanager
    def reopen_run(self, run_id: str) -> Generator[None, None, None]:
        """Переоткрывает существующий MLflow run для дологирования.

        Автоматически устанавливает нужный experiment_id чтобы избежать
        конфликта 'active experiment ID does not match environment run ID'.
        """
        try:
            client = MlflowClient()
            run_info = client.get_run(run_id)
            mlflow.set_experiment(experiment_id=run_info.info.experiment_id)
            logger.info(
                "reopen_run: experiment_id=%s run_id=%s",
                run_info.info.experiment_id,
                run_id,
            )
        except Exception as e:
            logger.warning("reopen_run: не удалось установить эксперимент: %s", e)

        with mlflow.start_run(run_id=run_id):
            yield

    # --- адаптеры ----------------------------------------------------------

    def save_adapter(
        self,
        cfg: Any,
        model_module: Any,
        tokenizer: Any,
        run_id: str,
        pipeline_name: str,
        best_score: float | None = None,
    ) -> None:
        logger.info("Сохранение LoRA-адаптера (run_id=%s)...", run_id)
        gc.collect()
        register_safe_globals()

        model_to_save = model_module.model
        client = MlflowClient()

        mlflow_model_name = OmegaConf.select(cfg, "model.architecture.mlflow_model_name")
        if mlflow_model_name is None:
            raise ValueError(
                "Не найден model.architecture.mlflow_model_name в конфиге. "
                f"Доступные ключи: {list(OmegaConf.to_container(cfg).keys())}"
            )
        reg_model_name = _build_reg_model_name(mlflow_model_name)

        registry_cfg = cfg.get("logger", {}).get("registry", {})
        artifact_path = registry_cfg.get("artifact_path", "lora_weights")
        register_on_success = registry_cfg.get("register_on_success", True)
        promote_to_staging = registry_cfg.get("promote_to_staging", True)

        _patch_peft_config_for_hydra(model_to_save)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _save_adapter_to_tempdir(model_to_save, tokenizer, tmp_path)
            with self.reopen_run(run_id):
                mlflow.log_artifacts(str(tmp_path), artifact_path=artifact_path)
                if best_score is not None:
                    mlflow.log_metric("promotion_candidate_val_loss", best_score)
            logger.info("Адаптер сохранён: run_id=%s path=%s", run_id, artifact_path)

        if not register_on_success:
            logger.info("Регистрация в Registry отключена.")
            return

        mv_version = _create_model_version(client, run_id, artifact_path, reg_model_name)
        logger.info("Зарегистрирована '%s' версия %s.", reg_model_name, mv_version)

        if promote_to_staging:
            client.set_registered_model_alias(
                name=reg_model_name, alias="Staging", version=mv_version
            )
            logger.info("'%s' v%s -> алиас 'Staging'.", reg_model_name, mv_version)

        if best_score is not None:
            client.set_model_version_tag(reg_model_name, mv_version, "val_loss", str(best_score))

    def load_adapter(
        self,
        resume_cfg: Any,
        tracking_uri: str | None = None,
    ) -> str | None:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        if isinstance(resume_cfg, DictConfig):
            resume_cfg = OmegaConf.to_container(resume_cfg, resolve=True)

        if not resume_cfg or not resume_cfg.get("enabled", False):
            return None

        run_id = resume_cfg.get("run_id")
        model_name = resume_cfg.get("model_name")
        alias = resume_cfg.get("alias")
        artifact_path = resume_cfg.get("artifact_path", "lora_weights")

        if run_id:
            downloaded = _download_by_run_id(run_id, artifact_path)
        elif model_name and alias:
            downloaded = _download_by_registry(model_name, alias, artifact_path)
        else:
            raise ValueError("Укажите 'run_id' или 'model_name' + 'alias'.")

        adapter_dir = _find_adapter_config(downloaded)
        if adapter_dir is None:
            raise FileNotFoundError(
                f"adapter_config.json не найден в: {downloaded}\n"
                f"Содержимое: {list(downloaded.rglob('*'))}"
            )
        logger.info("Адаптер загружен: %s", adapter_dir)
        return str(adapter_dir)

    # --- registry ----------------------------------------------------------

    def promote_model(
        self,
        reg_model_name: str,
        staging_alias: str = "Staging",
        production_alias: str = "Production",
        metric_tag: str = "val_loss",
    ) -> bool:
        client = MlflowClient()
        try:
            staging_mv = client.get_model_version_by_alias(reg_model_name, staging_alias)
        except MlflowException as e:
            raise MlflowException(
                f"Алиас '{staging_alias}' не найден для '{reg_model_name}'."
            ) from e

        staging_version = staging_mv.version
        staging_score_str = staging_mv.tags.get(metric_tag)
        if staging_score_str is None:
            raise ValueError(f"У {staging_alias} модели нет тега '{metric_tag}'.")
        staging_score = float(staging_score_str)

        try:
            current_prod = client.get_model_version_by_alias(reg_model_name, production_alias)
            if current_prod.version == staging_version:
                logger.warning("Версия %s уже является Production.", staging_version)
                return False
            prod_score = float(current_prod.tags.get(metric_tag) or "inf")
        except MlflowException:
            prod_score = float("inf")

        if staging_score < prod_score:
            client.set_registered_model_alias(reg_model_name, production_alias, staging_version)
            logger.info("УСПЕХ: v%s -> Production.", staging_version)
            return True

        logger.warning(
            "ОТКАЗ: Staging (%.4f) не лучше Production (%.4f).", staging_score, prod_score
        )
        return False

    def get_production_version(
        self,
        reg_model_name: str,
        production_alias: str = "Production",
    ) -> str:
        client = MlflowClient()
        try:
            mv = client.get_model_version_by_alias(reg_model_name, production_alias)
            return mv.version
        except MlflowException as e:
            raise MlflowException(
                f"Алиас '{production_alias}' не найден для '{reg_model_name}'."
            ) from e


# ===========================================================================
# LightningMLflowLogger — режим обучения (callbacks внутри Lightning цикла)
# ===========================================================================


class LightningMLflowLogger:
    """MLflow логгер для Lightning-режима.

    Метрики идут через pl_module.log (DDP-синхронизация).
    Создаётся внутри callback'ов, не через Hydra.
    Используется ТОЛЬКО внутри активного Lightning цикла (fit/validate).
    """

    def __init__(self, trainer: Any, pl_module: Any) -> None:
        self._trainer = trainer
        self._pl_module = pl_module
        # Делегируем non-Lightning методы standalone логгеру
        self._standalone = MLflowLogger()

    def log_metrics(self, metrics: dict[str, float], stage: str, step: int = 0) -> None:
        for name, value in metrics.items():
            self._pl_module.log(
                f"{stage}_{name}", value, sync_dist=True, prog_bar=True, logger=True
            )

    def log_table(
        self,
        df: pd.DataFrame,
        stage: str,
        step: int = 0,
        artifact_suffix: str = "",
    ) -> None:
        run_id = _extract_run_id_from_trainer(self._trainer)
        if not run_id:
            logger.warning("LightningMLflowLogger.log_table: run_id не найден.")
            return
        artifact_file = f"generations/{stage}_step_{step}_results{artifact_suffix}.json"
        if hasattr(self._trainer.logger, "experiment"):
            self._trainer.logger.experiment.log_table(
                run_id=run_id, data=df, artifact_file=artifact_file
            )
        else:
            mlflow.log_table(data=df, artifact_file=artifact_file)

    def save_adapter(self, cfg, model_module, tokenizer, run_id, pipeline_name, best_score=None):
        self._standalone.save_adapter(
            cfg, model_module, tokenizer, run_id, pipeline_name, best_score
        )

    def load_adapter(self, resume_cfg, tracking_uri=None):
        return self._standalone.load_adapter(resume_cfg, tracking_uri)

    def promote_model(
        self,
        reg_model_name,
        staging_alias="Staging",
        production_alias="Production",
        metric_tag="val_loss",
    ):
        return self._standalone.promote_model(
            reg_model_name, staging_alias, production_alias, metric_tag
        )

    def get_production_version(self, reg_model_name, production_alias="Production"):
        return self._standalone.get_production_version(reg_model_name, production_alias)

    def get_run_id(self, trainer=None) -> str | None:
        return _extract_run_id_from_trainer(trainer or self._trainer)

    @contextmanager
    def start_run(self, run_name: str = "") -> Generator[None, None, None]:
        yield  # в Lightning-режиме run уже открыт трейнером

    @contextmanager
    def reopen_run(self, run_id: str) -> Generator[None, None, None]:
        yield  # в Lightning-режиме run уже открыт — переоткрывать не нужно
