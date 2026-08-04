import gc
import logging
import os
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.utils.torch_utils import register_safe_globals


if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logger = logging.getLogger(__name__)

_INFERENCE_GROUP: str = "inference-core"


# ==========================================
# БЛОК 1: УПРАВЛЕНИЕ ЗАВИСИМОСТЯМИ
# ==========================================
def _strip_version_specifier(requirement: str) -> str:
    # Отрезает версии и extras, оставляя чистое имя: "torch[cuda]>=2.0" -> "torch"
    name = re.split(r"[<>=!~\[;]", requirement, maxsplit=1)[0].strip()
    return name


def get_inference_pip_requirements(pyproject_path: str | Path) -> list[str]:
    pyproject_path = Path(pyproject_path)
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    try:
        declared = data["project"]["optional-dependencies"][_INFERENCE_GROUP]
    except KeyError:
        logger.warning(
            "Группа [project.optional-dependencies.%s] не найдена в %s. ",
            _INFERENCE_GROUP,
            pyproject_path,
        )
        return []

    pinned: list[str] = []
    for requirement in declared:
        pkg_name = _strip_version_specifier(requirement)
        try:
            installed_version = version(pkg_name)
            pinned.append(f"{pkg_name}=={installed_version}")
        except PackageNotFoundError:
            logger.warning("Пакет '%s' не установлен — пропускаю.", pkg_name)

    return pinned


# ==========================================
# БЛОК 2: ЗАГРУЗКА АДАПТЕРОВ (RESUME)
# ==========================================
def _ensure_tracking_uri() -> None:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    else:
        logger.warning("MLFLOW_TRACKING_URI не задан. Используется ./mlruns")


def _find_adapter_config(root: Path) -> Path | None:
    """Ищет adapter_config.json рекурсивно — на случай вложенных структур артефактов."""
    for candidate in [root, root / "peft", root / "lora_weights", root / "adapter"]:
        if (candidate / "adapter_config.json").exists():
            return candidate
    matches = list(root.rglob("adapter_config.json"))
    if matches:
        return matches[0].parent
    return None


def resolve_lora_resume_path(
    resume_cfg: DictConfig | dict,
    tracking_uri: str | None = None,
) -> str | None:
    """Разрешает путь к PEFT-адаптеру из MLflow artifacts."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("MLflow tracking URI: %s", tracking_uri)
    else:
        _ensure_tracking_uri()

    if isinstance(resume_cfg, DictConfig):
        resume_cfg = OmegaConf.to_container(resume_cfg, resolve=True)

    if not resume_cfg or not resume_cfg.get("enabled", False):
        return None

    run_id = resume_cfg.get("run_id")
    model_name = resume_cfg.get("model_name")
    alias = resume_cfg.get("alias")
    artifact_path = resume_cfg.get("artifact_path", "lora_weights")

    if run_id:
        logger.info("Скачивание LoRA адаптера по run_id=%s artifact_path=%s", run_id, artifact_path)
        downloaded = Path(
            mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
        )
    elif model_name and alias:
        logger.info("Поиск LoRA в Registry: %s (alias='%s')", model_name, alias)
        client = MlflowClient()
        try:
            model_version = client.get_model_version_by_alias(model_name, alias)
        except MlflowException as e:
            raise MlflowException(
                f"Алиас '{alias}' не найден для модели '{model_name}': {e.message}"
            ) from e

        logger.info(
            "Найдена версия %s (run_id=%s, source=%s), скачиваем...",
            model_version.version,
            model_version.run_id,
            model_version.source,
        )

        if model_version.run_id:
            downloaded = Path(
                mlflow.artifacts.download_artifacts(
                    run_id=model_version.run_id, artifact_path=artifact_path
                )
            )
        elif model_version.source:
            downloaded = Path(
                mlflow.artifacts.download_artifacts(artifact_uri=model_version.source)
            )
        else:
            raise MlflowException(
                f"Версия {model_version.version} модели '{model_name}' "
                f"не содержит ни run_id, ни source URI."
            )
    else:
        raise ValueError("Укажите 'run_id' или комбинацию 'model_name' + 'alias' для резьюма.")

    adapter_dir = _find_adapter_config(downloaded)
    if adapter_dir is None:
        raise FileNotFoundError(
            f"adapter_config.json не найден в скачанных артефактах: {downloaded}\n"
            f"Содержимое: {list(downloaded.rglob('*'))}"
        )

    logger.info("PEFT-адаптер найден: %s", adapter_dir)
    return str(adapter_dir)


# ==========================================
# БЛОК 3: СОХРАНЕНИЕ АДАПТЕРОВ (LOGGING)
# ==========================================
def _patch_peft_config_for_hydra(model: Any) -> None:
    if not hasattr(model, "peft_config"):
        return

    for _, peft_cfg in model.peft_config.items():
        for key, value in vars(peft_cfg).items():
            if isinstance(value, (ListConfig, DictConfig)):
                setattr(peft_cfg, key, OmegaConf.to_container(value, resolve=True))


def _build_reg_model_name(mlflow_model_name: str, adapter_type: str = "LoRA") -> str:
    """Единственное место, где собирается имя модели в Registry."""
    return f"{mlflow_model_name}_{adapter_type}"


def _save_adapter_to_tempdir(model_to_save: Any, tokenizer: Any, tmp_path: Path) -> None:
    """Сохраняет веса адаптера во временную папку."""
    logger.info("Сохранение PEFT-адаптера во временную директорию: %s", tmp_path)
    model_to_save.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)

    if not (tmp_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"save_pretrained не создал adapter_config.json в {tmp_path}. "
            f"Содержимое: {list(tmp_path.iterdir())}"
        )

    saved_files = [f.name for f in tmp_path.iterdir()]
    logger.info("Файлы адаптера: %s", saved_files)


def _log_artifacts_to_run(run_id: str, tmp_path: Path, artifact_path: str) -> str:
    """Загружает артефакты в MLflow и возвращает их URI в формате runs:/."""
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifacts(str(tmp_path), artifact_path=artifact_path)
        logger.info("LoRA адаптер сохранён в run_id=%s artifact_path=%s", run_id, artifact_path)

    # Возвращаем runs:/ URI вместо локального file:// пути,
    # чтобы Model Registry мог корректно зарегистрировать модель.
    return f"runs:/{run_id}/{artifact_path}"


def _register_model_version(client: MlflowClient, artifact_uri: str, reg_model_name: str) -> str:
    """Регистрирует новую версию модели в Model Registry."""
    mv_version = mlflow.register_model(
        model_uri=artifact_uri,
        name=reg_model_name,
    ).version
    logger.info("Зарегистрирована '%s' версия %s.", reg_model_name, mv_version)
    return mv_version


def log_lora_to_mlflow(
    cfg: Any,
    model_module: Any,
    tokenizer: Any,
    run_id: str,
    pipeline_name: str,
    best_score: float | None = None,
) -> None:
    """Сохраняет PEFT LoRA-адаптер в MLflow через save_pretrained + log_artifacts.

    Args:
        cfg: Корневой конфиг Hydra.
        model_module: Lightning-модуль с атрибутом .model.
        tokenizer: Токенизатор для сохранения вместе с адаптером.
        run_id: MLflow run ID активного эксперимента.
        pipeline_name: Имя пайплайна («rag_pipeline» или «decoder_pipeline»).
            Используется для разрешения пути к mlflow_model_name в конфиге —
            зеркально тому, как это делает yaml:
            ``model_name: ${${pipeline_name}.model.architecture.mlflow_model_name}``
        best_score: Лучшее значение val_loss — логируется как тег версии.
    """
    import tempfile

    logger.info("Подготовка к сохранению LoRA-адаптера в MLflow (run_id=%s)...", run_id)

    gc.collect()
    register_safe_globals()

    model_to_save = model_module.model
    client = MlflowClient()

    # Берём имя модели через pipeline_name — зеркально yaml-конфигу:
    # model_name: ${${pipeline_name}.model.architecture.mlflow_model_name}
    pipeline_cfg = OmegaConf.select(cfg, pipeline_name)
    if pipeline_cfg is None:
        raise ValueError(
            f"Пайплайн '{pipeline_name}' не найден в конфиге. Доступные ключи: {list(cfg.keys())}"
        )
    mlflow_model_name = pipeline_cfg.model.architecture.mlflow_model_name
    reg_model_name = _build_reg_model_name(mlflow_model_name)

    registry_cfg = cfg.get("logger", {}).get("registry", {})
    artifact_path = registry_cfg.get("artifact_path", "lora_weights")
    register_on_success = registry_cfg.get("register_on_success", True)
    promote_to_staging = registry_cfg.get("promote_to_staging", True)

    logger.info("Registry модель: %s | artifact_path: %s", reg_model_name, artifact_path)

    _patch_peft_config_for_hydra(model_to_save)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Шаг 1. Сохраняем локально во временную директорию
        _save_adapter_to_tempdir(model_to_save, tokenizer, tmp_path)

        # Шаг 2. Логируем артефакты в run (используем ваш чистый log_artifacts)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifacts(str(tmp_path), artifact_path=artifact_path)
            logger.info("LoRA адаптер сохранён в run_id=%s artifact_path=%s", run_id, artifact_path)

            if best_score is not None:
                mlflow.log_metric("promotion_candidate_val_loss", best_score)

    if not register_on_success:
        logger.info("Регистрация в Model Registry отключена (register_on_success=false).")
        return

    # Шаг 3. Безопасная регистрация версии через клиента MLflow (исправленный вариант)
    # Убеждаемся, что зарегистрированная модель существует
    try:
        client.create_registered_model(reg_model_name)
    except Exception:
        pass  # Модель уже существует, это нормально

    # Формируем правильный URI для артефакта внутри рана
    model_uri = f"runs:/{run_id}/{artifact_path}"

    # Создаем версию модели напрямую через client, что корректно обрабатывает runs:/ пути для произвольных артефактов
    try:
        model_version_obj = client.create_model_version(
            name=reg_model_name,
            source=model_uri,
            run_id=run_id,
        )
        mv_version = model_version_obj.version
    except Exception as e:
        # Fallback на случай специфических версий MLflow
        logger.warning(
            "Не удалось зарегистрировать через client.create_model_version (%s). Пробуем mlflow.register_model...",
            e,
        )
        mv_version = _register_model_version(client, model_uri, reg_model_name)

    logger.info("Зарегистрирована '%s' версия %s.", reg_model_name, mv_version)

    # Шаг 4. Навешивание алиасов и тегов (вся ваша логика сохранена)
    if promote_to_staging:
        client.set_registered_model_alias(name=reg_model_name, alias="Staging", version=mv_version)
        logger.info("Модель '%s' версии %s помечена алиасом 'Staging'.", reg_model_name, mv_version)

    if best_score is not None:
        client.set_model_version_tag(reg_model_name, mv_version, "val_loss", str(best_score))
