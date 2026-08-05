"""Запуск MLflow UI с автовыбором backend по окружению."""

import os
import subprocess


tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
host = os.environ.get("MLFLOW_HOST", "127.0.0.1")
port = os.environ.get("MLFLOW_PORT", "5000")
db = os.environ.get("MLFLOW_DB", "sqlite:///logs/mlflow.db")
artifacts = os.environ.get("MLFLOW_ARTIFACTS", "./logs/mlartifacts")

if tracking_uri:
    print(f"Режим: remote ({tracking_uri})")
    cmd = ["mlflow", "ui", "--backend-store-uri", tracking_uri, "--host", host, "--port", port]
else:
    print(f"Режим: local ({db})")
    cmd = [
        "mlflow",
        "ui",
        "--backend-store-uri",
        db,
        "--default-artifact-root",
        artifacts,
        "--host",
        host,
        "--port",
        port,
    ]

print(f"Запуск MLflow UI на {host}:{port}...")
subprocess.run(cmd)
