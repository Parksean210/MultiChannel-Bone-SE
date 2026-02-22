#!/bin/bash
# MLflow 서버 시작 스크립트
# 사용법: bash mlflow_server.sh

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

pkill -f "mlflow ui" 2>/dev/null
sleep 1

nohup uv run mlflow ui \
  --backend-store-uri "sqlite:///${PROJECT_DIR}/results/mlflow.db" \
  --default-artifact-root "file://${PROJECT_DIR}/results/mlruns" \
  --host 0.0.0.0 \
  --port 5000 \
  > "${PROJECT_DIR}/results/mlflow.log" 2>&1 &

echo "MLflow 서버 시작: http://$(hostname -I | awk '{print $1}'):5000"
