#!/bin/bash
# MLflow 서버 시작 스크립트
# 사용법: bash mlflow_server.sh

pkill -f "mlflow ui" 2>/dev/null
sleep 1

nohup uv run mlflow ui \
  --backend-store-uri "file:./results/mlruns" \
  --host 0.0.0.0 \
  --port 6006 \
  > results/mlflow.log 2>&1 &

echo "MLflow 서버 시작: http://localhost:16006"
