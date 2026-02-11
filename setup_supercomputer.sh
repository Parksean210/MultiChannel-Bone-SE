#!/bin/bash

# 슈퍼컴퓨터 환경 설정 스크립트
# 사용법: source setup_supercomputer.sh

echo "Loading modules..."
# 사내 환경에서 제공하는 정확한 모듈 이름을 입력하세요.
module load python/3.10.14
module load cuda/12.1.0
module load ffmpeg/07.20

# 가상환경 활성화 (uv 또는 venv)
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Virtual environment activated."
else
    echo "Warning: .venv not found. Please run 'uv sync' first."
fi

# 내부 패키지 인덱스가 있는 경우 환경 변수 설정 (예시)
# export PIP_INDEX_URL=https://your-internal-mirror/simple

echo "Environment setup complete."
