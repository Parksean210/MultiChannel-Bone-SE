# 빠른 시작 가이드 (Quick Start)

> 이 가이드는 환경 설정부터 첫 번째 학습 실행까지 단계별로 안내합니다.
> 데이터가 준비된 상태에서 약 10~15분이면 학습을 시작할 수 있습니다.

---

## 사전 요구사항

| 항목 | 최소 사양 | 검증된 환경 |
|---|---|---|
| GPU | VRAM 8GB 이상 | RTX 3080 (10GB) |
| RAM | 16GB | 32GB 이상 |
| 디스크 | 50GB (데이터 별도) | 200GB+ |
| CUDA | 12.x | 12.6 |
| cuDNN | 8.x 이상 | 9.5.1 |
| Python | 3.10 (cp310 고정) | 3.10 |
| PyTorch | 2.7.x | 2.7.1+cu126 |
| mamba-ssm | 2.3.0 | 2.3.0 (로컬 whl) |
| causal-conv1d | 1.6.0 | 1.6.0 (로컬 whl) |

> **주의**: mamba-ssm과 causal-conv1d는 `wheels/` 디렉토리의 사전 빌드 whl을 사용합니다. Python 버전이 cp310이 아니거나 CUDA/PyTorch 버전이 다르면 별도 whl을 빌드해야 합니다.

---

## Step 1. 환경 설정

### 1-1. uv 설치 (처음 한 번만)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # 또는 source ~/.zshrc
```

### 1-2. 의존성 설치

프로젝트 루트에서 실행합니다.

```bash
cd /path/to/speech_enhancement
uv sync
```

`uv sync`는 `pyproject.toml`을 읽어 `.venv/`에 가상환경을 자동 생성하고 모든 패키지를 설치합니다.

### 1-3. 설치 확인

```bash
uv run python -c "
import torch
print('PyTorch:', torch.__version__)        # 2.7.1+cu126
print('CUDA:', torch.cuda.is_available())   # True
import mamba_ssm; print('mamba-ssm:', mamba_ssm.__version__)          # 2.3.0
import causal_conv1d; print('causal-conv1d:', causal_conv1d.__version__)  # 1.6.0
"
```

---

## Step 2. 데이터 준비

### 2-1. 디렉토리 구조 만들기

```bash
mkdir -p data/speech data/noise data/rirs
```

### 2-2. 데이터 배치

각 데이터를 다음 위치에 넣습니다.

```
data/
├── speech/
│   └── KsponSpeech/        # 음성 데이터 (.wav 또는 .npy)
├── noise/
│   └── traffic/            # 소음 데이터 (.wav 또는 .npy)
└── rirs/
    └── rir_00001.pkl        # Room Impulse Response (.pkl)
```

**파일 형식**: `.wav` (16kHz 모노) 또는 `.npy` 모두 지원합니다.

### 2-3. DB에 데이터 등록

데이터를 SQLite DB에 인덱싱합니다. 실제 파일을 이동시키지 않고 경로만 등록합니다.

```bash
# 음성 데이터 등록
uv run python scripts/manage_db.py speech \
    --path data/speech/KsponSpeech \
    --dataset KsponSpeech \
    --language ko

# 소음 데이터 등록
uv run python scripts/manage_db.py noise \
    --path data/noise/traffic \
    --dataset TrafficNoise \
    --category "교통소음" \
    --sub "도로"

# RIR 등록
uv run python scripts/manage_db.py rir \
    --path data/rirs \
    --dataset SimRIR_v1

# 등록 현황 확인
uv run python scripts/manage_db.py stats
```

### 2-4. Train/Val/Test 분할

```bash
uv run python scripts/manage_db.py realloc --type speech --ratio 0.8 0.1 0.1
uv run python scripts/manage_db.py realloc --type noise  --ratio 0.8 0.1 0.1
uv run python scripts/manage_db.py realloc --type rir    --ratio 0.8 0.1 0.1
```

> **중요**: `realloc`은 기존에 `val/test`로 지정된 데이터를 절대 `train`으로 되돌리지 않습니다. 새 데이터를 추가해도 기존 검증셋이 유지됩니다.

---

## Step 3. RIR 시뮬레이션 (RIR이 없는 경우)

RIR 파일이 없다면 `pyroomacoustics`로 가상 음향 공간을 시뮬레이션합니다.

```bash
# 1,000개 RIR 생성 (배경 실행, 약 10~30분 소요)
uv run python scripts/generate_rir_bank.py --count 1000

# 생성 후 DB 등록
uv run python scripts/manage_db.py rir --path data/rirs --dataset SimRIR_v1
uv run python scripts/manage_db.py realloc --type rir
```

---

## Step 4. 학습 실행

### 파이프라인 검증 (처음 실행 시 권장)

본격 학습 전, 전체 파이프라인이 올바르게 동작하는지 빠르게 확인합니다.

```bash
uv run python main.py fit \
    --config configs/baseline.yaml \
    --trainer.limit_train_batches=10 \
    --trainer.limit_val_batches=2 \
    --trainer.max_epochs=1
```

오류 없이 완료되면 파이프라인이 정상입니다.

### 메인 모델 학습

```bash
uv run python main.py fit --config configs/ic_conv_tasnet.yaml
```

학습이 시작되면 터미널에서 실시간으로 진행 상황을 확인할 수 있습니다.

```
Epoch 0:  42%|████      | 3300/7857 [05:12<07:10,  10.60it/s, v_num=..., train_loss_step=8.32]
```

---

## Step 5. MLflow 대시보드

별도 터미널에서 MLflow UI를 실행합니다.

```bash
nohup uv run mlflow ui \
    --backend-store-uri file:./results/mlruns \
    --host 0.0.0.0 \
    --port 5000 \
    > /dev/null 2>&1 &

echo "MLflow UI: http://localhost:5000"
```

브라우저에서 `http://localhost:5000`에 접속하면 실시간 메트릭과 아티팩트를 확인할 수 있습니다.

---

## 학습 중 확인 사항

### 정상 학습 신호

- `train_loss`가 에폭 진행에 따라 감소
- `val_si_sdr`이 점진적으로 증가 (음수 → 양수 방향)
- MLflow에서 `git_commit`, `model_type` 태그 확인

### 문제 진단

| 증상 | 원인 | 해결 방법 |
|---|---|---|
| `CUDA out of memory` | VRAM 부족 | `configs/`에서 `batch_size` 줄이기 |
| `No data found` | DB에 split 미지정 | `manage_db.py realloc` 실행 |
| `val_loss` 무한 상승 | learning rate 너무 높음 | `optimizer_config.lr: 1e-4`로 낮추기 |
| `train_loss`가 안 줄어듦 | 데이터 파이프라인 문제 | baseline.yaml로 파이프라인 재검증 |

---

## 체크포인트 위치

학습 완료 후 체크포인트는 다음 위치에 저장됩니다.

```
results/
└── lightning_logs/
    └── version_X/
        └── checkpoints/
            └── best-model-epoch=XX-val_loss=X.XX.ckpt
```

또는 MLflow UI → 해당 Run → Artifacts에서도 확인할 수 있습니다.

---

## 다음 단계

| 목표 | 참고 문서 |
|---|---|
| 데이터를 더 추가하고 싶다 | [DB 관리 가이드](02_Database_Management.md) |
| 학습 설정을 바꾸고 싶다 | [학습·추론 가이드](04_Training_and_Inference.md) |
| 실험 결과를 비교하고 싶다 | [MLflow 가이드](05_MLflow_Guide.md) |
| 새 모델을 만들고 싶다 | [모델 아키텍처 가이드](06_Model_Architecture.md) |
