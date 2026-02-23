# Speech Enhancement Framework

> 극한 소음 환경(공사장·공장·도로)에서 골전도(BCM) 마이크를 활용한 딥러닝 기반 다채널 음성 향상 연구 프레임워크

---

## 프로젝트 개요

AR 글래스에 장착된 **4채널 공기전도 마이크 + 1채널 골전도 센서(BCM)** = 총 5채널 입력을 처리하여 극한 소음에서도 음성을 복원합니다. 기존 노이즈 캔슬링과 달리, 피부·뼈를 통해 전달되어 외부 소음이 차단된 골전도 신호를 보조 참조(Reference)로 활용합니다.

---

## 빠른 시작

```bash
# 1. 의존성 설치
uv sync

# 2. 학습 시작
uv run python main.py fit --config configs/ic_mamba.yaml

# 3. MLflow 대시보드 (별도 터미널)
bash mlflow_server.sh
# 브라우저: http://<WSL_IP>:5000
```

처음 사용하신다면 **[빠른 시작 가이드](docs/CLAUDE/01_QuickStart.md)** 를 먼저 읽으세요.

---

## 시스템 구조

```
Raw Audio + Noise + RIR
        │  (CPU: Dataset)
        │  파일 로드 + 랜덤 크롭
        ▼
   DataLoader
        │  (GPU: src/utils/synthesis.py)
        │  FFT Convolution → BCM 모델링 → SNR 스케일링
        ▼
   Model (ICConvTasNet 등)
        │
        ▼
   Loss (SI-SDR + Multi-Res STFT)
        │
        ▼
   Metrics + MLflow 로깅
```

**핵심 설계 원칙**

| 원칙 | 내용 |
|---|---|
| GPU 실시간 합성 | RIR Convolution·믹싱을 GPU에서 수행 → 매 에폭 다른 데이터 조합 (Data Augmentation 효과) |
| YAML 중심 설정 | LightningCLI로 모델·손실·옵티마이저를 코드 수정 없이 YAML로 교체 |
| 모델 비교 용이 | `BaseSEModel` 인터페이스 준수 시 YAML 한 줄로 모델 교체 |
| BCM ablation | 전용 config(`ic_conv_tasnet_bcm_off.yaml`)으로 4ch/5ch 공정 비교 |

---

## 디렉토리 구조

```
speech_enhancement/
├── configs/                      # YAML 실험 설정
│   ├── baseline.yaml                  # DummyModel (파이프라인 검증용)
│   ├── ic_conv_tasnet.yaml            # ICConvTasNet (5ch, BCM 포함)
│   ├── ic_conv_tasnet_bcm_off.yaml    # ICConvTasNet BCM ablation (4ch)
│   ├── ic_mamba.yaml                  # ICMamba 메인 모델 (5ch, BCM 포함)
│   └── ic_mamba_bcm_off.yaml          # ICMamba BCM ablation (4ch)
│
├── data/                         # 데이터 저장소 (git 미추적)
│   ├── speech/                   # 클린 음성 (.wav/.npy)
│   ├── noise/                    # 환경 소음 (.wav/.npy)
│   ├── rirs/                     # Room Impulse Response (.pkl)
│   └── metadata.db               # SQLite 메타데이터 DB
│
├── src/
│   ├── models/                   # [Pure PyTorch] 모델 아키텍처
│   │   ├── base.py               # BaseSEModel (공통 인터페이스)
│   │   ├── ic_conv_tasnet.py     # ICConvTasNet (Dilated TCN 기반)
│   │   ├── ic_mamba.py           # ICMamba (mamba-ssm CUDA 커널, causal)
│   │   └── baseline.py           # DummyModel (파이프라인 검증)
│   ├── modules/                  # [Lightning] 학습 시스템
│   │   ├── se_module.py          # SEModule (학습/검증/테스트 루프)
│   │   └── losses.py             # CompositeLoss (SI-SDR + Freq)
│   ├── data/                     # [Lightning] 데이터 파이프라인
│   │   ├── dataset.py            # SpatialMixingDataset
│   │   └── datamodule.py         # SEDataModule
│   ├── callbacks/                # Lightning 콜백
│   │   ├── gpu_stats_monitor.py  # GPU 사용률/온도 모니터링
│   │   ├── audio_prediction_writer.py  # 추론 결과 WAV 저장
│   │   └── mlflow_auto_tag.py    # git/모델 메타데이터 자동 태깅
│   ├── utils/                    # 공유 유틸리티
│   │   ├── synthesis.py          # GPU 공간 합성 엔진
│   │   ├── audio_io.py           # 오디오 저장/변환/시각화
│   │   └── metrics.py            # 메트릭 계산·모델 비교
│   ├── db/                       # SQLite DB 엔진
│   └── simulation/               # pyroomacoustics RIR 시뮬레이터
│
├── scripts/                      # CLI 유틸리티
│   ├── manage_db.py              # DB 인덱싱/통계/동기화
│   ├── generate_rir_bank.py      # RIR 대량 생성
│   ├── generate_samples.py       # 합성 오디오 샘플 생성
│   ├── compare_checkpoints.py    # 체크포인트 간 성능 비교
│   ├── find_max_batch.py         # 최대 배치 사이즈 탐색
│   └── verify_pipeline.py        # 파이프라인 동작 검증
│
├── mlflow_server.sh              # MLflow 서버 시작 스크립트
│
├── results/                      # 실험 결과 (git 미추적)
│   ├── mlruns/                   # MLflow 실험 데이터
│   ├── predictions/              # 추론 결과 오디오
│   └── */checkpoints/            # 모델 체크포인트
│
├── docs/CLAUDE/                  # 상세 가이드 문서
├── test_model/                   # 오픈소스 모델 분석 공간 (비프로덕션)
├── main.py                       # LightningCLI 진입점
└── pyproject.toml                # 의존성 명세 (uv)
```

---

## 주요 명령어

### 데이터 준비

```bash
# DB 현황 확인
uv run python scripts/manage_db.py stats

# 음성 데이터 등록
uv run python scripts/manage_db.py speech \
    --path data/speech/KsponSpeech --dataset KsponSpeech

# 소음 데이터 등록
uv run python scripts/manage_db.py noise \
    --path data/noise/traffic --dataset TrafficNoise \
    --category "교통소음" --sub "도로"

# RIR 등록
uv run python scripts/manage_db.py rir \
    --path data/rirs --dataset SimRIR_v1

# Train/Val/Test 분할 (8:1:1)
uv run python scripts/manage_db.py realloc --type speech
uv run python scripts/manage_db.py realloc --type noise
uv run python scripts/manage_db.py realloc --type rir
```

### 학습

```bash
# IC-Mamba 학습 (5ch, BCM 포함) ← 메인
uv run python main.py fit --config configs/ic_mamba.yaml

# IC-Mamba BCM ablation (4ch)
uv run python main.py fit --config configs/ic_mamba_bcm_off.yaml

# ICConvTasNet (비교용)
uv run python main.py fit --config configs/ic_conv_tasnet.yaml

# 디버깅 (소량 배치)
uv run python main.py fit --config configs/ic_mamba.yaml \
    --trainer.limit_train_batches=10 \
    --trainer.limit_val_batches=2 \
    --trainer.max_epochs=1 \
    --trainer.devices=1 \
    --trainer.strategy=auto

# 최대 배치 사이즈 탐색 (config 기반)
uv run python scripts/find_max_batch.py --config configs/ic_mamba.yaml
```

### 추론 및 평가

```bash
# 추론 (결과 WAV 자동 저장)
uv run python main.py predict \
    --config configs/ic_conv_tasnet.yaml \
    --ckpt_path results/.../checkpoints/best-model.ckpt

# 테스트 (SI-SDR, PESQ 등 최종 평가)
uv run python main.py test \
    --config configs/ic_conv_tasnet.yaml \
    --ckpt_path results/.../checkpoints/best-model.ckpt

# 복수 체크포인트 비교
uv run python scripts/compare_checkpoints.py \
    --checkpoints ckpt_a.ckpt ckpt_b.ckpt \
    --num_samples 50
```

### MLflow

```bash
# 서버 시작 (SQLite 백엔드)
bash mlflow_server.sh

# 서버 종료
pkill -f "mlflow ui"
```

---

## MLflow 실험 구조

| Experiment | 목적 | 사용 Config |
|---|---|---|
| `Architecture` | 모델 아키텍처 비교 (IC-ConvTasNet vs IC-Mamba 등) | `ic_mamba.yaml`, `ic_conv_tasnet.yaml`, `baseline.yaml` |
| `BCM-Ablation` | BCM 채널 유무에 따른 성능 비교 | `ic_mamba_bcm_off.yaml`, `ic_conv_tasnet_bcm_off.yaml` |
| `Hyperparameter` | lr, loss weight 등 하이퍼파라미터 탐색 | 별도 config 작성 |

모든 run에는 다음 태그가 자동 기록됩니다: `git_commit`, `git_dirty`, `model_type`, `in_channels`, `target_type`, `sample_rate`

---

## 평가 지표

| 지표 | 설명 | 좋은 범위 |
|---|---|---|
| **SI-SDR** (dB) | 잡음 대비 음성 신호 강도. 핵심 지표 | 10 ~ 20 dB+ |
| **SDR** (dB) | 신호 대 왜곡 비율 | 10 ~ 20 dB+ |
| **PESQ** | 인지 음질 (Wide-Band) | 3.0 ~ 4.5 |
| **STOI** | 말소리 명료도 | 0.8 ~ 0.95 |

---

## 기술 스택

- **프레임워크**: PyTorch 2.7.1+cu126 + PyTorch Lightning 2.x + LightningCLI
- **주요 모델**: ICMamba (mamba-ssm CUDA 커널, causal SSM)
- **실험 추적**: MLflow (SQLite 백엔드, 자동 태깅, 오디오·스펙트로그램 아티팩트)
- **DB**: SQLite + SQLModel (메타데이터 인덱싱)
- **음향 시뮬레이션**: pyroomacoustics (RIR 생성)
- **패키지 관리**: uv
- **Python**: 3.10 (cp310)

### 검증된 패키지 버전 (2026-02-23 기준)

| 패키지 | 버전 | 비고 |
|---|---|---|
| PyTorch | 2.7.1+cu126 | |
| CUDA | 12.6 | |
| cuDNN | 9.5.1 | |
| mamba-ssm | 2.3.0 | 로컬 whl 설치 (아래 참고) |
| causal-conv1d | 1.6.0 | 로컬 whl 설치 (아래 참고) |
| GPU | RTX 3080 (10GB) | |

### mamba-ssm / causal-conv1d whl 설치

mamba-ssm과 causal-conv1d는 PyPI에서 제공하는 빌드가 환경에 따라 동작하지 않을 수 있어, `wheels/` 디렉토리에 사전 빌드된 whl 파일을 포함합니다.

```
wheels/
├── mamba_ssm-2.3.0+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
├── mamba_ssm-2.3.0+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
├── causal_conv1d-1.6.0+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
└── causal_conv1d-1.6.0+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

`pyproject.toml`의 `[tool.uv.sources]`에 `cxx11abiTRUE` whl이 고정되어 있으므로 `uv sync` 한 번으로 자동 설치됩니다.

> **cxx11abi 선택 기준**: GCC 5+ / glibc 2.17+ 환경(Ubuntu 18.04+)은 `TRUE`, 구형 환경은 `FALSE`.

---

## 문서 목록

| 문서 | 내용 |
|---|---|
| [빠른 시작](docs/CLAUDE/01_QuickStart.md) | 환경 설정 → 첫 학습까지 단계별 안내 |
| [DB 관리](docs/CLAUDE/02_Database_Management.md) | 데이터 등록, 분할, 스키마 상세 |
| [데이터 합성](docs/CLAUDE/03_Data_Synthesis.md) | GPU 실시간 합성 파이프라인 원리 |
| [학습·추론](docs/CLAUDE/04_Training_and_Inference.md) | 실행 명령어, YAML 설정, CLI 오버라이드 |
| [MLflow 가이드](docs/CLAUDE/05_MLflow_Guide.md) | 실험 추적, 대시보드, 모델 비교 |
| [모델 아키텍처](docs/CLAUDE/06_Model_Architecture.md) | BaseSEModel 인터페이스, 새 모델 추가 방법 |
| [RIR 시뮬레이션](docs/CLAUDE/07_RIR_Simulation.md) | pyroomacoustics 기반 공간 음향 생성 |
| [Git 동기화](docs/CLAUDE/08_Git_Sync_Guide.md) | 사외·사내망 Git 동기화 및 브랜치 전략 |
| [오디오 변환 도구](docs/CLAUDE/09_Audio_Tool_Guide.md) | PCM·WAV·NPY 변환, 리샘플링 (데이터 전처리) |
