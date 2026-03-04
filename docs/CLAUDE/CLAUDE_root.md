# Speech Enhancement Framework

## Project Overview
극한 소음 환경(공사장, 공장, 도로)에서 골전도(Bone Conduction) 마이크 신호를 활용한 음성 향상(Speech Enhancement) 딥러닝 프레임워크.
AR 글래스에 장착된 4채널 공기전도 마이크 + 1채널 골전도 센서(BCM) = 총 5채널 다채널 입력을 처리한다.

## Tech Stack
- **Framework**: PyTorch Lightning 2.x + LightningCLI (YAML 기반 설정 관리)
- **Experiment Tracking**: MLflow (오디오/스펙트로그램 아티팩트 자동 로깅, 태그, 체크포인트 비교)
- **Database**: SQLite + SQLModel (메타데이터 인덱싱)
- **Simulation**: pyroomacoustics (방 임펄스 응답 생성)
- **Package Manager**: uv
- **Python**: >= 3.10

## Architecture

### Data Flow
```
Raw Audio (.npy/.wav) + RIR (.pkl) + Noise (.npy/.wav)
    -> SQLite DB 인덱싱 (scripts/manage_db.py)
    -> SpatialMixingDataset (CPU: 파일 로드 + random crop + Data Augmentation)
    -> src/utils/synthesis.py (GPU: FFT Convolution + BCM 모델링 + SNR 스케일링)
    -> Model Forward (ICConvTasNet 등)
    -> Loss (SI-SDR + Multi-Resolution STFT)
    -> Metrics (SI-SDR, SDR, STOI, PESQ)
```

### Key Design Decisions
1. **GPU 기반 실시간 합성**: RIR convolution과 노이즈 믹싱을 GPU에서 수행하여 매 에폭 다른 조합 생성 (Data Augmentation 효과)
2. **YAML 중심 설정**: LightningCLI를 통해 모델/손실함수/옵티마이저를 YAML에서 class_path로 지정
3. **채널 0 기준 평가**: 안경 전면 참조 마이크(Channel 0)에 대해서만 손실 계산 및 지표 측정
4. **BCM ablation**: `configs/ic_conv_tasnet_bcm_off.yaml` 사용 (in_channels=4, BCM 채널 제외)
5. **BCM 커널 캐싱**: `register_buffer`로 sinc+hann LPF 커널을 1회 생성하여 매 스텝 재계산 방지
6. **LR Warmup + Cosine Annealing**: LinearLR(5 epoch) + CosineAnnealingLR로 안정적 학습
7. **Gradient Clipping**: `gradient_clip_val: 5.0`으로 학습 안정성 확보
8. **Data Augmentation**: Speed Perturbation(0.9~1.1x) + Gain(-6~+6dB), train split에서만 적용

## Directory Structure
```
configs/          - YAML 실험 설정 파일
  baseline.yaml               - DummyModel 파이프라인 검증용 기준선
  ic_conv_tasnet.yaml         - 메인 모델 (5ch, BCM 포함)
  ic_conv_tasnet_bcm_off.yaml - BCM ablation (4ch, BCM 제외)
data/             - 원본 데이터(speech, noise), RIR, SQLite DB
docs/             - 상세 가이드 문서
scripts/          - CLI 유틸리티 (DB 관리, RIR 생성, 오디오 변환)
src/              - 핵심 소스 코드
  models/         - [Pure PyTorch] 모델 아키텍처 (BaseSEModel 상속)
  modules/        - [Lightning] 학습 시스템 (SEModule, 손실함수)
  data/           - [Lightning] 데이터 파이프라인 (Dataset, DataModule)
  db/             - SQLite 엔진 및 관리자
  callbacks/      - Lightning 콜백 (GPU 모니터, 오디오 저장, MLflow 태깅)
  simulation/     - pyroomacoustics 기반 RIR 시뮬레이션
  utils/          - 공유 유틸리티 (합성 엔진, 오디오 I/O, 메트릭)
test_model/       - 오픈소스 모델 분석 공간 (프로덕션 코드 아님)
results/          - 실험 결과 (체크포인트, MLflow, 추론 오디오)
```

## Common Commands
```bash
# 환경 설정
uv sync

# Mamba 계열 모델 (ICMamba, ICMamba2BCMGuide 등) 사용 시 추가 설치 필요
# mamba-ssm / causal-conv1d 는 CUDA 버전 + Python 버전 + PyTorch 버전에 맞는
# .whl 파일을 직접 다운로드해서 설치해야 한다 (pip install 로 소스 빌드 시 실패 가능)
#
# 다운로드: https://github.com/state-spaces/mamba/releases
#           https://github.com/Dao-AILab/causal-conv1d/releases
#
# 예시 (CUDA 12.1 / Python 3.10 / torch 2.x):
#   mamba_ssm-2.x.x+cu121torch2.x-cp310-cp310-linux_x86_64.whl
#   causal_conv1d-1.x.x+cu121torch2.x-cp310-cp310-linux_x86_64.whl
#
# 설치:
uv pip install /path/to/causal_conv1d-*.whl
uv pip install /path/to/mamba_ssm-*.whl

# 데이터 인덱싱
uv run python scripts/manage_db.py speech --path data/speech/KsponSpeech --dataset KsponSpeech
uv run python scripts/manage_db.py noise --path data/noise/... --dataset "극한소음"
uv run python scripts/manage_db.py rir --path data/rirs --dataset "SimRIR"
uv run python scripts/manage_db.py realloc --type speech

# 학습
uv run python main.py fit --config configs/ic_conv_tasnet.yaml

# BCM 유무 비교 실험 (전용 config 사용)
uv run python main.py fit --config configs/ic_conv_tasnet_bcm_off.yaml

# 추론
uv run python main.py predict --config configs/ic_conv_tasnet.yaml --ckpt_path <path>

# 테스트
uv run python main.py test --config configs/ic_conv_tasnet.yaml --ckpt_path <path>

# MLflow UI
bash mlflow_server.sh
```

## MLflow Experiment Structure
MLflow run은 세 개의 experiment로 분리 관리된다.

| Experiment | 목적 | Config |
|---|---|---|
| `Architecture` | 모델 아키텍처 비교 (IC-ConvTasNet vs IC-Mamba 등) | `ic_conv_tasnet.yaml`, `baseline.yaml` |
| `BCM-Ablation` | BCM 채널 유무에 따른 성능 비교 | `ic_conv_tasnet_bcm_off.yaml` |
| `Hyperparameter` | 학습률, 손실 가중치 등 하이퍼파라미터 탐색 | (별도 config 작성) |

### 자동 태깅 (MLflowAutoTagCallback)
학습 시작 시 run에 다음 태그를 자동으로 기록한다:
- `git_commit` — 재현성을 위한 커밋 해시
- `git_dirty` — 미커밋 변경사항 존재 여부
- `model_type` — 모델 클래스명 (ICConvTasNet, ICMamba 등)
- `in_channels` — 입력 채널 수 (5: BCM 포함, 4: BCM 제외)
- `target_type` — aligned_dry / spatialized
- `sample_rate` — 샘플 레이트

YAML config 파일도 `config/` 아티팩트로 자동 저장된다.

## 새 모델 추가하기

새 아키텍처를 실험할 때 수정할 파일은 딱 3개다.

### 1. 모델 파일 생성 — `src/models/new_model.py`
```python
from .base import BaseSEModel

class MyNewModel(BaseSEModel):
    def __init__(self, in_channels: int = 5, ...):
        super().__init__(in_channels=in_channels, n_fft=..., hop_length=...)
        # 레이어 정의

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, M, T) → return: (B, M, T)
        ...
```

### 2. `__init__` 등록 — `src/models/__init__.py`
```python
from .my_new_model import MyNewModel  # 한 줄 추가
```

### 3. YAML config 생성 — `configs/my_new_model.yaml`
기존 config(`ic_conv_tasnet.yaml` 등)를 복사한 뒤 아래 부분만 수정:
```yaml
model:
  class_path: src.models.MyNewModel   # 클래스 경로
  init_args:
    in_channels: 5
    ...                                # 모델 하이퍼파라미터
```

데이터 파이프라인, 학습 루프, 로깅, 콜백은 수정 불필요.

## Coding Conventions
- 모델 클래스는 반드시 `BaseSEModel`을 상속하고 `in_channels` 속성과 `forward(x: (B, C, T)) -> (B, C, T)` 인터페이스를 준수
- LightningModule(`SEModule`)과 Pure PyTorch 모델은 분리하여 관리
- 공유 로직(합성, 오디오 I/O, 메트릭)은 `src/utils/`에 구현하고 SEModule, 스크립트에서 import
- 텐서 차원 표기: `(B, C, T)` = (Batch, Channel, Time), `(B, C, F, T)` = (Batch, Channel, Freq, Time)
- 재현성 보장: `seed_everything: 42` + DB 쿼리 결과 ID 기반 정렬
- 손실함수/모델 추가 시 YAML `class_path`로 등록하면 코드 수정 없이 사용 가능
- `snr` 등 Python float를 collate한 텐서는 float64가 될 수 있으므로 사용 전 `.float()` 명시 (16-mixed AMP 충돌 방지)
