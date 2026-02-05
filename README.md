# 🎧 극한 소음 환경에서의 본전도 기반 음성 향상 (Speech Enhancement)

본 프로젝트는 **극한의 소음 환경(공사장, 공장, 도로 등)**에서도 명료한 음성 통신을 가능하게 하는 것을 목표로 합니다. 특히 공기 전도(Air Conducted) 소음에 강인한 **본전도(Bone Conduction) 마이크** 신호를 활용하여, 오염된 음성 신호로부터 깨끗한 음성을 복원하는 딥러닝 모델을 연구 및 개발합니다.

---

## 🎯 프로젝트 목표 (Objective)

1.  **극한 소음 극복**: 단순한 노이즈 캔슬링을 넘어, SNR이 극도로 낮은 환경에서의 음성 명료도 확보.
2.  **본전도 센서 활용 (User Bone Conduction)**:
    *   사용자의 성대 진동을 직접 감지하는 본전도 센서의 특성을 활용.
    *   외부 소음이 차단된 본전도 신호를 가이드(Reference)로 사용하여 음성 복원 성능 극대화.
3.  **경량화 및 최적화**: 로컬 연구 환경(On-Premise)에서 최소한의 의존성으로 최대의 효율을 내는 파이프라인 구축.

---

## 🏗️ 시스템 아키텍처 (System Architecture)

이 프로젝트는 **"Less Dependencies, More Reproducibility"** 철학을 바탕으로 설계되었습니다. 복잡한 의존성을 제거하고, 각 도구의 본질적인 기능에 집중합니다.

### 🛠️ Tech Stack
- **Environment**: `uv` (Rust 기반의 초고속 Python 패키지 매니저)
- **Framework**: `PyTorch Lightning` (학습 루프 및 시스템 구조화) + `LightningCLI` (설정 자동화)
- **Tracking**: `MLflow` (실험 결과 및 아티팩트 자동 기록)
- **Data Management**: `SQLModel` + `SQLite` (가볍고 강력한 로컬 메타데이터 관리)

---

### 📂 폴더 구조 (Directory Structure)

프로젝트는 **시스템(설정)**, **데이터**, **코드(모델/로직)**가 명확히 분리된 구조를 따릅니다.

```text
/
├── configs/            # ⚙️ 실험 설정 (YAML)
├── data/               # 💾 데이터 저장소
│   ├── raw/            # 원본 데이터 (Speech, Noise)
│   ├── rirs/           # 시뮬레이션된 Room Impulse Responses (.pkl)
│   ├── samples/        # 합성 로직 검증용 오디오 샘플
│   ├── outputs/        # 딥러닝 모델 출력물 (Validation/Test 결과)
│   └── metadata.db     # SQLite 데이터베이스 (인덱싱된 메타데이터)
│
├── src/                # 💻 소스 코드
│   ├── models/         # [Pure PyTorch] 모델 아키텍처 (BaseSEModel 등)
│   ├── modules/        # [Lightning] 학습 로직 및 시스템
│   ├── data/           # [Lightning] 데이터 파이프라인 (Dataset, DataLoader)
│   ├── db/             # DB 관리 코드 (SQLModel 스키마 및 Manager)
│   └── simulation/     # 음향 시머니레이션 (RIR 생성, 믹싱 로직)
│
├── mlruns/             # 📊 MLflow 실험 데이터 (로컬 파일시스템)
├── scripts/            # 📜 유틸리티 스크립트
│   ├── manage_db.py         # 🗄️ 통합 DB 관리 CLI
│   ├── generate_rir_bank.py  # 🏟️ RIR 대량 시뮬레이션 생성
│   ├── visualize_rirs.py    # 🎨 RIR 시뮬레이션 결과 시각화
│   ├── utils/              
│   │   └── convert_pcm_to_wav.py # 🔄 PCM -> WAV 고속 변환기
│   └── tests/
│       └── test_base_model.py    # 🧪 모델 아키텍처 검증 (Perfect Reconstruction)
│
├── docs/               # 📚 상세 문서 (가이드라인)
│   ├── Database_Management_Guide.md  # DB 상세 관리 및 SQLModel 사용법
│   ├── RIR_Simulation_Guide.md      # RIR 생성 및 메타데이터 구조 가이드
│   ├── Data_Synthesis_Guide.md      # 온더플라이 데이터 합성 가이드
│   ├── Base_Model_Architecture_Guide.md # 모델 아키텍처 설계 및 구현 가이드
│   └── Git_Sync_Guide.md            # 📘 사외/사내 망 Git 동기화 가이드
├── main.py             # 🚀 실행 엔트리포인트 (LightningCLI)
└── pyproject.toml      # 📦 의존성 명세서 (uv)
```

---

## 🚀 워크플로우 (Research Workflow)

### 1. 환경 설정
`uv`를 사용하여 모든 의존성을 한 번에 동기화합니다.
```bash
uv sync
```

### 2. 데이터 준비 (Preprocessing)
원본 데이터를 DB에 등록하고, 고속 학습을 위한 RIR을 생성합니다.

```bash
# 통합 DB 관리 도구 사용 (음성, 소음, RIR 순차 등록)
uv run python3 scripts/manage_db.py speech --path data/raw/speech/KsponSpeech --dataset KsponSpeech --language "ko"
uv run python3 scripts/manage_db.py noise --path data/raw/noise/traffic --dataset "TrafficNoise" --category "교통수단"
uv run python3 scripts/manage_db.py rir --path data/rirs --dataset "SimRIR_v1"

# 8:1:1 데이터 분할 자동 실행
uv run python3 scripts/manage_db.py realloc --type speech
uv run python3 scripts/manage_db.py realloc --type noise

# (Optional) 모든 인덱싱 과정을 한 번에 수행하는 자동화 스크립트
uv run python3 scripts/final_indexing_v2.py
```

상세한 데이터베이스 관리 방법은 [Database_Management_Guide.md](docs/Database_Management_Guide.md)를 참고하세요.

### 3. 모델 학습 및 검증 (Training & Verification)
학습 파이프라인 구축 및 검증이 완료되었습니다. `generate_samples.py`를 통해 합성 결과를 미리 확인할 수 있습니다.

```bash
# 합성 샘플 생성 (CPU 검증용)
uv run python3 scripts/generate_samples.py --num 10 --split val

# 모델 학습 실행 (LightningCLI)
uv run python3 main.py fit --config configs/ic_conv_tasnet.yaml
```

### 4. 실험 분석 (Tracking)
로컬 파일시스템에 기록된 실험 결과를 MLflow UI를 통해 확인합니다.
```bash
uv run mlflow ui
```

---

## 📊 데이터셋 (Datasets)

- **Speech (Target/Reference)**: KsponSpeech (한국어 대화 음성)
- **Noise (Interference)**: NIA 163-2 극한 소음 데이터 (공사장, 공장, 교통 소음 등)
- **RIRs (Augmentation)**: 시뮬레이션된 공간 임펄스 응답