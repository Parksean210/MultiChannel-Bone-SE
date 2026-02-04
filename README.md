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
│   └── Base_Model_Architecture_Guide.md # 모델 아키텍처 설계 및 구현 가이드
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
uv run python scripts/manage_db.py speech --path data/raw/speech/train --dataset KsponSpeech
uv run python scripts/manage_db.py noise --path data/raw/noise/train --category urban
uv run python scripts/manage_db.py rir --path data/rirs

# RIR 시뮬레이션 생성
uv run python scripts/generate_rir_bank.py --count 1000

# 시뮬레이션 결과 시각화 검증
uv run python scripts/visualize_rirs.py data/rirs/rir_00000.pkl
```

### 3. 모델 학습 (Training - In Progress)
`main.py`를 통해 학습을 실행합니다 (현재 Lightning Module 구현 진행 중).
```bash
# (예정/WIP) 
# uv run python main.py fit --config configs/baseline.yaml
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

📘 사외(External) Git 저장소 동기화 및 협업 가이드이 문서는 보안/네트워크 문제로 분리된 사외 Git 저장소의 코드를 사내 Git으로 안전하게 가져오고, 우리만의 기능을 확장하기 위한 워크플로우를 설명합니다.1. 핵심 전략 (Vendor Branch 패턴)우리는 사내 저장소에서 두 개의 핵심 브랜치를 운영합니다.브랜치 이름역할 및 설명권한 규칙external-base사외 Git의 원본 미러링.외부 코드가 업데이트되면 이 브랜치만 최신화됩니다.⛔ 직접 수정 금지 (Read-Only)main사내 협업용 메인 브랜치.external-base를 기반으로 우리만의 커스텀 기능이 추가됩니다.✅ 개발 및 배포 진행2. [관리자용] 초기 세팅 (최초 1회)담당자: 사외망과 사내망 모두에 접근 가능한 PC를 가진 관리자사내 저장소 클론 및 리모트 설정Bash# 1. 사내 저장소(origin) 클론
git clone [사내_Git_주소] my-project
cd my-project

# 2. 사외 저장소(upstream) 추가
git remote add upstream [사외_Git_주소]
external-base 브랜치 생성 및 푸시Bash# 사외 저장소 내용 가져오기
git fetch upstream

# 사외 메인 코드를 기반으로 브랜치 생성
git checkout -b external-base upstream/main

# 사내 저장소에 공유
git push origin external-base
3. [관리자용] 사외 코드 업데이트 절차 (정기 수행)목표: 사외 Git에 업데이트가 생겼을 때, external-base를 갱신하고 main에 병합합니다.Step 1. 원본 동기화 (external-base 갱신)가장 먼저 사외 코드를 그대로 가져와 사내 서버에 업데이트합니다. 이 과정에서는 충돌이 발생하지 않습니다.Bash# 1. 최신 내용 가져오기
git fetch upstream

# 2. external-base 브랜치로 이동
git checkout external-base

# 3. 사외 코드로 덮어쓰기 (Fast-forward)
git merge upstream/main

# 4. 사내 저장소에 원본 최신화 반영
git push origin external-base
Step 2. 우리 코드에 합치기 (main 갱신)이제 업데이트된 원본을 우리가 개발 중인 코드에 합칩니다.Bash# 1. 개발용 브랜치로 이동
git checkout main

# 2. 업데이트된 external-base를 병합
git merge external-base
⚠️ 주의: 이 단계에서 **Merge Conflict(충돌)**가 발생할 수 있습니다.충돌 발생 시: 우리가 수정한 파일과 외부 업데이트 파일이 겹친 것입니다.해결 방법: 사내 개발 담당자와 상의하여 충돌을 해결(add & commit)한 후 푸시합니다.Bash# 3. 최종 결과 사내 공유
git push origin main
4. [개발자용] 동료들을 위한 활용 팁일반 개발자는 복잡한 upstream 설정 없이 사내 저장소(origin)만 사용하면 됩니다.Q1. 순수 원본 코드와 우리가 수정한 코드를 비교하고 싶다면?external-base는 항상 순수한 외부 코드 상태입니다. 아래 명령어로 우리의 작업 내역만 발라내어 볼 수 있습니다.Bash# external-base(원본)와 main(우리꺼)의 차이점 비교
git diff external-base main
Q2. 개발은 어떻게 하나요?평소처럼 main 브랜치에서 Feature 브랜치를 따서 작업하시면 됩니다. external-base 브랜치는 참고용으로만 확인하세요.Bashgit checkout main
git checkout -b feature/my-new-function
5. ⚠️ 절대 지켜야 할 규칙 (Safety Rules)external-base 브랜치에서는 절대 작업하지 마세요.이 브랜치는 사외 Git과 100% 동일하게 유지되어야 합니다. 여기에 사내 코드가 섞이면 나중에 업데이트를 받을 때 족보가 꼬이게 됩니다.main 브랜치 업데이트는 신중하게.관리자가 external-base를 main에 병합할 때는, 기존 기능이 깨지지 않는지 반드시 테스트가 필요합니다.