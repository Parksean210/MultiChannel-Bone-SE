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

### 📂 폴더 구조 (Directory Structure)

```text
/
├── configs/            # ⚙️ 실험 설정 (YAML)
├── data/               # 💾 데이터 저장소
│   ├── raw/            # 원본 데이터 (Speech, Noise)
│   ├── rirs/           # 시뮬레이션된 Room Impulse Responses (.pkl)
│   └── metadata.db     # SQLite 데이터베이스 (인덱싱된 메타데이터)
│
├── src/                # 💻 소스 코드
│   ├── models/         # [Pure PyTorch] 모델 아키텍처 (base.py, ic_conv_tasnet.py 등)
│   ├── modules/        # [Lightning] 학습 루프 및 시스템 (se_module.py, losses.py)
│   ├── data/           # [Lightning] 데이터 파이프라인 (dataset.py, datamodule.py)
│   ├── callbacks/      # 이벤트 처리 (audio_prediction_writer.py, gpu_stats_monitor.py)
│   ├── db/             # DB 관리 (manager.py, engine.py)
│   └── simulation/     # 가상 음향 시뮬레이션 (generator.py, config.py)
│
├── results/            # 🎧 실험 결과물
│   └── predictions/    # 모델별/샘플별 추론 결과 오디오
├── mlruns/             # � MLflow 실험 데이터
├── scripts/            # 📜 유틸리티 스크립트 (manage_db.py, generate_samples.py 등)
│
├── setup_supercomputer.sh # 🚀 슈퍼컴퓨터(사내망) 환경 설정 스크립트
├── docs/               # 📚 상세 문서 (가이드라인)
│   ├── Database_Management_Guide.md  # DB 상세 관리 및 SQLModel 사용법
│   ├── RIR_Simulation_Guide.md      # RIR 생성 및 메타데이터 구조 가이드
│   ├── Data_Synthesis_Guide.md      # 온더플라이 데이터 합성 가이드
│   ├── Data_Pipeline_Deep_Dive.md   # 데이터 흐름 및 텐서 차원 심층 분석
│   ├── Execution_Configuration_Guide.md # LightningCLI 실행 및 YAML 설정 가이드
│   ├── MLflow_Guide.md              # MLflow 실험 추적 및 지표 분석 가이드
│   ├── Base_Model_Architecture_Guide.md # 모델 아키텍처 설계 및 구현 가이드
│   └── Git_Sync_Guide.md            # 📘 사외/사내 망 Git 동기화 가이드
├── main.py             # 🚀 실행 엔트리포인트 (LightningCLI)
└── pyproject.toml      # 📦 의존성 명세서 (uv)
```

---

## 🚀 워크플로우 (Research Workflow)

### 1. 환경 설정
`uv`를 사용하여 모든 의존성을 동기화합니다. 사내망 환경에서는 전용 스크립트를 활용합니다.
```bash
uv sync
# 또는
source setup_supercomputer.sh
```

### 2. 데이터 준비 (Preprocessing)
```bash
# DB 등록 및 데이터 분할
uv run python3 scripts/manage_db.py speech --path data/raw/speech/KsponSpeech --dataset KsponSpeech
uv run python3 scripts/manage_db.py noise --path data/raw/noise/traffic --dataset "TrafficNoise"
uv run python3 scripts/manage_db.py rir --path data/rirs --dataset "SimRIR_v1"
uv run python3 scripts/manage_db.py realloc --type speech
```

### 3. 모델 학습 (Training)
```bash
# 모든 평가지표(DNSMOS 등)가 실시간으로 로깅됨
PYTHONPATH=. uv run main.py fit --config configs/ic_conv_tasnet.yaml
```

### 4. 추론 및 결과 확인 (Inference)
```bash
# 특정 조건(ID, SNR)을 필터링하여 추론 실행
PYTHONPATH=. uv run main.py predict \
  --config configs/ic_conv_tasnet.yaml \
  --ckpt_path path/to/model.ckpt \
  --data.speech_id 3 --data.noise_ids [8,16] --data.fixed_snr 5
```
*   **저장 경로**: `results/predictions/<모델명>/sid_X_nids_Y_Z...wav`

### 6. 본전도(BCM) 유무 비교 실험
별도의 코드 수정 없이 설정 파일의 채널 수만 조절하여 실험을 수행합니다.
```bash
# 1. Bone 포함 (기존 5채널)
uv run python main.py fit --config configs/ic_conv_tasnet.yaml --trainer.logger.init_args.run_name "With_Bone"

# 2. Bone 제외 (4채널)
# --model.model.init_args.in_channels 4만 추가하면 자동으로 BCM 채널이 제외됨
uv run python main.py fit --config configs/ic_conv_tasnet.yaml --trainer.logger.init_args.run_name "No_Bone" --model.model.init_args.in_channels 4
```
*   **재현성**: `seed_everything: 42`와 데이터셋의 ID 정렬 로직이 결합되어, 두 실험은 완벽하게 동일한 데이터 조합과 순서로 진행됩니다.

### 5. 실험 분석 (Tracking)
```bash
# MLflow UI 백그라운드 실행
nohup uv run mlflow ui --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &
```

---

## 📊 평가지표 (Evaluation Metrics)

| 지표 | 설명 |
| :--- | :--- |
| **SI-SDR** | 음원 분리 및 향상 성능의 핵심 척도 |
| **PESQ** | 사람의 귀로 느끼는 인지적 음질 점수 (WB) |
| **STOI** | 음성의 말소리가 얼마나 잘 들리는지 나타내는 명료도 |

---

## 📅 최신 변경 사항 (Recent Updates)

- **[2026-02-20]**:
    - **BCM 비교 실험 지원**: `SEModule`의 `forward` 메서드 수정(자동 슬라이싱)을 통해 설정 변경만으로 BCM 유무 비교가 가능하도록 개선.
    - **재현성(Determinism) 보장**: DB 조회 데이터의 ID 기반 정렬을 통해 시드 고정 시 실험 간 완벽한 공정성(데이터 일치) 확보.
    - **출력 경로 구조 최적화**: `default_root_dir`을 활용하여 체크포인트가 MLflow의 Run ID 폴더 내부에 자동으로 깔끔하게 저장되도록 개선 (로그-모델 통합 관리).
- **[2026-02-19]**:
    - `main.py` 슬림화 및 YAML 중심 설정 리팩토링 (`LightningCLI` 완전 전환).
    - **PESQ/STOI** 실시간 로깅 및 텐서 파싱 로직 완전 통합.
    - **학습 고도화**: `EarlyStopping`(조기 종료) 및 `Adaptive LR`(Plateau 기반 학습률 자동 조절) 도입.
    - 가변 길이 노이즈 ID 배치 처리 안정화 (Padded Tensor 방식).
    - 검증 및 추론 결과 자동 폴더링 및 상세 메타데이터 파일명 규칙 도입.