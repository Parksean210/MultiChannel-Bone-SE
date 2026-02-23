# MLflow 실험 추적 가이드

> 학습 결과를 체계적으로 기록하고 모델 간 성능을 비교하는 방법을 설명합니다.
> 별도 설정 없이 학습을 시작하면 MLflow에 자동으로 기록됩니다.

---

## 서버 실행

### 백그라운드로 실행 (권장)

프로젝트 루트의 `mlflow_server.sh` 스크립트를 사용합니다. 파일 백엔드와 상대경로 artifact 저장소를 자동으로 설정합니다.

```bash
bash mlflow_server.sh
```

WSL2 환경에서는 `localhost` 대신 WSL IP로 접속합니다.

```bash
# WSL IP 확인
hostname -I | awk '{print $1}'
# 브라우저: http://<WSL_IP>:6006
```

### 서버 종료

```bash
pkill -f "mlflow ui"
```

### 포트 충돌 시

```bash
# 점유 프로세스 확인
ss -tlnp | grep 6006
# 다른 포트로 변경 시 mlflow_server.sh 내 --port 값 수정
```

### 백엔드: 파일

모든 실험 메타데이터(메트릭, 파라미터, 태그)와 아티팩트(WAV, PNG, YAML)가 `results/mlruns/`에 저장됩니다.

```
results/
└── mlruns/         ← 메트릭/파라미터/태그/아티팩트 파일 (WAV, PNG, YAML)
```

---

## 실험 구조

학습 run은 세 가지 Experiment(폴더)로 분리 관리됩니다.

| Experiment | 목적 | 사용 Config |
|---|---|---|
| `Architecture` | 모델 아키텍처 비교 (IC-ConvTasNet vs IC-Mamba 등) | `ic_conv_tasnet.yaml`, `baseline.yaml` |
| `BCM-Ablation` | BCM 채널 유무에 따른 성능 차이 검증 | `ic_conv_tasnet_bcm_off.yaml` |
| `Hyperparameter` | lr, loss 가중치 등 하이퍼파라미터 탐색 | 별도 config 작성 |

Experiment는 YAML의 `experiment_name` 필드로 지정됩니다.

```yaml
trainer:
  logger:
    class_path: lightning.pytorch.loggers.MLFlowLogger
    init_args:
      experiment_name: "Architecture"      # ← Experiment 지정
      run_name: "IC-Mamba-5ch-causal"      # ← Run 이름
      tracking_uri: "file:./results/mlruns"
```

---

## 자동 태깅 (MLflowAutoTagCallback)

모든 config에 포함된 `MLflowAutoTagCallback`이 학습 시작 시 자동으로 태그와 아티팩트를 기록합니다.

### 자동 기록되는 Tags

| Tag | 예시 값 | 설명 |
|---|---|---|
| `git_commit` | `9a82454` | 현재 커밋 해시 (재현성 보장) |
| `git_dirty` | `true` | 미커밋 변경사항 존재 여부 |
| `model_type` | `ICConvTasNet` | 모델 클래스명 |
| `in_channels` | `5` | 입력 채널 수 (5: BCM 포함, 4: 제외) |
| `target_type` | `aligned_dry` | 학습 타겟 종류 |
| `sample_rate` | `16000` | 샘플 레이트 |

### 자동 저장되는 Artifacts

- `config/<config_filename>.yaml` — 실행에 사용된 YAML 설정 파일

MLflow UI → Run → Artifacts → `config/`에서 확인할 수 있습니다.

---

## 자동 기록되는 지표

### 학습 중 (매 에폭)

| 지표 | 설명 |
|---|---|
| `train_loss` | 학습 손실 (step + epoch 평균) |
| `val_loss` | 검증 손실 |
| `val_si_sdr` | SI-SDR (dB). 핵심 음성 향상 지표 |
| `val_sdr` | SDR (dB) |
| `val_stoi` | 말소리 명료도 (0~1) |
| `val_pesq` | 인지 음질 (1~4.5) |

### 테스트 완료 후 (1회)

| 지표 | 설명 |
|---|---|
| `test_si_sdr_final` | 테스트셋 최종 SI-SDR |
| `test_sdr_final` | 테스트셋 최종 SDR |
| `test_stoi_final` | 테스트셋 최종 STOI |
| `test_pesq_final` | 테스트셋 최종 PESQ |

---

## 오디오 아티팩트

Validation 첫 번째 배치의 `num_val_samples_to_log`개 샘플에 대해 자동으로 저장됩니다.

**저장 위치**: `Artifacts → audio_samples/sample_N/`

| 파일 | 내용 |
|---|---|
| `Noisy_*.wav` | 모델 입력 (소음 혼합) |
| `Enhanced_*.wav` | 모델 출력 (향상된 음성) |
| `Target_*.wav` | 정답 신호 |
| `Noisy_spec_*.png` | Noisy 스펙트로그램 |
| `Enhanced_spec_*.png` | Enhanced 스펙트로그램 |
| `Target_spec_*.png` | Target 스펙트로그램 |

스펙트로그램을 통해 잡음 제거 효과를 시각적으로 확인할 수 있습니다.

---

## UI 사용법

### 실험 간 비교

1. 왼쪽 사이드바에서 Experiment 선택 (예: `Architecture`)
2. 비교할 Run들의 체크박스 선택
3. `Compare` 버튼 클릭

**Parallel Coordinates Plot**: 여러 하이퍼파라미터와 성능 지표의 상관관계를 시각화합니다. `lr`, `alpha` 등 값이 다른 run들 중 `val_si_sdr`이 높은 run들이 공통으로 어떤 설정을 갖는지 직관적으로 파악할 수 있습니다.

### 학습 곡선 해석

| 패턴 | 진단 | 조치 |
|---|---|---|
| `train_loss` ↓ `val_loss` ↓ | 정상 수렴 | Early Stopping 대기 |
| `train_loss` ↓ `val_loss` ↑ | 과적합 | `weight_decay` 증가, 데이터 추가 |
| 두 loss 모두 평행 | 학습 불가 | `lr` 10배 증가 시도 |
| loss 급등 | 발산 | `lr` 10분의 1로 감소 |

### 음성 향상 지표 해석

| 지표 | 좋은 범위 | 나쁜 신호 |
|---|---|---|
| `val_si_sdr` | 10dB 이상 | 음수이면 소음보다 못한 것 |
| `val_pesq` | 3.0 이상 | 1.0~1.5이면 심하게 왜곡 |
| `val_stoi` | 0.8 이상 | 0.5 이하이면 알아듣기 어려움 |

---

## 복수 체크포인트 비교

Python API로 여러 체크포인트를 직접 비교합니다.

```bash
uv run python scripts/compare_checkpoints.py \
    --checkpoints \
        results/.../best-model-5ch.ckpt \
        results/.../best-model-4ch.ckpt \
    --num_samples 100 \
    --output_dir results/comparison
```

결과로 CSV 파일과 각 샘플의 오디오가 저장됩니다.

---

## 스토리지 관리

### 오래된 run 삭제 (UI에서)

1. 삭제할 run 체크박스 선택
2. 상단 `Delete` 클릭
3. 영구 삭제: `uv run mlflow gc`

### 대량 삭제 (CLI)

```bash
# 초기화 (실험 데이터 전체 삭제, 복구 불가!)
rm -rf results/mlruns/
# 이후 bash mlflow_server.sh 로 서버 재시작하면 자동 재생성됨
```
