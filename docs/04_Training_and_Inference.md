# 학습·추론·테스트 실행 가이드

> 모든 실행은 `main.py`를 통한 LightningCLI 방식으로 동작합니다.
> YAML 파일로 모든 설정을 관리하며, CLI에서 개별 값을 오버라이드할 수 있습니다.

---

## 실행 모드 개요

| 모드 | 명령 | 설명 |
|---|---|---|
| `fit` | 학습 | 전체 학습 루프 실행. 자동으로 validation 포함 |
| `validate` | 검증 | 체크포인트 로드 후 validation set 평가 |
| `test` | 테스트 | 체크포인트 로드 후 test set 평가 + 최종 메트릭 |
| `predict` | 추론 | 체크포인트 로드 후 오디오 파일 생성 |

---

## 1. 학습 (fit)

### 기본 실행

```bash
uv run python main.py fit --config configs/ic_conv_tasnet.yaml
```

### BCM ablation 학습

```bash
uv run python main.py fit --config configs/ic_conv_tasnet_bcm_off.yaml
```

### 빠른 디버깅

```bash
uv run python main.py fit --config configs/ic_conv_tasnet.yaml \
    --trainer.limit_train_batches=10 \
    --trainer.limit_val_batches=2 \
    --trainer.max_epochs=1
```

### 멀티 GPU (DDP)

```bash
uv run python main.py fit --config configs/ic_conv_tasnet.yaml \
    --trainer.devices=4 \
    --trainer.strategy=ddp_find_unused_parameters_true
```

---

## 2. 테스트 (test)

```bash
uv run python main.py test \
    --config configs/ic_conv_tasnet.yaml \
    --ckpt_path results/lightning_logs/version_0/checkpoints/best-model.ckpt
```

테스트 완료 후 MLflow에 최종 지표가 자동 기록됩니다 (`test_si_sdr_final` 등).

---

## 3. 추론 (predict)

```bash
uv run python main.py predict \
    --config configs/ic_conv_tasnet.yaml \
    --ckpt_path results/.../checkpoints/best-model.ckpt
```

추론 결과는 `AudioPredictionWriter` 콜백이 자동으로 WAV 파일로 저장합니다.

**저장 경로**: `results/predictions/<체크포인트명>/`

**파일명 규칙**: `sid_{speech_id}_nids_{noise_ids}_rid_{rir_id}_snr_{N}dB_{type}.wav`

- `type`: `noisy`, `enhanced`, `target` 세 종류가 각각 저장됩니다.

---

## 4. YAML 설정 파일 구조

`configs/ic_conv_tasnet.yaml`을 예시로 각 섹션을 설명합니다.

### data 섹션

```yaml
data:
  class_path: src.data.SEDataModule
  init_args:
    db_path: "data/metadata.db"   # SQLite DB 경로
    batch_size: 4                  # 배치 크기 (VRAM에 따라 조절)
    num_workers: 8                 # DataLoader 병렬 로딩 수
    target_sr: 16000               # 목표 샘플 레이트 (자동 리샘플링)
```

### model 섹션

```yaml
model:
  class_path: src.modules.se_module.SEModule
  init_args:
    target_type: "aligned_dry"    # "aligned_dry": 잔향 제거 / "spatialized": 잡음만 제거
    sample_rate: 16000
    num_val_samples_to_log: 4     # validation 시 MLflow에 저장할 오디오 샘플 수

    model:                        # 실제 신경망 모델
      class_path: src.models.ICConvTasNet
      init_args:
        in_channels: 5            # 5: BCM 포함 / 4: BCM 제외
        use_checkpoint: true      # Gradient Checkpointing (RTX 3080 등 소형 GPU 필수)

    loss:
      class_path: src.modules.losses.CompositeLoss
      init_args:
        alpha: 0.1                # Frequency Loss 가중치 (1-alpha: SI-SDR Loss)

    optimizer_config:
      lr: 1e-3
      weight_decay: 1e-5
```

### trainer 섹션

```yaml
trainer:
  max_epochs: 50
  accelerator: "gpu"
  devices: "auto"                 # 사용 가능한 GPU 자동 감지
  strategy: "ddp_find_unused_parameters_true"  # 단일 GPU: "auto"
  precision: "16-mixed"           # 메모리 절약. A100: "bf16-mixed" 권장
  log_every_n_steps: 10
  val_check_interval: 7857        # N 스텝마다 검증 (7857 ≈ 1 epoch)
```

---

## 5. CLI 오버라이드

YAML 파일을 수정하지 않고 터미널에서 특정 값만 변경합니다.

```bash
# 학습률 변경
uv run python main.py fit --config configs/ic_conv_tasnet.yaml \
    --model.init_args.optimizer_config.lr=5e-4

# 배치 크기 변경
uv run python main.py fit --config configs/ic_conv_tasnet.yaml \
    --data.init_args.batch_size=8

# run 이름 변경 (MLflow에 표시됨)
uv run python main.py fit --config configs/ic_conv_tasnet.yaml \
    --trainer.logger.init_args.run_name="My_Experiment_v2"

# Epoch 수 제한 (빠른 실험)
uv run python main.py fit --config configs/ic_conv_tasnet.yaml \
    --trainer.max_epochs=5
```

---

## 6. 체크포인트 관리

### 자동 저장 규칙

`ModelCheckpoint` 콜백 설정 (`configs/ic_conv_tasnet.yaml`):

```yaml
- class_path: lightning.pytorch.callbacks.ModelCheckpoint
  init_args:
    monitor: "val_loss"       # 모니터링 지표
    mode: "min"               # val_loss가 낮을수록 좋음
    save_top_k: 1             # 가장 좋은 1개만 보존
    filename: "best-model-{epoch:02d}-{val_loss:.2f}"
```

### 체크포인트 위치

```
results/
└── lightning_logs/
    └── version_0/
        └── checkpoints/
            └── best-model-epoch=12-val_loss=5.32.ckpt
```

### 학습 재개

```bash
uv run python main.py fit \
    --config configs/ic_conv_tasnet.yaml \
    --ckpt_path results/.../checkpoints/best-model.ckpt
```

---

## 7. 학습 자동 제어

### Early Stopping

`val_loss`가 10 에폭 동안 개선되지 않으면 학습을 자동 중단합니다.

```yaml
- class_path: lightning.pytorch.callbacks.EarlyStopping
  init_args:
    monitor: "val_loss"
    patience: 10
    mode: "min"
    verbose: True
```

### Adaptive Learning Rate (ReduceLROnPlateau)

`val_loss`가 5 에폭 정체되면 학습률을 0.5배로 감소시킵니다. `SEModule.configure_optimizers`에서 자동으로 설정됩니다.

```python
# SEModule.configure_optimizers (코드 수정 불필요)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
```

---

## 8. 슈퍼컴퓨터 환경

A100 / H100 등 대형 GPU 환경에서는 다음 설정을 권장합니다.

```yaml
# configs/ic_conv_tasnet.yaml 수정 사항
trainer:
  precision: "bf16-mixed"       # A100은 BFloat16이 더 안정적
  strategy: "ddp"               # find_unused_parameters 불필요
  devices: 4                    # GPU 수에 맞게 고정

model:
  init_args:
    model:
      init_args:
        use_checkpoint: false   # 충분한 VRAM → Gradient Checkpoint 불필요
    batch_size: 16              # VRAM이 크면 배치 크기 증가
```

슈퍼컴퓨터 초기 설정은 `setup_supercomputer.sh`를 참고하세요.

---

## 9. 진행 상황 모니터링

### 터미널 Progress Bar

```
Epoch 5:  62%|██████    | 4872/7857 train_loss_step=6.32, val_loss=7.21, val_si_sdr=-2.1
```

### 실시간 지표 (MLflow)

```bash
# MLflow UI 실행
bash mlflow_server.sh
```

`http://localhost:6006`에서 에폭별 `val_loss`, `val_si_sdr`, `val_pesq` 등을 실시간 확인할 수 있습니다.
