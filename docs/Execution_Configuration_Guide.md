## 🚀 1. 학습 및 추론 실행 (Execution)

본 프로젝트는 `LightningCLI`를 통해 모든 기능을 제어합니다. `main.py`는 매우 슬림하며, 대부분의 상세 설정은 `configs/` 폴더의 YAML 파일에서 관리됩니다.

### 모델 학습 (Training)
```bash
uv run main.py fit --config configs/ic_conv_tasnet.yaml
```

### 모델 추론 (Inference / Prediction)
학습된 체크포인트를 사용하여 오디오를 복원합니다.
```bash
PYTHONPATH=. uv run main.py predict \
    --config configs/ic_conv_tasnet.yaml \
    --ckpt_path mlruns/path/to/best_model.ckpt \
    --trainer.callbacks.init_args.output_dir "results/my_inference"
```

---

## 🛠️ 2. YAML 설정 파일 및 오버라이드 (Override)

### 주요 설정 섹션
*   **`data`**: `batch_size`, `num_workers`, `db_path` 등 데이터 관련 설정.
*   **`model`**: 모델 아키텍처 및 내부 손실 함수 설정.
*   **`trainer`**: `max_epochs`, `precision`, `callbacks` 등 학습 엔진 설정.

### 고급 학습 제어 (Advanced Training Control)
모델의 과적합을 방지하고 최적화 성능을 높이기 위한 기능이 기본 포함되어 있습니다.

*   **Early Stopping (조기 종료)**: `val_loss`가 10 에폭 동안 개선되지 않으면 학습을 자동 중단합니다.
*   **Adaptive LR (학습률 자동 조절)**: `val_loss`가 5 에폭 동안 정체되면 학습률을 0.5배로 감소시켜 더 미세한 최적화를 수행합니다.

```yaml
# configs/ic_conv_tasnet.yaml (예시)
trainer:
  callbacks:
    - class_path: lightning.pytorch.callbacks.EarlyStopping
      init_args:
        monitor: "val_loss"
        patience: 10
        mode: "min"
```

### 터미널에서 설정 덮어쓰기 (CLI Override)
YAML 파일을 수정하지 않고도 터미널 옵션으로 설정을 즉시 변경할 수 있습니다. 이는 실험 변수를 빠르게 바꿔가며 테스트할 때 매우 유용합니다.

```bash
# 특정 데이터 필터링 및 배치 사이즈 변경 예시
uv run main.py fit \
    --config configs/ic_conv_tasnet.yaml \
    --data.batch_size 16 \
    --data.snr_range [-5, 5] \
    --trainer.max_epochs 100
```

## 🥗 3. 오디오 예측 결과물 관리 (Audio Outputs)

추론(`predict`) 단계에서 생성되는 오디오는 `AudioPredictionWriter` 콜백에 의해 관리됩니다.

*   **저장 원칙**: 각 샘플은 `sid_X_nids_Y_Z_rid_W_snr_VdB_<타입>.wav` 형식으로 고유한 메타데이터를 포함하여 저장됩니다.
*   **유연한 경로 지정**: `--trainer.callbacks.init_args.output_dir` 옵션을 통해 저장 위치를 변경할 수 있으며, 지정하지 않을 경우 체크포인트 이름을 폴더명으로 하여 기본 경로에 저장됩니다.

---

## 📊 4. 학습 상태 확인 (Monitoring)

학습 중에는 **MLflow**를 통해 **SI-SDR, DNSMOS, PESQ, STOI** 등 다양한 음향 지표를 실시간으로 확인할 수 있습니다.

1. 서버 실행: `uv run mlflow ui`
2. 웹 브라우저: `http://localhost:5000` 접속

---
**최종 수정일**: 2024-02-19
