# MLFlow 및 모니터링 가이드

본 문서는 학습 과정 중 모델의 성능과 시스템 상태를 모니터링하기 위해 추가된 MLFlow 및 PyTorch Lightning 기능에 대해 설명합니다.

---

## 🎨 주요 모니터링 기능

### 1. 스펙트로그램 시각화 (Spectrograms)
오디오 파일(`Noisy`, `Enhanced`, `Target`)과 함께 주파수 특성을 보여주는 스펙트로그램 이미지(`.png`)가 자동으로 로깅됩니다.
- **확인 방법**: MLFlow UI -> Run 클릭 -> `Artifacts` -> `audio_samples/sample_i` 폴더 확인
- **활용**: 고주파 잡음 제거 정도, 잔향 제거 특성(Dereverberation) 등을 시각적으로 즉시 파악할 수 있습니다.

### 2. 시스템 메트릭 모니터링 (System Metrics)
`DeviceStatsMonitor`를 통해 학습 중 하드웨어 상태를 실시간으로 기록합니다.
- **기록 항목**: GPU 사용률(Utilization), GPU 메모리 점유, CPU 부하, GPU 온도 등.
- **확인 방법**: MLFlow UI -> Run 클릭 -> `Metrics` 탭에서 `sys/` 접두사가 붙은 항목 선택.

### 3. 최적 모델 자동 추적 (ModelCheckpoint)
`val_loss`가 가장 낮은 "최고의 모델"을 자동으로 선별하여 체크포인트를 저장합니다.
- **파일명 형식**: `best-model-epoch={epoch}-val_loss={val_loss}.ckpt`
- **설정**: 1개의 최고 성능 모델만 유지하며, 새로운 최고 기록이 나오면 자동으로 갱신됩니다.

---

## ⚙️ 설정 방법 (YAML)

`configs/ic_conv_tasnet.yaml` 파일의 다음 항목들을 통해 로깅 동작을 제어할 수 있습니다.

```yaml
model:
  init_args:
    num_val_samples_to_log: 4  # 로깅할 샘플 개수 (기본값: 4)

trainer:
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        monitor: "val_loss"
        save_top_k: 1
    - class_path: lightning.pytorch.callbacks.DeviceStatsMonitor
      init_args:
        cpu_stats: true
```

---

## 💡 팁
- **수렴 확인**: `Metrics` 탭에서 `train_loss_step`과 `val_loss` 그래프를 겹쳐보며 과적합(Overfitting) 여부를 판단하세요.
- **무결성 검사**: 학습 시작 직후 `step0` 파일들이 올라옵니다. 이는 학습 전 상태이므로 이후 `step1000`, `step2000` 등과 비교하는 기준점으로 활용하세요.
