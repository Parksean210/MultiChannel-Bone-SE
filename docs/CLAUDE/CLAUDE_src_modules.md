# src/modules/ - Training System (Lightning)

## Overview
PyTorch Lightning 기반 학습/검증/추론 루프와 손실함수를 관리하는 패키지.

## Files
| File | Class | Description |
|---|---|---|
| `se_module.py` | `SEModule` | LightningModule. 학습/검증/테스트/추론 루프, 지표 로깅 |
| `losses.py` | (여러 클래스) | 시간/주파수/멜 도메인 손실함수 전체 모음 |

## SEModule 핵심 동작
1. **`_apply_gpu_synthesis(batch)`**: `src.utils.synthesis.apply_spatial_synthesis`에 위임
   - BCM 커널은 `__init__`에서 `register_buffer`로 1회 캐싱 (매 스텝 재생성 없음)

2. **`forward(x)`**: 모델 입력 채널 자동 슬라이싱
   - `model.in_channels` (BaseSEModel 표준 인터페이스) 기준으로 자동 대응

3. **지표 로깅**: `src.utils.metrics.compute_and_log_metrics` 사용
   - SI-SDR, SDR, STOI, PESQ (모두 Channel 0 기준)

4. **오디오 샘플 로깅**: `src.utils.audio_io` 유틸리티 사용
   - MLflow: WAV + 스펙트로그램 PNG 아티팩트 저장
   - TensorBoard: `add_audio` 직접 호출

5. **`on_save_checkpoint(checkpoint)`**: 체크포인트에 모델 클래스명 저장
   - `checkpoint["model_class_name"] = type(self.model).__name__`
   - `load_model_from_checkpoint`가 이 값을 읽어 config 없이 모델 자동 복원

## Loss Functions
| Class | Type | Description |
|---|---|---|
| `SISDRLoss` | Time-domain | Negative SI-SDR (최소화 목적) |
| `SDRLoss` | Time-domain | Negative SDR (스케일 고정, alpha projection 없음) |
| `WaveformLoss` | Time-domain | L1 또는 L2 파형 거리 (`loss_type='l1'/'l2'`) |
| `STFTLoss` | Freq-domain | Spectral Convergence + Log Magnitude. `register_buffer`로 window/f_weight 캐싱. `use_f_weight=True`로 고주파 가중치 활성화 |
| `MultiResolutionSTFTLoss` | Freq-domain | 다해상도(512/1024/2048) STFT 손실 평균 |
| `ComplexSTFTLoss` | Freq-domain | L1(real) + L1(imag) + L1(log mag). 위상 정보 직접 학습 |
| `MelSpectrogramLoss` | Mel-domain | 멜 스펙트로그램 L1. 멜 필터뱅크 `register_buffer` 캐싱 |
| `MultiScaleMelLoss` | Mel-domain | 다해상도(512/1024/2048) 멜 스펙트로그램 손실 평균. Encodec/DAC 방식 |
| `AWeightedSTFTLoss` | Perceptual | IEC 61672 A-가중치 STFT 손실. 1~4 kHz(말소리 핵심 대역) 강조. A-가중치 커브 `register_buffer` 캐싱 |
| `CompositeLoss` | Hybrid | SI-SDR + alpha * MR-STFT (기본 alpha=0.1) |

## Adding a New Loss
1. `nn.Module` 상속, `forward(preds, targets) -> scalar` 구현
2. YAML에서 `class_path: src.modules.losses.YourLoss`로 지정

## Optimizer & Scheduler
- **AdamW** + **Linear Warmup** (5 epoch) + **Cosine Annealing** (SequentialLR)
- `optimizer_config` 딕셔너리로 lr/weight_decay 제어
- `scheduler_config` 딕셔너리로 warmup_epochs/min_lr 제어
- Gradient Clipping: `gradient_clip_val: 5.0` (YAML trainer 섹션)

## DDP 호환성
- 오디오 로깅은 `self.trainer.is_global_zero` 조건으로 rank 0에서만 수행
- 메트릭은 torchmetrics 분산 집계 사용 (sync_dist=True)
