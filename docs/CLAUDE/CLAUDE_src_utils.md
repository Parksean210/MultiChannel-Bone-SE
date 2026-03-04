# src/utils/ - Shared Utilities

## Overview
SEModule, 스크립트, 콜백에서 공통으로 사용하는 재사용 가능한 유틸리티 패키지.
공간 합성, 오디오 I/O, 메트릭 계산, 모델 비교 기능을 제공합니다.

## Files
| File | Description |
|---|---|
| `synthesis.py` | GPU/CPU 통합 공간 합성 엔진 (FFT Conv, BCM, SNR 스케일링) |
| `audio_io.py` | 오디오 저장/변환/스펙트로그램 생성 유틸리티 |
| `metrics.py` | 메트릭 계산, 체크포인트 로드, 모델 비교 도구 |

## synthesis.py - 공간 합성

| Function | Input → Output | Description |
|---|---|---|
| `create_bcm_kernel(cutoff_hz, sample_rate, num_taps)` | params → `(1,1,N)` | Sinc+Hann LPF 커널. SEModule에서 `register_buffer`로 1회 캐싱 |
| `spatialize_sources(raw_speech, raw_noises, rir_tensor)` | `(B,T), (B,S-1,T), (B,M,S,L)` → `(B,M,T), (B,M,T)` | FFT Convolution으로 음원 공간화 |
| `apply_bcm_modeling(speech_mc, noise_mc, bcm_kernel, atten_db)` | `(B,M,T)` → `(B,M,T)` | 마지막 채널(BCM) LPF + 노이즈 감쇄 |
| `scale_noise_to_snr(speech_mc, noise_mc, snr)` | `(B,M,T), (B,)` → `(B,M,T)` | Channel 0 기준 RMS로 목표 SNR 달성 |
| `generate_aligned_dry(raw_speech, rir_tensor)` | `(B,T), (B,M,S,L)` → `(B,1,T)` | RIR peak 기반 시간 정렬 (벡터화, for 루프 없음) |
| `apply_spatial_synthesis(batch, bcm_kernel)` | Dict → Dict | 위 함수 조합 고수준 API. `noisy/clean/aligned_dry` 키 추가 |

### 합성 파이프라인 순서
1. `spatialize_sources` - FFT Conv으로 speech/noise 공간화
2. `apply_bcm_modeling` - BCM 채널 LPF + 노이즈 감쇄 (선택적)
3. `scale_noise_to_snr` - 목표 SNR로 노이즈 스케일링
4. Mix: `noisy = speech_mc + noise_mc`
5. `generate_aligned_dry` - Dereverberation 타겟 생성

## audio_io.py - 오디오 I/O

| Function | Description |
|---|---|
| `prepare_audio_for_saving(tensor, channel)` | NaN 제거, 클리핑(-1~1), CPU float32 변환 |
| `save_audio_file(path, audio, sample_rate)` | 단일 채널 WAV 저장 (디렉토리 자동 생성) |
| `build_metadata_filename(speech_id, noise_ids, rir_id, snr, ...)` | 표준 파일명 생성. 패턴: `sid_X_nids_Y_Z_rid_W_snr_NdB` |
| `create_spectrogram_image(signal, sample_rate, n_fft, title, save_path)` | matplotlib 스펙트로그램 PNG 생성 |

## metrics.py - 메트릭 & 모델 비교

| Function | Description |
|---|---|
| `create_metric_suite(sample_rate)` | torchmetrics 객체 세트 생성 `{si_sdr, sdr, stoi, pesq}`. DDP 분산 집계 지원 |
| `compute_and_log_metrics(module, metrics, est, target, prefix)` | 메트릭 계산 + Lightning `module.log()` 통합 |
| `compute_metrics(estimated, target, sample_rate)` | 학습 루프 밖 독립 메트릭 계산 → `{name: float}` |
| `load_model_from_checkpoint(ckpt_path, device)` | `on_save_checkpoint`로 저장된 `model_class_name` + `model_init_args`로 모델 자동 복원. config 파일 불필요. |
| `transfer_to_device(batch, device)` | 배치 내 텐서 재귀적 디바이스 이동 |
| `compare_models(checkpoint_paths, dataset, snrs, output_dir)` | 다중 체크포인트 비교: 추론 + 메트릭 + 오디오 저장 + 결과표 출력 |

## 사용 패턴

### SEModule에서 (학습 루프)
```python
# __init__: BCM 커널 캐싱
bcm_kernel = create_bcm_kernel(cutoff_hz=500.0, sample_rate=16000, num_taps=101)
self.register_buffer('bcm_kernel', bcm_kernel)

# training_step: 합성 위임
batch = apply_spatial_synthesis(batch, bcm_kernel=self.bcm_kernel)

# validation_step: 메트릭 로깅
compute_and_log_metrics(self, metric_suite, est, target, prefix="val")
```

### 스크립트에서 (독립 실행)
```python
# 독립 메트릭 계산
results = compute_metrics(estimated, target, sample_rate=16000)

# 체크포인트 로드 (config 없이 ckpt만으로 동작)
pl_module = load_model_from_checkpoint("results/.../best-model.ckpt", device="cuda")

# 모델 비교 (CLI는 scripts/compare_checkpoints.py 사용)
compare_models(
    checkpoint_paths=["ckpt_a.ckpt", "ckpt_b.ckpt"],
    dataset=dataset,        # SpatialMixingDataset(split="test") 인스턴스
    snrs=[-5, 0, 10, 20],
    num_samples=5,
    output_dir="results/comparison",
    device="cuda",
)
```

## load_model_from_checkpoint 동작 원리
```
1. torch.load(ckpt_path)로 체크포인트 로드
2. ckpt["model_class_name"] 읽기 (SEModule.on_save_checkpoint에서 저장)
3. getattr(src.models, class_name)으로 모델 클래스 획득
4. ckpt["model_init_args"]로 모델 인스턴스 생성 (init args 자동 복원)
5. SEModule.load_from_checkpoint(..., strict=False)로 최종 로드
```
- 전제 조건: 해당 체크포인트가 `on_save_checkpoint`가 있는 SEModule로 저장되었을 것

## 주의사항
- `create_bcm_kernel`은 1회만 호출하고 `register_buffer`로 캐싱해야 함 (매 스텝 재생성 금지)
- `compare_models`의 `dataset` 인자는 `SpatialMixingDataset(split="test")` 인스턴스 필요
- `save_audio_file(path, tensor, sr, channel=0)`: tensor는 최소 2D `(C, T)` 형태 필요 (1D 불가)
- 메트릭 계산 실패 시 `float('nan')` 반환 + warning 로깅 (짧은 오디오에서 PESQ 실패 가능)
- `compare_models`의 평균 계산은 NaN을 제외하는 `_nanmean` 사용
