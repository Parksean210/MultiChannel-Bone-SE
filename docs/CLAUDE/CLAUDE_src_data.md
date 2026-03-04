# src/data/ - Data Pipeline

## Overview
SQLite 메타데이터 기반의 데이터 로딩 파이프라인. CPU에서 원본 오디오를 로드하고, GPU 합성은 SEModule에서 수행한다.

## Files
| File | Class | Description |
|---|---|---|
| `models.py` | `SpeechFile`, `NoiseFile`, `RIRFile` | SQLModel ORM 테이블 정의 |
| `dataset.py` | `SpatialMixingDataset` | PyTorch Dataset. DB 조회 + 오디오/RIR 로딩 |
| `datamodule.py` | `SEDataModule` | LightningDataModule. Train/Val/Test DataLoader 관리 |

## Data Flow
```
SpatialMixingDataset.__getitem__(idx)
  1. Speech 로드 (random crop to 48000 samples = 3s @16kHz)
  2. Data Augmentation (train only): Speed Perturbation (0.9~1.1x) + Gain (-6~+6dB)
  3. RIR (.pkl) 랜덤 선택 + 캐시 로드
  4. Noise 원본 N개 수집 (RIR 소스 수만큼)
  -> Dict: {raw_speech, raw_noises, rir_tensor, snr, mic_config, ...}

collate_gpu_synthesis()
  -> RIR 길이 패딩 (배치 내 최대 길이 기준)
  -> noise_ids 패딩 (max_sources=8, -1로 패딩)

SEModule._apply_gpu_synthesis(batch)  [GPU]
  -> FFT Convolution + BCM + SNR 스케일링
  -> {noisy, clean, aligned_dry} 생성
```

## ORM Models (models.py)
- `SpeechFile`: path, dataset_name, speaker, language, duration_sec, sample_rate, split
- `NoiseFile`: path, dataset_name, category, sub_category, duration_sec, sample_rate, split
- `RIRFile`: path, room_type, num_noise, num_mic, num_bcm, rt60, duration_sec, sample_rate, split

## Key Design Points
- `.npy` 파일은 `mmap_mode='r'`로 로드하여 I/O 최소화 (int16 저장, float32 변환)
- RIR 캐시: 최대 100개까지 메모리 보관 (FIFO 교체)
- DB 쿼리 결과를 초기화 시 전체 메모리 캐싱 (DB 커넥션 병목 제거)
- 데이터 정렬: `sort()` 호출로 ID 기반 결정론적 순서 보장 (시드 재현성)
- `worker_init_fn`: DataLoader 멀티프로세싱 시드 충돌 방지
- Data Augmentation: Speed Perturbation + Gain (train split에서만, 공간화 이전 적용)
  - 증강 후 `clamp(-1.0, 1.0)`으로 클리핑 방지
  - SNR 스케일링은 증강+공간화 후 RMS 기반이므로 SNR 정확도 영향 없음

## Predict Mode Filtering
`SEDataModule`에서 `speech_id`, `noise_ids`, `rir_id`, `fixed_snr` 파라미터로 특정 조합 필터링 가능.
