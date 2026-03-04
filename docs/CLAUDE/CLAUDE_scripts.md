# scripts/ - CLI Utility Scripts

## Overview
데이터 전처리, DB 관리, RIR 생성, 검증용 샘플 생성 등 오프라인 유틸리티 스크립트 모음.
모든 스크립트는 `uv run python scripts/<name>.py`로 실행한다.

## Files
| File | Purpose |
|---|---|
| `manage_db.py` | SQLite 메타데이터 DB 통합 관리 CLI (인덱싱, 분할, 통계, 동기화) |
| `generate_rir_bank.py` | pyroomacoustics로 대량 RIR 병렬 생성 (멀티프로세싱) |
| `generate_samples.py` | 데이터 파이프라인 검증용 오디오 샘플 생성 (CPU 합성) |
| `audio_tool.py` | 오디오 포맷 변환 CLI (PCM->WAV, WAV->NPY, NPY->WAV, 리샘플링) |
| `final_indexing.py` | 최종 데이터 인덱싱 스크립트 |
| `compare_checkpoints.py` | 다중 체크포인트 추론 비교 CLI. SNR별 메트릭(SI-SDR/SDR/STOI/PESQ) + 오디오 저장 |
| `verify_ic_conv_tasnet.py` | IC-Conv-TasNet 모델 검증 |
| `verify_pipeline_improvements.py` | 파이프라인 개선 검증 |
| `visualize_rirs.py` | RIR 시각화 (방 구조 + 마이크/소스 위치 플롯) |

## Key Commands

### DB 관리 (manage_db.py)
```bash
# 음성 인덱싱
uv run python scripts/manage_db.py speech --path data/speech/KsponSpeech --dataset KsponSpeech

# 노이즈 인덱싱
uv run python scripts/manage_db.py noise --path data/noise/... --dataset "극한소음"

# RIR 인덱싱
uv run python scripts/manage_db.py rir --path data/rirs --dataset "SimRIR"

# Train:Val:Test 재배치
uv run python scripts/manage_db.py realloc --type speech --ratio 0.8 0.1 0.1

# DB 통계 확인
uv run python scripts/manage_db.py stats

# WAV->NPY 경로 동기화
uv run python scripts/manage_db.py sync
```

### RIR 생성 (generate_rir_bank.py)
```bash
uv run python scripts/generate_rir_bank.py --count 1000 --workers 8 --rir_len 1.0
```

### 모델 비교 (compare_checkpoints.py)
```bash
# 기본 사용법: 체크포인트 2개를 SNR -5, 0, 10, 20 dB에서 5샘플씩 비교
uv run python scripts/compare_checkpoints.py \
    --checkpoints results/.../best-conv-tasnet.ckpt results/.../best-mamba2.ckpt \
    --snrs -5 0 10 20 \
    --num_samples 5 \
    --output_dir results/comparison \
    --db_path data/metadata.db

# 단일 체크포인트 성능 확인
uv run python scripts/compare_checkpoints.py \
    --checkpoints results/.../best-model.ckpt \
    --snrs 0 10 \
    --num_samples 10
```

**주요 인자**:
| 인자 | 기본값 | 설명 |
|---|---|---|
| `--checkpoints` | (필수) | 체크포인트 경로 1개 이상 |
| `--snrs` | `-5 0 5 10 15 20` | 테스트할 SNR 목록 (dB) |
| `--num_samples` | `20` | 비교할 테스트 샘플 수 |
| `--output_dir` | `results/comparison` | 오디오/결과 저장 경로 |
| `--db_path` | `data/metadata.db` | 메타데이터 DB 경로 |
| `--device` | `cuda` / `cpu` | 추론 디바이스 |

**출력 구조**:
```
results/comparison/
  sample_0/snr_-5dB/input_noisy_*.wav
  sample_0/snr_-5dB/target_clean_*.wav
  sample_0/snr_-5dB/output_best-conv-tasnet_*.wav
  sample_0/snr_-5dB/output_best-mamba2_*.wav
  ...
```

### 오디오 변환 (audio_tool.py)
```bash
# WAV -> NPY (고속 학습용)
uv run python scripts/audio_tool.py wav2npy data/speech --delete

# PCM -> WAV
uv run python scripts/audio_tool.py pcm2wav data/raw --sr 16000

# 리샘플링 (FFmpeg 기반)
uv run python scripts/audio_tool.py resample data/noise --sr 16000
```

## tests/ Subfolder
| File | Purpose |
|---|---|
| `test_base_model.py` | BaseSEModel 단위 테스트 |
