# src/callbacks/ - Lightning Callbacks

## Overview
학습/추론 과정에서의 이벤트 처리를 담당하는 Lightning 콜백 모음.

## Files
| File | Class | Description |
|---|---|---|
| `gpu_stats_monitor.py` | `GPUStatsMonitor` | pynvml 기반 GPU 사용률/메모리/온도 실시간 모니터링 |
| `audio_prediction_writer.py` | `AudioPredictionWriter` | 추론 결과를 WAV 파일로 자동 저장 |
| `mlflow_auto_tag.py` | `MLflowAutoTagCallback` | MLflow run에 git/모델 메타데이터 자동 태깅 + config YAML 아티팩트 저장 |

## GPUStatsMonitor
- `on_train_batch_end`마다 GPU utilization, memory usage, temperature 로깅
- MLflow System Metrics 탭과 호환되는 키 형식 사용 (`system/gpu_0_*`)
- SLURM 등 슈퍼컴퓨터 환경 대응: `CUDA_VISIBLE_DEVICES` 기반 물리 GPU 인덱스 매핑

## AudioPredictionWriter
- `predict` 모드에서 각 배치 추론 완료 시 자동 호출
- 저장 경로: `results/predictions/<checkpoint_name>/`
- 파일명 규칙: `sid_{speech_id}_nids_{noise_ids}_rid_{rir_id}_snr_{dB}dB_{type}.wav`
- Channel 0 (참조 마이크) 기준으로 noisy/enhanced/target 3종 저장
- NaN/Clipping 방어 처리 포함 (`src/utils/audio_io.py` 유틸리티 사용)

## MLflowAutoTagCallback
- `fit` 단계 시작 시 MLflow run에 메타데이터 자동 태깅
- Tags: `git_commit`, `git_dirty`, `model_type`, `in_channels`, `target_type`, `sample_rate`
- sys.argv에서 `--config`로 지정된 YAML 파일을 run artifacts의 `config/` 경로에 저장
- 모든 config에 추가하여 사용: `class_path: src.callbacks.mlflow_auto_tag.MLflowAutoTagCallback`
