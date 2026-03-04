# data/ - Data Storage

## Overview
음성, 노이즈, RIR 원본 데이터와 SQLite 메타데이터 데이터베이스를 저장하는 디렉토리.
학습 시에는 이 디렉토리의 데이터를 직접 읽어 GPU 합성을 수행한다.

## Structure
```
data/
  metadata.db              # SQLite 메타데이터 (SpeechFile, NoiseFile, RIRFile 테이블)
  speech/
    KsponSpeech/           # 한국어 음성 데이터 (KsponSpeech 코퍼스)
      KsponSpeech_01/      # .npy 또는 .wav 형식
      eval_clean/
      eval_other/
  noise/
    136-2.극한 소음 환경 소리 데이터/  # 극한 소음 데이터셋
      TS_*/                # Train Split (교통수단, 공사장, 공장, 시설류 등)
      VS_*/                # Validation Split
  rirs/                    # 시뮬레이션된 Room Impulse Responses (.pkl)
    viz/                   # RIR 시각화 이미지 (.png)
  samples/                 # 검증용 합성 오디오 샘플
  outputs/                 # 출력 샘플
    test_samples/
    val_samples/
```

## Data Formats
| Type | Format | Description |
|---|---|---|
| Speech | `.npy` (int16) | 모노 16kHz. mmap 로딩 최적화. 32768로 나누어 float 변환 |
| Speech | `.wav` | fallback 포맷. torchaudio로 로드 |
| Noise | `.npy` / `.wav` | Speech와 동일 |
| RIR | `.pkl` | pickle. {meta, source_info, rirs} 딕셔너리 |
| Metadata | `.db` (SQLite) | SQLModel ORM 기반. 3개 테이블 |

## Notes
- `.npy`가 `.wav`보다 5~10배 빠른 로딩 속도 (mmap 지원)
- `scripts/audio_tool.py wav2npy`로 일괄 변환 가능
- 노이즈 폴더명에서 카테고리 자동 파싱 (TS_01.교통수단_01.사람의비언어적소리 등)
- RIR은 `scripts/generate_rir_bank.py`로 사전 생성
- 이 디렉토리는 .gitignore에 포함되어 있으므로 별도 관리 필요
