# Audio Tool (`scripts/audio_tool.py`) 사용 가이드

이 스크립트는 프로젝트 내 오디오 파일의 포맷 변환 및 리샘플링을 고속으로 처리하기 위한 통합 도구입니다.

## 주요 기능

1. **PCM to WAV**: `.pcm`(int16) 파일을 `.wav`로 변환합니다. (KsponSpeech 원본 처리용)
2. **WAV to NPY**: `.wav` 파일을 고속 로딩용 `.npy`(int16)로 변환하고 원본을 삭제합니다.
3. **Resample (Turbo)**: 시스템 `ffmpeg`를 사용하여 오디오를 특정 샘플 레이트로 초고속 리샘플링합니다.

## 공통 옵션

- `--workers [N]`: 사용할 프로세스 개수 (기본값: 시스템 전체 코어 수)

---

## 명령어 상세 가이드

### 1. PCM 파일을 WAV로 변환
KsponSpeech와 같이 `.pcm` 확장자로 된 파일을 `.wav`로 일괄 변환합니다.
```bash
# 기본값(16kHz)으로 변환
uv run python scripts/audio_tool.py pcm2wav data/raw/speech

# 특정 샘플 레이트 명시 및 원본(.pcm) 삭제하며 변환 (용량 절약)
uv run python scripts/audio_tool.py pcm2wav data/raw/speech --sr 16000 --delete
```

### 2. WAV 파일을 NPY로 변환 (학습 가속화)
학습 시 CPU 부하를 줄이기 위해 `.npy` 포맷으로 변환합니다. 변환 후 원본 `.wav`는 삭제됩니다.
```bash
uv run python scripts/audio_tool.py wav2npy data/raw/speech
```

### 3. 오디오 리샘플링 (Turbo Mode)
시스템 `ffmpeg`를 활용하여 폴더 내 모든 `.wav` 파일을 리샘플링합니다.
```bash
# data/raw/noise 폴더 내 모든 wav를 16kHz로 리샘플링 (원본 유지)
uv run python scripts/audio_tool.py resample data/raw/noise --sr 16000

# 리샘플링 후 원본 파일 즉시 교체 (용량 부족 시 권장)
uv run python scripts/audio_tool.py resample data/raw/noise --sr 16000 --delete
```


## 주의사항
- **FFmpeg 기반:** `resample` 기능을 사용하려면 시스템에 `ffmpeg`가 설치되어 있어야 합니다.
- **용량 관리:** `--delete` 옵션 사용 시 복구가 불가능하므로 주의하십시오.
