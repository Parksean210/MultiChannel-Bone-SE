# 오디오 파일 변환 도구 가이드

> `scripts/audio_tool.py`는 오디오 파일 포맷 변환과 리샘플링을 멀티프로세싱으로 고속 처리하는 전처리 도구입니다.
> 학습 전 데이터를 최적 포맷으로 정리할 때 사용합니다.

---

## 기능 요약

| 명령 | 기능 |
|---|---|
| `pcm2wav` | `.pcm` → `.wav` 변환 (KsponSpeech 원본 처리용) |
| `wav2npy` | `.wav` → `.npy` 변환 (학습 I/O 가속화) |
| `npy2wav` | `.npy` → `.wav` 복원 (변환 결과 청취 확인용) |
| `resample` | `.wav` 샘플 레이트 변경 (ffmpeg 기반 고속) |

**공통 옵션**: `--workers N` — 병렬 프로세스 수 (기본값: 시스템 전체 코어 수)

---

## 1. PCM → WAV 변환

KsponSpeech 등 `.pcm`(int16) 파일을 `.wav`로 일괄 변환합니다.

```bash
# 기본 변환 (16kHz)
uv run python scripts/audio_tool.py pcm2wav data/speech/KsponSpeech

# 변환 후 원본 .pcm 삭제 (디스크 절약)
uv run python scripts/audio_tool.py pcm2wav data/speech/KsponSpeech --sr 16000 --delete
```

---

## 2. WAV → NPY 변환 (학습 가속화 권장)

`.wav`보다 `.npy`(NumPy 배열)가 DataLoader에서 수십 배 빠르게 로딩됩니다. 대용량 데이터셋은 변환해두는 것을 권장합니다.

```bash
# 원본 유지하며 변환
uv run python scripts/audio_tool.py wav2npy data/speech/KsponSpeech

# 변환 후 원본 .wav 삭제 (권장 — DB 경로도 .npy로 자동 업데이트됨)
uv run python scripts/audio_tool.py wav2npy data/speech/KsponSpeech --delete
```

> **DB 경로 업데이트**: 변환 후 `manage_db.py sync`를 실행하면 DB의 파일 경로가 `.wav` → `.npy`로 자동 갱신됩니다.

---

## 3. NPY → WAV 복원 (검증용)

변환이 정상적으로 됐는지 청취로 확인하고 싶을 때 사용합니다.

```bash
uv run python scripts/audio_tool.py npy2wav data/speech/KsponSpeech --sr 16000

# 확인 후 .npy 삭제 (원본 .wav만 유지)
uv run python scripts/audio_tool.py npy2wav data/speech/KsponSpeech --sr 16000 --delete
```

---

## 4. 리샘플링 (ffmpeg 기반)

샘플 레이트가 다른 파일을 16kHz로 통일합니다. `torch`나 GPU 없이 시스템 `ffmpeg`만으로 동작합니다.

```bash
# 소음 데이터 전체를 16kHz로 리샘플링 (원본 유지)
uv run python scripts/audio_tool.py resample data/noise --sr 16000

# 리샘플링 후 원본 교체 (디스크 용량 부족 시)
uv run python scripts/audio_tool.py resample data/noise --sr 16000 --delete
```

> **ffmpeg 설치 필요**: `resample` 명령은 시스템에 `ffmpeg`가 설치되어 있어야 합니다.
> ```bash
> # Ubuntu
> sudo apt install ffmpeg
> ```

---

## 권장 데이터 준비 순서

```bash
# 1. KsponSpeech PCM 파일 변환
uv run python scripts/audio_tool.py pcm2wav data/speech/KsponSpeech --delete

# 2. 샘플 레이트 통일 (다른 SR인 경우)
uv run python scripts/audio_tool.py resample data/speech/KsponSpeech --sr 16000 --delete

# 3. NPY로 변환하여 I/O 가속화
uv run python scripts/audio_tool.py wav2npy data/speech/KsponSpeech --delete

# 4. DB 경로 동기화
uv run python scripts/manage_db.py sync
```

---

## 주의사항

- `--delete` 옵션은 원본 파일을 **즉시 영구 삭제**합니다. 복구가 불가능하므로 중요 파일은 별도 백업 후 사용하세요.
- `torch` 의존성 없이 `soundfile`과 `numpy`만으로 동작하므로 별도 환경에서도 실행 가능합니다.
