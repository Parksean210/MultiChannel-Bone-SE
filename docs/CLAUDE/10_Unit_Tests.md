# 10. 유닛 테스트 가이드

## 개요

이 프로젝트는 **pytest** 기반의 유닛 테스트를 제공합니다.
실제 DB나 오디오 파일 없이 합성 텐서만으로 **CPU에서 빠르게** 동작합니다 (전체 실행 약 4초).

### 통합 검증 vs 유닛 테스트

| 구분 | 파일 | 목적 | 실행 조건 |
|---|---|---|---|
| **유닛 테스트** | `tests/` | 개별 함수·클래스 동작 검증 | DB 없음, CPU, ~4초 |
| **통합 검증** | `scripts/verify_pipeline.py` | 실제 데이터로 End-to-End 파이프라인 확인 | DB 필요, GPU 권장 |

CI/CD에는 유닛 테스트를 사용하고, 데이터 파이프라인 변경 시 통합 검증도 함께 실행합니다.

---

## 테스트 파일 구조

```
tests/
├── __init__.py
├── conftest.py           # 공유 픽스처 (합성 텐서)
├── test_synthesis.py     # src/utils/synthesis.py 테스트
├── test_losses.py        # src/modules/losses.py 테스트
├── test_audio_io.py      # src/utils/audio_io.py 테스트
├── test_models.py        # src/models/base.py 테스트
└── test_metrics.py       # src/utils/metrics.py 테스트
```

---

## 빠른 실행

```bash
# 전체 테스트 실행 (권장)
uv run pytest

# 특정 파일만 실행
uv run pytest tests/test_synthesis.py

# 특정 클래스·함수만 실행
uv run pytest tests/test_losses.py::TestSISDRLoss
uv run pytest tests/test_losses.py::TestSISDRLoss::test_perfect_less_than_random

# 느린 테스트 제외 (STOI 등)
uv run pytest -m "not slow"

# 커버리지 포함
uv run pytest --cov=src --cov-report=term-missing
```

---

## 테스트 파일별 상세 설명

### `conftest.py` — 공유 픽스처

모든 테스트가 공통으로 사용하는 합성 데이터를 pytest fixture로 제공합니다.

| 픽스처 | shape | 설명 |
|---|---|---|
| `raw_speech` | `(B=2, T=16000)` | 합성 음성 신호 |
| `raw_noises` | `(B=2, S-1=7, T=16000)` | 합성 노이즈 신호 |
| `rir_tensor` | `(B=2, M=5, S=8, L=512)` | 델타 함수 RIR (t=0에서 1.0) |
| `snr_tensor` | `(B=2,)` | 목표 SNR [10.0, 5.0] dB |
| `bcm_kernel` | `(1, 1, 101)` | 500Hz LPF 커널 |
| `synthesis_batch` | `dict` | apply_spatial_synthesis용 완전한 배치 |
| `sinusoidal_speech` | `(1, 1, T)` | 440Hz 사인파 (메트릭 테스트용) |

> **델타 함수 RIR 사용 이유**: `t=0`에 1.0인 델타 함수는 컨볼루션에서 항등 연산 → 실제 DB 없이도 공간화 전후 값을 수학적으로 검증 가능

---

### `test_synthesis.py` — 합성 파이프라인

**대상**: `src/utils/synthesis.py`

| 클래스 | 테스트 항목 |
|---|---|
| `TestCreateBCMKernel` | shape `(1,1,101)`, 정규화(합=1.0), LPF 특성(2kHz↑ 에너지 < 저주파×10%), NaN 없음 |
| `TestSpatializeSources` | shape `(B,M,T)`, NaN 없음, 개별 노이즈 리스트, 개별합=전체 노이즈, 델타 RIR 항등성 |
| `TestApplyBCMModeling` | shape 유지, BCM 채널 LPF 효과 (2kHz+ 에너지 90% 감소), 비BCM 채널 변화없음 |
| `TestScaleNoiseToSNR` | SNR 정확도 (±0.5dB), shape 유지, float64 입력 처리, 높은 SNR→낮은 노이즈 에너지 |
| `TestGenerateAlignedDry` | shape `(B,1,T)`, NaN 없음, 델타 t=0 항등성, 딜레이 D → D샘플 정렬 |
| `TestApplySpatialSynthesis` | 출력 키 확인, shape, NaN 없음, BCM 비활성화, 개별 노이즈 반환 |

**주요 수학적 검증 예시**:
```python
# 델타 RIR 항등성: f * δ(t) = f(t)
rir[:, :, :, 0] = 1.0  # t=0 델타
speech_mc, _ = spatialize_sources(raw_speech, raw_noises, rir_tensor)
assert torch.allclose(speech_mc[:, 0, :], raw_speech, atol=1e-4)
```

---

### `test_losses.py` — 손실 함수

**대상**: `src/modules/losses.py`

| 클래스 | 테스트 항목 |
|---|---|
| `TestSISDRLoss` | 스칼라 출력, 완벽 예측 < 0, 완벽 < 무작위, shape 불일치 오류, 멀티채널, 역전파, 스케일 불변성 |
| `TestSTFTLoss` | 스칼라, 비음수, 동일 신호 < 다른 신호, 멀티채널, 역전파 |
| `TestMultiResolutionSTFTLoss` | 스칼라, 비음수, STFTLoss 개수 확인, 크기 불일치 오류, 역전파 |
| `TestCompositeLoss` | 스칼라, alpha=0 → SISDRLoss 동일, alpha 스케일링, 역전파 |

**핵심 검증 로직**:
```python
# SI-SDR 스케일 불변성 확인
loss1 = SISDRLoss()(preds, targets)
loss2 = SISDRLoss()(preds * 2.0, targets)  # 2배 스케일
assert torch.allclose(loss1, loss2, atol=1e-4)  # 동일해야 함

# alpha=0이면 CompositeLoss == SISDRLoss
composite = CompositeLoss(alpha=0.0)(preds, targets)
sisdr = SISDRLoss()(preds, targets)
assert torch.allclose(composite, sisdr, atol=1e-5)
```

---

### `test_audio_io.py` — 오디오 I/O

**대상**: `src/utils/audio_io.py`

| 클래스 | 테스트 항목 |
|---|---|
| `TestPrepareAudioForSaving` | shape (T,), 채널 선택, NaN→0, Inf 클리핑, ±1 클램핑, float32 dtype, detach |
| `TestBuildMetadataFilename` | 기본 패턴, prefix/suffix, step 번호, -1 패딩 제거, 텐서 ID, 음수 SNR, 문자열 반환 |
| `TestSaveAudioFile` | 파일 생성(텐서/numpy), 올바른 샘플레이트, 올바른 길이, 부모폴더 자동생성, 값 보존(±1e-3) |
| `TestCreateSpectrogramImage` | Figure 반환, PNG 파일 저장 |

> **WAV 정밀도 주의**: 기본 WAV는 16-bit PCM이므로 저장-읽기 후 오차 ≈ 1/32768 ≈ 3e-5.
> `atol=1e-3`으로 검증합니다.

---

### `test_models.py` — 모델 베이스 클래스

**대상**: `src/models/base.py`

| 클래스 | 테스트 항목 |
|---|---|
| `TestBaseSEModelInit` | `in_channels` 속성, window buffer 등록, window shape, Hann 특성, hamming/rect, 잘못된 타입 오류, forward NotImplementedError |
| `TestSTFT` | shape `(B,C,F,T)`, 복소수 출력, NaN 없음, 서로 다른 신호 다름 |
| `TestISTFT` | shape `(B,C,T)`, 라운드트립 정확도(atol=1e-4), length 파라미터 제어 |
| `TestToFrames` | 4D 출력, 배치/채널 유지, 마지막차원=win_length, NaN 없음, 윈도우 적용 확인 |

**STFT 라운드트립 테스트 (핵심)**:
```python
x = torch.randn(2, 5, 16000)
spec = model.stft(x)         # (2, 5, 257, T_frames) complex
reconstructed = model.istft(spec, length=16000)  # (2, 5, 16000)
assert torch.allclose(x, reconstructed, atol=1e-4)
# center=True + Hann window → COLA 조건 만족 → 거의 완벽한 복원
```

---

### `test_metrics.py` — 메트릭

**대상**: `src/utils/metrics.py`

| 클래스 | 테스트 항목 |
|---|---|
| `TestCreateMetricSuite` | 4개 키 존재, callable, dict 반환, 실제 텐서 처리 |
| `TestComputeMetrics` | dict 반환, float 값, 완벽 예측 SI-SDR > 30dB, 노이즈 추가 시 저하, 부분 메트릭, 1D/2D/3D 입력, 미존재 메트릭 무시, SI-SDR 범위 |

> **`@pytest.mark.slow`**: STOI 테스트는 계산이 비교적 느리므로 slow 마커로 분류됩니다.
> CI에서 빠른 실행이 필요하면 `pytest -m "not slow"`를 사용하세요.

---

## 새 테스트 추가 방법

### 1. 새 함수 테스트

```python
# tests/test_synthesis.py에 추가 예시
class TestMyNewFunction:
    def test_output_shape(self):
        """기대 shape 확인"""
        from src.utils.synthesis import my_new_function
        result = my_new_function(torch.randn(2, 5, 16000))
        assert result.shape == (2, 5, 16000)

    def test_no_nan(self):
        result = my_new_function(torch.randn(2, 5, 16000))
        assert not torch.isnan(result).any()
```

### 2. 느린 테스트 표시

```python
@pytest.mark.slow
def test_stoi_on_real_signal(self):
    """STOI는 처리 시간이 길어 CI에서 제외"""
    ...
```

### 3. 공유 픽스처 추가

`tests/conftest.py`에 새 픽스처를 추가하면 모든 테스트 파일에서 바로 사용 가능합니다:

```python
@pytest.fixture
def noisy_speech():
    """고정 SNR 5dB의 합성 노이즈 음성"""
    clean = torch.randn(2, 16000) * 0.1
    noise = torch.randn(2, 16000) * 0.056  # ~5dB SNR
    return clean + noise
```

---

## CI/CD 통합

GitHub Actions 또는 Jenkins에서 아래와 같이 통합합니다:

```yaml
# .github/workflows/test.yml 예시
- name: Run unit tests
  run: uv run pytest tests/ -v --tb=short --junitxml=test-results.xml

- name: Run with coverage
  run: uv run pytest tests/ --cov=src --cov-report=xml
```

커버리지 리포트 생성:
```bash
uv run pytest --cov=src --cov-report=html
# htmlcov/index.html 에서 시각적으로 확인
```

---

## 테스트 실행 결과 예시

```
======================== 111 passed in 3.66s ==============================

tests/test_audio_io.py         25 passed
tests/test_losses.py           21 passed
tests/test_metrics.py          16 passed
tests/test_models.py           17 passed
tests/test_synthesis.py        32 passed
```

---

## FAQ

**Q: GPU가 없어도 테스트를 실행할 수 있나요?**
A: 네. 모든 테스트는 CPU에서 동작합니다. GPU 의존 코드 (`cuda`, `device`) 없이 설계되었습니다.

**Q: DB 파일이 없어도 됩니까?**
A: 유닛 테스트는 DB 불필요. DB 검증이 필요하면 `scripts/verify_pipeline.py`를 사용하세요.

**Q: PESQ 테스트가 가끔 실패합니다.**
A: PESQ는 외부 코덱에 의존하며, 특정 신호(매우 짧거나 무음)에서 예외가 발생할 수 있습니다.
`compute_metrics`는 내부에서 예외를 잡아 fallback(1.0)을 반환하므로, 단순 에러는 아닙니다.

**Q: 기존 체크포인트가 테스트에 영향을 미치나요?**
A: 아닙니다. 유닛 테스트는 체크포인트를 전혀 사용하지 않습니다.

**Q: 테스트 파일에서 `from src.xxx import ...`가 실패합니다.**
A: `pyproject.toml`의 `pythonpath = ["."]` 설정으로 프로젝트 루트가 경로에 추가됩니다.
반드시 `uv run pytest`로 실행하세요. (`python -m pytest` 직접 실행 시 경로 문제 발생 가능)
