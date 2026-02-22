# 데이터 합성 파이프라인 가이드

> 디스크에서 로드한 원본 파일들을 GPU에서 실시간으로 합성하여 학습 데이터를 만드는 과정을 설명합니다.
> 매 에폭마다 다른 조합이 생성되므로 별도의 오프라인 Data Augmentation 없이도 풍부한 학습 데이터가 확보됩니다.

---

## 설계 철학: CPU / GPU 역할 분리

```
[Disk]
  │  (CPU — Dataset.__getitem__)
  │  파일 로드 + 랜덤 크롭 + 패딩
  ▼
[DataLoader]  ← num_workers로 병렬 프리페치
  │  (GPU — src/utils/synthesis.py)
  │  FFT Convolution → BCM 모델링 → SNR 스케일링 → 믹싱
  ▼
[Model Input]
```

- **CPU**: 파일 I/O(느린 작업)를 `num_workers`로 병렬화
- **GPU**: 행렬 연산(FFT Conv, 필터링)을 배치 단위로 한 번에 처리

---

## Step 1. CPU: 데이터 로드 (Dataset)

**위치**: `src/data/dataset.py` — `SpatialMixingDataset.__getitem__`

디스크에서 "재료"를 꺼내오는 단계입니다. 무거운 연산은 일절 하지 않습니다.

### 로드하는 데이터

| 데이터 | Shape | 설명 |
|---|---|---|
| `raw_speech` | `(T,)` | 클린 음성 모노 파형. 랜덤 크롭하여 3초(48,000 샘플) 고정 |
| `raw_noises` | `(S_max-1, T)` | 소음 파형 묶음. 미사용 슬롯은 무음(0) 패딩 |
| `rir_tensor` | `(M, S_max, L)` | Room Impulse Response. 미사용 소스 슬롯은 0 패딩 |
| `snr` | `float` | 이 샘플에 적용할 목표 SNR (dB) |
| `mic_config` | `dict` | BCM 사용 여부, 컷오프 주파수, 감쇄량 등 |

**배치 구성 (DataLoader)**: 모든 텐서가 고정 크기이므로 `default_collate`로 자연스럽게 배치됩니다.

```python
# 배치 후 Shape (batch_size=4 예시)
raw_speech:  (4, 48000)
raw_noises:  (4, 7, 48000)    # S_max=8이면 노이즈 슬롯 7개
rir_tensor:  (4, 5, 8, 16000) # M=5채널, S_max=8소스, L=16000
snr:         (4,)              # float64 → synthesis에서 float32로 변환
```

---

## Step 2. GPU: 공간 합성 (synthesis.py)

**위치**: `src/utils/synthesis.py` — `apply_spatial_synthesis`

GPU로 이동한 배치를 물리적으로 공간화합니다. 전체 과정은 5단계로 구성됩니다.

### 2-1. FFT Convolution으로 공간화

```
raw_speech (B, T)  ×  rir_tensor[:, :, 0, :] (B, M, L)
                            ↓  fftconvolve
              speech_mc (B, M, T)   ← 방 안에서 말하는 소리

raw_noises (B, S-1, T)  ×  rir_tensor[:, :, 1:, :] (B, M, S-1, L)
                            ↓  fftconvolve (소스별 병렬)
              noise_mc (B, M, T)    ← 방 안에서 들리는 소음
```

`torchaudio.functional.fftconvolve`를 사용합니다. 시간 도메인 Convolution보다 수십 배 빠릅니다.

### 2-2. BCM 채널 물리 모델링

골전도(Bone Conduction) 마이크의 물리적 특성을 소프트웨어로 모사합니다.

**마지막 채널(`M-1`번)이 BCM 채널로 처리됩니다.**

| 처리 | 이유 | 구현 |
|---|---|---|
| Low-Pass Filter (500Hz) | 피부·근육을 통과하며 고주파 손실 | sinc+Hann 커널 (101 tap) |
| 소음 감쇄 (-20dB) | 골전도 마이크는 외부 공기 전도 소음을 잘 차단 | 노이즈 BCM 채널에 0.1 곱하기 |

**BCM 커널 캐싱**: 커널은 `SEModule.__init__`에서 1회 생성 후 `register_buffer`에 저장됩니다. 매 스텝마다 재생성하지 않습니다.

```python
# SEModule.__init__
bcm_kernel = create_bcm_kernel(cutoff_hz=500.0, sample_rate=16000, num_taps=101)
self.register_buffer('bcm_kernel', bcm_kernel)  # GPU 자동 이동, 저장/로드 포함
```

### 2-3. SNR 스케일링

공기전도 마이크(Channel 0) 기준 RMS 에너지로 목표 SNR을 달성하도록 소음 볼륨을 조절합니다.

```
target_factor = (speech_rms / 10^(snr/20)) / noise_rms
noise_mc = noise_mc × target_factor
```

> **dtype 주의**: `snr`이 Python float이면 DataLoader에서 `torch.float64`로 변환됩니다.
> `16-mixed` AMP는 float32만 half로 변환하므로, float64는 반드시 `.float()`로 명시 변환합니다.

### 2-4. 최종 믹싱

```python
noisy = speech_mc + noise_mc  # (B, M, T)
```

### 2-5. Aligned Dry (Dereverberation 타겟)

`target_type = "aligned_dry"` 학습 시, RIR peak 시간만큼 클린 음성을 시간 정렬하여 타겟으로 사용합니다.

```python
# RIR peak 인덱스 탐색 (벡터화, for 루프 없음)
peak_idx = rir_tensor[:, 0, 0, :].abs().argmax(dim=-1)
aligned_dry = roll_each(raw_speech, peak_idx)  # (B, 1, T)
```

---

## 최종 배치 구조

`apply_spatial_synthesis` 호출 후 배치에 추가되는 키:

| 키 | Shape | 설명 |
|---|---|---|
| `noisy` | `(B, M, T)` | 모델 입력: 공간화 + BCM + 소음 혼합 신호 |
| `clean` | `(B, M, T)` | 타겟 옵션 A: 잡음 제거 후 잔향 포함 신호 |
| `aligned_dry` | `(B, 1, T)` | 타겟 옵션 B: 완전 클린 + 시간 정렬 (dereverberation) |

---

## 합성 파이프라인 코드 위치

```
src/utils/synthesis.py
├── create_bcm_kernel()       ← sinc+Hann LPF 커널 생성 (1회)
├── spatialize_sources()      ← FFT Convolution
├── apply_bcm_modeling()      ← BCM 채널 LPF + 감쇄
├── scale_noise_to_snr()      ← 목표 SNR 달성
├── generate_aligned_dry()    ← Dereverberation 타겟 생성
└── apply_spatial_synthesis() ← 위 5개를 순서대로 호출하는 고수준 API
```

---

## 스크립트에서 합성 사용하기

`scripts/generate_samples.py`는 동일한 `apply_spatial_synthesis`를 사용하여 오프라인 샘플을 생성합니다.

```bash
# 합성 샘플 생성 (시각화·청취 목적)
uv run python scripts/generate_samples.py \
    --db_path data/metadata.db \
    --output_dir results/samples \
    --n_samples 10
```

---

## BCM 모델링 비활성화 (Ablation)

`mic_config['use_bcm'] = False`이거나 모델의 `in_channels=4`이면 BCM 채널 합성이 자동으로 생략됩니다. 전용 config를 사용하는 것을 권장합니다.

```bash
uv run python main.py fit --config configs/ic_conv_tasnet_bcm_off.yaml
```
