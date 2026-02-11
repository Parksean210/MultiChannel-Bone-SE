# 🧪 데이터 합성 가이드 (Data Synthesis Guide)

본 문서는 `SpatialMixingDataset` (`src/data/dataset.py`)이 어떻게 원본 음성과 소음 데이터를 RIR(공간 임펄스 응답)과 결합하여 학습용 멀티채널 데이터를 **실시간(On-the-fly) 고속 합성**하는지 설명합니다.

---

## � 1. 핵심 개선 사항 (Optimization)

기존의 느린 CPU 연산을 개선하여 대규모 데이터 학습에 적합하도록 최적화되었습니다.

*   **FFT Convolution**: 기존 `scipy.signal.convolve` (시간 도메인) 대비 수십 배 빠른 `torchaudio.functional.fftconvolve` (주파수 도메인) 사용. 긴 RIR 필터 적용 시 병목 현상 제거.
*   **Tensor-Centric**: 데이터 로드 직후부터 `PyTorch Tensor`로 변환하여, 불필요한 `NumPy <-> Torch` 변환 오버헤드 최소화.
*   **Fixed Chunking**: 모든 출력을 3초(48,000 samples)로 강제 Crop/Pad 하여 `DataLoader`의 배치 구성을 안정화.

---

## 🏗️ 2. 데이터 합성 파이프라인 (Pipeline)

본 프로젝트는 효율성을 위해 **"데이터 로딩은 CPU(Dataset)가, 무거운 합성 연산은 GPU(SEModule)가"** 처리하는 이 단계 구성을 가집니다.

```mermaid
graph TD
    subgraph "CPU (Dataset)"
        A[Load Raw Speech] --> B[Load Raw Noises]
        B --> C[Load RIR Tensor]
    end
    
    subgraph "GPU (SEModule)"
        C --> D[FFT Convolution]
        D --> E[BCM Modeling]
        E --> F[SNR Scaling]
        F --> G[Final Mixing]
    end
```

### 🔍 상세 로직 분석

#### 1. Audio Loading & Shaping (CPU)
*   **코드 위치**: `src/data/dataset.py` -> `__getitem__`
*   **기능**: `soundfile`, `torchaudio` 또는 `.npy` (mmap)으로 오디오를 읽고, 설정된 `chunk_size`에 맞춰 랜덤 자르기(Crop) 혹은 제로 패딩(Pad)을 수행하여 `(T,)` 형태의 오디오 텐서를 반환합니다.

#### 2. GPU 기반 실시간 합성 (GPU)
*   **코드 위치**: `src/modules/se_module.py` -> `_apply_gpu_synthesis`
*   **핵심 기술**:
    *   **FFT Convolution**: `torchaudio.functional.fftconvolve`를 사용하여 Batch 단위의 다채널 컨볼루션을 고속 처리합니다.
    *   **BCM Modeling**: 마지막 채널에 저대역 통과 필터(LPF) 및 잡음 감쇄(Attenuation)를 적용하여 골전도 센서 특성을 모사합니다.
    *   **Dynamic Mixing**: 매 배치마다 랜덤하게 설정된 SNR에 맞춰 음성과 소음을 혼합합니다.

### 🔍 상세 로직 분석

#### 1. Audio Loading & Shaping
*   **코드 위치**: `__getitem__` 초반부
*   **기능**: `soundfile`로 읽은 오디오를 즉시 텐서로 변환하고, 설정된 `chunk_size`(기본 48,000)에 맞춰 랜덤 자르기(Crop) 혹은 제로 패딩(Pad)을 수행합니다. 이는 GPU 텐서 연산의 효율을 극대화합니다.

#### 2. RIR Application (FFT Convolution)
*   **코드 위치**: `_apply_rir` 메서드
*   **핵심 기술**:
    ```python
    # (1, T) * (M, R) -> (M, T+R-1) FFT Convolution
    output = F_audio.fftconvolve(audio, rir_tensor, mode="full")
    ```
    마이크 개수($M$)만큼의 컨볼루션을 한 번의 FFT 연산으로 처리하여 속도를 비약적으로 높입니다.

#### 3. BCM (Bone Conduction) Physics
*   **코드 위치**: `_apply_bcm_modeling` 메서드
*   **기능**: 마지막 채널(Channel 4)에 골전도 센서의 물리적 특성을 입힙니다.
    1.  **LPF (Low Pass Filter)**: 500Hz 이하 주파수만 통과 (피부 진동 특성)
    2.  **Noise High Attenuation**: 외부 소음은 공기 전도 대비 약 20dB 감쇄 (차음 효과)

#### 3. SNR Scaling & Mixing
*   **코드 위치**: `SEModule.py` -> `_apply_gpu_synthesis`
*   **로직**:
    *   에너지 계산 시 **BCM 채널을 제외한** 공기 전도 마이크(Air Mics)만을 기준으로 삼습니다.
    *   랜덤하게 설정된 `snr_range` (예: -5~20dB)에 맞춰 소음의 진폭을 조절한 뒤 음성과 합칩니다.

---

## 📦 3. 반환 데이터 구조 (Dataset Output)

`Dataset` 클래스가 `DataLoader`를 통해 모델로 전달하는 원본 재료 데이터입니다.

| Key | Shape | 설명 및 용도 |
| :--- | :--- | :--- |
| `raw_speech` | `(T,)` | 공간감이 입혀지기 전의 깨끗한 음성 (Mono) |
| `raw_noises` | `(S-1, T)` | 공간감이 입혀지기 전의 노이즈 원본들 |
| `rir_tensor` | `(M, S, L)` | 마이크별/위치별 공간 임펄스 응답 (RIR) |
| `num_sources`| `int` | 현재 RIR에서 사용 가능한 실제 소스 개수 |
| `snr` | `float` | 이 샘플에 적용될 목표 SNR 값 |
| `mic_config` | `dict` | BCM 사용 여부 등 마이크 설정 정보 |

---

## ⚙️ 4. 사용 방법

### 데이터셋 초기화
```python
from src.data.dataset import SpatialMixingDataset

dataset = SpatialMixingDataset(
    db_path="data/metadata.db",
    target_sr=16000,
    chunk_size=48000,
    split="train"  # "train", "val", "test" 중 선택
)
```

### 성능 팁 (Performance Tip)
*   **Num Workers**: `DataLoader`에서 `num_workers`를 충분히(4~8) 주어야 합니다. FFT 연산은 빠르지만, 데이터 로딩과 전처리는 병렬로 처리하는 것이 유리합니다.
