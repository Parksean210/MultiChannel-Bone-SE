# 🏗️ BaseSEModel 아키텍처 가이드

**`BaseSEModel`**은 본 프로젝트의 모든 음성 향상(Speech Enhancement) 모델이 상속받아야 하는 **기반 클래스(Parent Class)**입니다. 

## 1. 핵심 철학 (Philosophy)
1.  **Unified Interface**: 어떤 모델을 쓰더라도 입력과 출력은 항상 **Raw Waveform `(Batch, Channels, Time)`** 형태여야 합니다.
2.  **Toolbox Included**: 주파수 변환(STFT)이나 타임 프레임 조각내기(Framing) 같은 복잡한 전처리는 모델 내부에서 부모 클래스의 메서드로 해결합니다.

---

## 2. 주요 메서드 (Methods)

### 🔊 주파수 도메인 (Frequency-domain)
주로 CRN, DCCRN 등 스펙트로그램(Spectrogram)을 입력으로 받는 모델을 위해 사용합니다.

#### `stft(x)`
- **기능**: Waveform을 복소수(Complex) 스펙트로그램으로 변환합니다.
- **입력**: `(Batch, Channels, Time)`
- **출력**: `(Batch, Channels, Freq, Time)`
- **내부 함수**: `torch.stft(..., return_complex=True)`

#### `istft(x_spec)`
- **기능**: 스펙트로그램을 다시 Waveform으로 복원합니다.
- **입력**: `(Batch, Channels, Freq, Time)`
- **출력**: `(Batch, Channels, Time)`
- **내부 함수**: `torch.istft`

---

### ⏱️ 타임 도메인 (Time-domain)
주로 RNN, Transformer 등 긴 시계열을 짧은 프레임으로 쪼개서 처리하는 모델을 위해 사용합니다.

#### `to_frames(x, center=True)`
- **기능**: Waveform을 윈도우를 적용하여 여러 프레임으로 조각냅니다 (Overlap 지원).
- **입력**: `(Batch, Channels, Time)`
- **출력**: `(Batch, Channels, NumFrames, WinLength)`
- **내부 함수**:
    - `F.pad`: 양 끝단 정보 손실 방지를 위한 Reflection Padding.
    - `F.unfold`: 텐서를 슬라이딩 윈도우 방식으로 펼쳐줍니다.

#### `from_frames(frames, length=None)`
- **기능**: 조각난 프레임들을 다시 겹쳐서 원본 파형으로 합칩니다 (Overlap-and-Add).
- **입력**: `(Batch, Channels, NumFrames, WinLength)`
- **출력**: `(Batch, Channels, Time)`
- **원리**: 윈도우가 겹쳐지며 커진 에너지를 보정하기 위해, `window^2`의 합으로 나누어 정규화합니다.
- **내부 함수**: `F.fold`

---

## 3. 내부 사용된 PyTorch 핵심 함수

이 클래스가 마법(?)을 부리기 위해 내부적으로 사용한 PyTorch의 저수준(Low-level) 함수들입니다.

| 함수 | 설명 | 사용처 |
| :--- | :--- | :--- |
| **`F.unfold`** | 이미지를 패치 단위로 뜯어낼 때 주로 쓰지만, 여기서는 1D 오디오를 윈도우 단위로 뜯어내는 데 사용했습니다. | `to_frames` |
| **`F.fold`** | 뜯어진 패치들을 다시 원본 캔버스 위치에 더해줍니다(Summation). Overlap-and-Add 구현의 핵심입니다. | `from_frames` |
| **`F.pad (reflect)`** | 거울처럼 반사되는 패딩을 적용하여, 시작과 끝부분에서 윈도우 때문에 값이 0이 되는 현상을 막습니다. | `to_frames` |
| **`register_buffer`** | 윈도우 함수(Hann 등)를 모델의 '상태'로 등록하여, 모델이 GPU로 이동할 때 윈도우도 같이 따라가게 만듭니다. | `__init__` |

---

## 4. 모델 구현 예시

### 예시 1: 스펙트로그램 모델 (CRN 등)
```python
class MySpecModel(BaseSEModel):
    def forward(self, x):
        # 1. 변환 (Wave -> Spec)
        spec = self.stft(x) 
        
        # 2. 지능적 처리 (마스크 예측 등)
        mask = self.network(torch.abs(spec))
        enhanced_spec = spec * mask
        
        # 3. 복원 (Spec -> Wave)
        return self.istft(enhanced_spec, length=x.shape[-1])
```

### 예시 2: 타임 프레임 모델 (RNN 등)
```python
class MyTimeModel(BaseSEModel):
    def forward(self, x):
        # 1. 쪼개기 (Wave -> Frames)
        frames = self.to_frames(x) # (B, C, N, W)
        
        # 2. 프레임별 처리
        # (Batch*Frames, WinLength) 형태로 바꿔서 MLP 통과 등...
        out_frames = self.network(frames)
        
        # 3. 합치기 (Frames -> Wave)
        return self.from_frames(out_frames, length=x.shape[-1])
```
