# 모델 아키텍처 가이드

> 새 모델을 추가하는 방법과 `BaseSEModel` 인터페이스를 설명합니다.
> 인터페이스를 지키면 YAML 한 줄로 모델을 교체할 수 있습니다.

---

## 구조 개요

```
src/models/
├── base.py               ← BaseSEModel (모든 모델의 부모 클래스)
├── ic_mamba.py           ← ICMamba (mamba-ssm CUDA 커널, causal) ← 메인
├── ic_conv_tasnet.py     ← ICConvTasNet (Dilated TCN 기반, 비교용)
└── baseline.py           ← DummyModel (파이프라인 검증용)
```

**SEModule**(`src/modules/se_module.py`)은 `model.in_channels`만 참조합니다. `BaseSEModel`을 상속하면 나머지는 자동으로 연결됩니다.

---

## BaseSEModel 인터페이스

### 필수 구현 사항

```python
class MyNewModel(BaseSEModel):
    def __init__(self, in_channels: int = 5, ...):
        super().__init__(in_channels=in_channels)   # ← 반드시 전달
        # self.in_channels가 자동으로 설정됨

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 입력: (B, in_channels, T)
        # 출력: (B, 1, T)  ← Channel 0 (참조 마이크) enhanced output
        ...
```

**규칙**:
- `in_channels`를 `super().__init__()`에 전달해야 `SEModule.forward`의 자동 슬라이싱이 작동합니다.
- 출력은 반드시 `(B, 1, T)` 또는 `(B, C, T)` 형태여야 합니다. SEModule이 `[:, 0:1, :]`로 Channel 0만 사용합니다.

### 제공되는 유틸리티 메서드

`BaseSEModel`에 이미 구현되어 있어 자유롭게 사용할 수 있습니다.

#### 주파수 도메인

```python
# Waveform → Complex Spectrogram
spec = self.stft(x)         # (B, C, T) → (B, C, F, T) complex

# Complex Spectrogram → Waveform
wave = self.istft(spec, length=T)  # (B, C, F, T) → (B, C, T)
```

#### 타임 프레임 도메인

```python
# Waveform → 겹치는 프레임
frames = self.to_frames(x)         # (B, C, T) → (B, C, N_frames, WinLen)

# 프레임 → Waveform (Overlap-and-Add)
wave = self.from_frames(frames, length=T)  # (B, C, N_frames, WinLen) → (B, C, T)
```

---

## 메인 모델: ICMamba

`src/models/ic_mamba.py`

mamba-ssm의 공식 CUDA 커널 기반 causal SSM 모델입니다. ICConvTasNet과 동일한 Encoder/Bottleneck/Decoder 구조에서 분리(Separation) 모듈만 Mamba로 교체했습니다.

### 핵심 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `in_channels` | 5 | 입력 마이크 채널 수 (5: BCM 포함, 4: 제외) |
| `enc_kernel` | 256 | Encoder 1D Conv 커널 크기 (16ms @ 16kHz) |
| `enc_num_feats` | 512 | Encoder 출력 차원 |
| `bot_num_feats` | 128 | Bottleneck 차원 (= Mamba d_model) |
| `d_state` | 64 | SSM 상태 차원. 클수록 표현력 증가 |
| `d_conv` | 4 | Mamba 내부 depthwise conv 커널 크기 |
| `expand` | 2 | d_inner = expand × bot_num_feats = 256 |
| `num_layers` | 12 | Mamba 레이어 수 |
| `use_checkpoint` | true | Gradient Checkpointing (RTX 3080 권장) |

### 아키텍처 흐름

```
(B, M, T)
  → Encoder (Conv1d)          → (B, M*F, L)
  → Bottleneck (Conv1d)       → (B, M*N, L)
  → ChannelProj               → (B*C, L, N)   ← 채널별 독립 처리
  → MambaLayer × num_layers   → (B*C, L, N)   ← causal SSM
  → MaskGen (Sigmoid)         → (B, C, F, L)
  → Decoder (ConvTranspose1d) → (B, M, T)
```

### YAML 설정 예시

```yaml
model:
  class_path: src.models.ICMamba
  init_args:
    in_channels: 5
    bot_num_feats: 128
    d_state: 64
    num_layers: 12
    use_checkpoint: true    # RTX 3080(10GB): true / A100/H100: false
```

### 레이턴시

causal 설계로 최소 레이턴시는 인코더 윈도우 크기입니다.

```
enc_kernel = 256 samples → 16ms @ 16kHz
enc_stride = 128 samples →  8ms 업데이트 주기
```

스트리밍 추론 시 `mamba_ssm.utils.generation.InferenceParams`로 상태를 유지하며 프레임 단위 처리가 가능합니다.

---

## 비교 모델: ICConvTasNet

`src/models/ic_conv_tasnet.py`

다채널 음성 향상을 위한 IC(Inter-Channel)-Conv-TasNet입니다.

### 핵심 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `in_channels` | 5 | 입력 마이크 채널 수 (5: BCM 포함, 4: 제외) |
| `enc_kernel` | 256 | Encoder 1D Conv 커널 크기 |
| `enc_num_feats` | 512 | Encoder 출력 차원 |
| `tcn_hidden` | 256 | TCN 내부 채널 수 |
| `num_layers` | 8 | TCN 레이어 수 |
| `num_stacks` | 3 | TCN 스택 수 |
| `use_checkpoint` | true | Gradient Checkpointing (소형 GPU에서 VRAM 절약) |

### YAML 설정 예시

```yaml
model:
  class_path: src.models.ICConvTasNet
  init_args:
    in_channels: 5
    enc_kernel: 256
    enc_num_feats: 512
    tcn_hidden: 256
    num_layers: 8
    num_stacks: 3
    use_checkpoint: true    # RTX 3080(10GB): true / A100: false
```

---

## 새 모델 추가 방법

### Step 1. 모델 파일 생성

`src/models/my_model.py`를 만들고 `BaseSEModel`을 상속합니다.

```python
# src/models/my_model.py
import torch
from src.models.base import BaseSEModel


class MyModel(BaseSEModel):
    def __init__(
        self,
        in_channels: int = 5,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__(in_channels=in_channels)

        self.encoder = torch.nn.Conv1d(in_channels, hidden_dim, kernel_size=16, stride=8)
        self.separator = torch.nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = torch.nn.ConvTranspose1d(hidden_dim, 1, kernel_size=16, stride=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, T)
        enc = self.encoder(x)                          # (B, hidden_dim, T')
        out, _ = self.separator(enc.transpose(1, 2))   # (B, T', hidden_dim)
        dec = self.decoder(out.transpose(1, 2))        # (B, 1, T)
        return dec
```

### Step 2. `src/models/__init__.py`에 등록

```python
from src.models.my_model import MyModel
```

### Step 3. YAML config 작성

```yaml
# configs/my_model.yaml
model:
  class_path: src.modules.se_module.SEModule
  init_args:
    model:
      class_path: src.models.MyModel       # ← 여기만 바꾸면 됩니다
      init_args:
        in_channels: 5
        hidden_dim: 256
        num_layers: 4
    loss:
      class_path: src.modules.losses.CompositeLoss
      init_args:
        alpha: 0.1
    optimizer_config:
      lr: 1e-3
      weight_decay: 1e-5
trainer:
  logger:
    init_args:
      experiment_name: "Architecture"
      run_name: "MyModel-5ch"
  # ... (나머지는 ic_conv_tasnet.yaml과 동일)
```

### Step 4. 학습

```bash
uv run python main.py fit --config configs/my_model.yaml
```

MLflow에서 `model_type: MyModel`로 자동 태깅됩니다.

---

## 스펙트로그램 기반 모델 예시

```python
class MySpecModel(BaseSEModel):
    def __init__(self, in_channels: int = 5, hidden: int = 128):
        super().__init__(in_channels=in_channels)
        freq_bins = self.n_fft // 2 + 1   # 257 (n_fft=512 기준)
        self.mask_net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels * 2, hidden, 3, padding=1),  # real+imag
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden, 2, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-1]
        spec = self.stft(x)                          # (B, C, F, T_frames) complex
        # Real/Imag을 채널로 결합
        feat = torch.cat([spec.real, spec.imag], dim=1)
        mask = self.mask_net(feat)                   # (B, 2, F, T_frames)
        # 마스크 적용 (Channel 0만)
        enhanced_spec = spec[:, 0:1, :, :] * torch.view_as_complex(
            mask.permute(0, 2, 3, 1).contiguous()
        )
        return self.istft(enhanced_spec, length=T)   # (B, 1, T)
```

---

## 타임 프레임 기반 모델 예시

```python
class MyFrameModel(BaseSEModel):
    def __init__(self, in_channels: int = 5, hidden: int = 256):
        super().__init__(in_channels=in_channels)
        frame_len = self.win_length  # BaseSEModel의 윈도우 길이
        self.net = torch.nn.GRU(
            in_channels * frame_len, hidden, batch_first=True
        )
        self.out = torch.nn.Linear(hidden, frame_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-1]
        frames = self.to_frames(x)              # (B, C, N, WinLen)
        B, C, N, W = frames.shape
        inp = frames.reshape(B, N, C * W)       # (B, N, C*WinLen)
        out, _ = self.net(inp)                  # (B, N, hidden)
        out_frames = self.out(out)              # (B, N, WinLen)
        out_frames = out_frames.unsqueeze(1)    # (B, 1, N, WinLen)
        return self.from_frames(out_frames, length=T)  # (B, 1, T)
```

---

## BCM ablation 자동 처리

`in_channels`를 4로 설정하면 SEModule이 자동으로 5채널 입력에서 앞 4채널만 모델에 전달합니다.

```python
# src/modules/se_module.py - forward
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (B, 5, T) — 항상 5채널 입력
    # model.in_channels=4라면 앞 4채널만 슬라이싱
    return self.model(x[:, :self.model.in_channels, :].contiguous())
```

코드 수정 없이 `in_channels: 4` YAML 설정만으로 BCM 채널이 자동 제외됩니다.
