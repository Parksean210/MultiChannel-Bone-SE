# src/models/ - Model Architectures (Pure PyTorch)

## Overview
음성 향상 모델의 순수 PyTorch 구현부. Lightning 의존성 없이 독립적으로 동작한다.

## Files
| File | Class | Description |
|---|---|---|
| `base.py` | `BaseSEModel` | 모든 SE 모델의 추상 부모 클래스. STFT/iSTFT 유틸리티, 프레임 분할/합성(OLA) 제공 |
| `baseline.py` | `DummyModel` | 파이프라인 검증용 Identity 모델 (STFT -> iSTFT 통과만 수행) |
| `ic_conv_tasnet.py` | `ICConvTasNet` | 논문 기반 Inter-Channel Conv-TasNet 구현 (메인 모델) |
| `ic_mamba.py` | `ICMamba` | Conv-TasNet의 TCN 블록을 Mamba SSM으로 대체한 모델 (**dev/mamba-integration** 전용) |
| `ic_mamba2_bcm_guide.py` | `ICMamba2BCMGuide` | Mamba2 기반 + BCM 채널 가이드 어텐션 (**dev/mamba-integration** 전용) |
| `ic_mamba2_bcm_guide_v2.py` | `ICMamba2BCMGuideV2` | Mamba2 + FiLM 컨디셔닝 + ReLU 마스크 개선 버전 (**dev/mamba-integration** 전용) |
| `ic_mamba2_ft.py` | `ICMamba2FT` | ICMamba2BCMGuideV2 파인튜닝 전용 래퍼 (**dev/mamba-integration** 전용) |
| `spatial_net.py` | `SpatialNet` | CrossBand + NarrowBand 블록 기반 주파수 도메인 모델 |
| `spatial_net_fsq_split.py` | `SpatialNetFSQSplit` | SpatialNet + FSQ Split Computing (AR Glass ↔ Mobile Phone) |

## BaseSEModel Interface
모든 자식 모델은 반드시 `forward(x: (B, C, T)) -> (B, C, T)` 시그니처를 구현해야 한다.

주요 유틸리티:
- `stft(x)`: (B, C, T) -> (B, C, F, T) 복소수 스펙트로그램
- `istft(x_spec)`: (B, C, F, T) -> (B, C, T) 파형 복원
- `to_frames(x)` / `from_frames(frames)`: 시간 도메인 프레임 분할/OLA 합성

## ICConvTasNet Architecture
```
Input (B, M, T)
  -> Shared Encoder (Conv1d: 1->F) per channel
  -> Bottleneck (Conv1d: F->N)
  -> Channel Projection (Conv2d: M->C)
  -> IC-TCN Blocks x (S stacks * D layers)
     - 2D Depthwise Conv (Feature x Time)
     - 1x1 Conv (Channel mixing)
     - Residual + Skip connections
  -> Skip Accumulation
  -> Mask Estimation (Conv2d: C->M, Linear: N->F, Sigmoid)
  -> Masking (Element-wise multiply with encoder output)
  -> Shared Decoder (ConvTranspose1d: F->1)
Output (B, M, T)
```

**Default Parameters**: M=5, C=64, F=512, N=128, H=256, D=8, S=3, K=256

## ICMamba2BCMGuideV2 Architecture (dev/mamba-integration)
```
Input (B, M, T)  — M번째(마지막) 채널이 BCM
  -> Shared Encoder (Conv1d: 1->F) per channel
  -> Bottleneck (Conv1d: F->N)
  -> Channel Projection (Conv2d: M->C)
  -> BCM 채널 분리 -> FiLM 컨디셔닝 파라미터(γ, β) 생성
  -> Mamba2 Blocks x num_layers
     - FiLM(γ, β)으로 각 레이어 입력 컨디셔닝 (BCM이 처리 흐름 가이드)
     - Mamba2 SSM (d_state, d_conv, expand, headdim 설정)
  -> ReLU 기반 마스크 추정 (Sigmoid 대신 ReLU로 비음수 마스크)
  -> Masking + Shared Decoder (ConvTranspose1d: F->1)
Output (B, M, T)
```

**Default Parameters**: M=5, C=64, F=512, N=128, d_state=128, d_conv=4, expand=2, headdim=64, num_layers=8

## Adding a New Model
1. `BaseSEModel`을 상속한 새 클래스 생성
2. `forward(x: (B, C, T)) -> (B, C, T)` 구현
3. `__init__.py`에 import 추가
4. YAML에서 `class_path: src.models.YourNewModel`로 지정

## Notes
- `use_checkpoint=True`: Gradient Checkpointing 활성화 (메모리 절약, 속도 감소). RTX 3080(10GB)에서 권장, A100/H100에서는 False.
- SEModule의 `forward`에서 `model.in_channels`를 참조하여 자동 채널 슬라이싱 수행
- Mamba 계열 모델은 `mamba-ssm`, `causal-conv1d` 패키지 별도 설치 필요 (CUDA/Python/PyTorch 버전 호환 whl 직접 설치)
- Mamba 계열은 `strategy: "ddp"` 사용 (`ddp_find_unused_parameters_true` 사용 시 에러 발생)
