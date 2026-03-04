# test_model/ - 오픈소스 모델 분석 공간

## 목적
오픈소스 모델을 **읽고 이해하기 위한 전용 공간**입니다.
여기서 분석한 내용을 바탕으로 `src/models/`에 커스텀 모델을 직접 구현합니다.

**프로덕션 코드가 아닙니다. 학습/실험용으로만 사용합니다.**

## 현재 분석 중인 모델

### mamba/ — Mamba SSM (state-spaces/mamba)
- **출처**: https://github.com/state-spaces/mamba
- **논문**: Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)
- **목표**: Mamba 블록 구조를 이해하여 음성 향상용 IC-Mamba 모델 구현

#### 핵심 파일 (분석 순서 권장)
| 파일 | 설명 |
|---|---|
| `mamba_ssm/modules/mamba_simple.py` | Mamba1 블록 — 입문용, 핵심 selective scan 로직 |
| `mamba_ssm/modules/mamba2.py` | Mamba2 (SSD) — 멀티헤드 구조, 실제 사용 권장 |
| `mamba_ssm/ops/selective_scan_interface.py` | S6 알고리즘 Python/CUDA 인터페이스 |
| `mamba_ssm/modules/block.py` | Mamba Block = Mamba + LayerNorm + Residual |
| `mamba_ssm/models/mixer_seq_simple.py` | 블록을 쌓은 전체 시퀀스 모델 예시 |

#### 자체 분석 문서 (`mamba/docs/grand_audit/`)
직접 작성한 코드-수식 매핑 문서. 핵심 파일별로 라인 단위 해설 포함.

## 다음 단계: src/models/ 구현 계획

여기서 습득한 내용으로 구현할 예정인 모델:

```
src/models/
├── ic_conv_tasnet.py   ← 현재 기준 모델 (IC-Conv-TasNet)
├── ic_mamba.py         ← 구현 예정: Mamba 기반 IC 음성 향상 모델
└── base.py             ← BaseSEModel (in_channels, forward 인터페이스)
```

### 구현 시 준수할 인터페이스
```python
class ICMamba(BaseSEModel):
    def __init__(self, in_channels: int = 5, ...):
        super().__init__(in_channels=in_channels, ...)
        # self.in_channels 자동 설정됨

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input:  (B, in_channels, T)
        # Output: (B, 1, T)  ← Channel 0 (ref mic) enhanced output
        ...
```

SEModule은 `model.in_channels`만 참조하므로 BaseSEModel을 상속하면
YAML config 변경만으로 IC-ConvTasNet ↔ IC-Mamba 교체 가능.

## 주의
- `test_model/`의 코드를 직접 import하거나 `src/`에서 참조하지 않습니다
- 완성된 커스텀 구현만 `src/models/`에 배치합니다
