# RIR 시뮬레이션 가이드

> Room Impulse Response(RIR)는 특정 공간에서 소리가 어떻게 반사·흡수되는지를 담은 필터입니다.
> 이를 음성에 Convolution하면 실제 방 안에서 말하는 것처럼 만들 수 있습니다.
> `pyroomacoustics` 라이브러리를 활용하여 다양한 방 환경을 시뮬레이션합니다.

---

## 1. RIR 생성

### 기본 실행

```bash
# 1,000개 RIR 생성 (약 10~30분 소요)
uv run python scripts/generate_rir_bank.py --count 1000
```

생성된 파일은 `data/rirs/rir_XXXXX.pkl` 형식으로 저장됩니다.

### 생성 파라미터 커스텀

`src/simulation/config.py`에서 방 크기, 마이크 위치, 재질 등을 조정할 수 있습니다.

```python
# src/simulation/config.py (예시)
MIC_CONFIG = {
    "name": "Default_4Mic_Glasses",
    "use_bcm": True,                        # BCM 채널 포함 여부
    "bcm_cutoff_hz": 500.0,                 # BCM LPF 컷오프 (Hz)
    "bcm_noise_attenuation_db": 20.0,       # BCM 소음 감쇄 (dB)
    "relative_positions": [                 # 마이크 상대 좌표 (중심 기준, 단위: m)
        [-0.04,  0.0,  0.0],  # Mic 0: 좌전면
        [ 0.04,  0.0,  0.0],  # Mic 1: 우전면
        [-0.04, -0.05, 0.0],  # Mic 2: 좌후면
        [ 0.04, -0.05, 0.0],  # Mic 3: 우후면
    ],
    "bcm_rel_pos": [0.0, 0.0, -0.02],      # BCM 위치 (관자놀이 부근)
}
```

### 시뮬레이션 환경 범위

`generate_rir_bank.py`는 다음 범위에서 랜덤 샘플링합니다.

| 파라미터 | 범위 | 설명 |
|---|---|---|
| 방 크기 | 3~8m × 3~8m × 2.4~3.5m | Shoebox 방 |
| RT60 | 0.1~0.8초 | 잔향 시간 |
| 소음원 수 | 1~7개 | 방 내 소음 소스 |
| 재질 | 콘크리트~카펫 등 | pyroomacoustics 내장 재질 |

---

## 2. .pkl 파일 내부 구조

각 RIR 파일은 단순 오디오가 아닌 풍부한 메타정보를 담고 있습니다.

```python
import pickle

with open("data/rirs/rir_00001.pkl", "rb") as f:
    data = pickle.load(f)

# 상위 키
# data['rirs']       → 실제 RIR 신호 텐서 (M, S, L)
# data['meta']       → 시뮬레이션 환경 설정
# data['source_info'] → 음원 위치 및 타입
```

### `data['meta']` 상세

| 키 | 설명 | 예시 |
|---|---|---|
| `fs` | 샘플링 레이트 | `16000` |
| `rt60` | 잔향 시간 (초) | `0.35` |
| `mic_pos` | 마이크 절대 좌표 (3×M) | `[[x1...], [y1...], [z1...]]` |
| `room_config` | 방 기하학 정보 | 아래 참조 |
| `mic_config` | 마이크 배열 설정 | 아래 참조 |

#### `room_config`

| 키 | 설명 | 예시 |
|---|---|---|
| `room_type` | 방 형태 | `shoebox` |
| `dimensions` | [Width, Depth, Height] (m) | `[5.2, 4.8, 2.7]` |
| `materials` | 바닥·천장·벽 재질 | `{"floor": "carpet", "ceiling": "concrete", ...}` |
| `max_order` | 이미지 소스법 계산 차수 | `7` |

#### `mic_config`

| 키 | 설명 |
|---|---|
| `name` | 마이크 배열 이름 |
| `use_bcm` | BCM 센서 포함 여부 |
| `relative_positions` | 중심 기준 마이크 상대 좌표 (3×M) |
| `bcm_rel_pos` | BCM 센서 상대 좌표 |
| `bcm_cutoff_hz` | BCM LPF 컷오프 |
| `bcm_noise_attenuation_db` | BCM 소음 감쇄량 |

### `data['source_info']`

```python
# 음원 리스트
[
    {"pos": [x, y, z], "type": "target"},  # 사용자 입 위치
    {"pos": [x, y, z], "type": "noise"},   # 소음원 1
    {"pos": [x, y, z], "type": "noise"},   # 소음원 2
    ...
]
```

---

## 3. RIR 시각화

생성된 RIR을 이미지로 확인합니다.

```bash
# 단일 파일 시각화
uv run python scripts/visualize_rirs.py data/rirs/rir_00001.pkl
# → data/rirs/rir_00001.png 생성

# 폴더 전체 시각화
uv run python scripts/visualize_rirs.py data/rirs/
# → data/rirs/viz/ 폴더에 모든 파일 저장
```

시각화에 포함되는 내용:
- **2D/3D Room Layout**: 방 형태, 마이크 위치, 음원 위치
- **RIR Waveforms**: 채널별·소스별 임펄스 응답 파형

---

## 4. DB 등록

생성 후 DB에 등록해야 학습에 사용됩니다.

```bash
uv run python scripts/manage_db.py rir \
    --path data/rirs \
    --dataset SimRIR_v1

# 분할 지정
uv run python scripts/manage_db.py realloc --type rir --ratio 0.8 0.1 0.1
```

DB에 자동으로 추출되는 정보: `room_type`, `num_noise`, `num_mic`, `num_bcm`, `rt60`

---

## 5. RIR 텐서 형식

Dataset이 DataLoader에 전달하는 RIR 텐서의 형식입니다.

```
rir_tensor: (M, S_max, L)
  M       = 마이크 수 (4 또는 5, BCM 포함 시 5)
  S_max   = 최대 소스 수 (음성 1개 + 노이즈 최대 7개 = 8)
  L       = RIR 길이 (샘플 단위, 보통 8,000~16,000)
```

배치 구성 후: `(B, M, S_max, L)`

미사용 소스 슬롯은 0으로 패딩됩니다.

---

## 6. 팁

**다양한 RT60 범위 설정**: RT60이 작을수록(0.1~0.2초) 잔향이 적은 환경, 클수록(0.6~0.8초) 큰 홀·공장 환경입니다. 극한 소음 환경을 타겟으로 한다면 RT60 0.3~0.6초 범위를 중점적으로 생성합니다.

**마이크 위치 변경**: `src/simulation/config.py`의 `relative_positions`를 수정하면 이후 생성되는 모든 RIR에 반영됩니다. 기존 RIR과의 일관성을 위해 데이터셋별로 config를 버전 관리하는 것을 권장합니다.

**대용량 생성 시**: `--count` 수치가 크면 멀티프로세싱이 자동 활성화됩니다. CPU 코어 수가 많을수록 빠릅니다. 10,000개 기준 약 2~4시간(8코어).
