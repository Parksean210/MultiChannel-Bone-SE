# 🏟️ RIR 시뮬레이션 가이드 (RIR Simulation Guide)

이 문서는 **공간 음향 데이터(Room Impulse Response, RIR)**를 생성하고 관리하는 방법을 설명합니다. 특히 시뮬레이션 결과물인 `.pkl` 파일에 포함된 상세 메타데이터 구조를 집중적으로 다룹니다.

---

## 🛠️ 1. RIR 생성하기 (Generation)

`scripts/generate_rir_bank.py` 스크립트를 사용하여 대량의 RIR을 멀티프로세싱으로 생성합니다.

```bash
# 기본 실행 (1,000개 생성)
uv run python scripts/generate_rir_bank.py --count 1000
```

---

## 💾 2. .pkl 파일 메타데이터 구조 (Metadata Detail)

시뮬레이션 결과물인 `rir_{id}.pkl` 파일은 단순한 오디오 신호 이상의 방대한 시뮬레이션 정보를 담고 있습니다.

### 📝 파일 구조 개요
```python
import pickle
with open("rir_00000.pkl", "rb") as f:
    data = pickle.load(f)

# data의 상위 키
# - data['rirs']: 실제 시뮬레이션된 RIR 신호
# - data['meta']: 시뮬레이션 환경 설정 정보
# - data['source_info']: 음원(Target/Noise)의 위치 정보
```

### 🔍 1. `data['meta']` 상세
| 키 (Key) | 설명 | 데이터 예시 |
| :--- | :--- | :--- |
| `fs` | 샘플링 레이트 | `16000` |
| `rt60` | 측정된(또는 이론상) 잔향 시간 | `0.35` (단위: 초) |
| `rir_gain` | Normalize 전의 global peak 값 | `0.12` (원본 크기 복원 시 사용) |
| `mic_pos` | 마이크들의 방 내 **절대 좌표** (3xN) | `[[x1, x2...], [y1, y2...], [z1, z2...]]` |
| `room_config` | 방의 기하학적 정보 및 재질 | 아래 상세 참조 |
| `mic_config` | 마이크 배열 및 골전도 센서 설정 | 아래 상세 참조 |

#### 🏠 `meta['room_config']` (방 정보)
- `room_type`: `shoebox`, `l_shape`, `polygon` 중 하나.
- `dimensions`: [Width, Depth, Height] (shoebox인 경우).
- `corners`: 다각형 방인 경우의 2D 코너 좌표.
- `materials`: 바닥, 천장, 벽면(`east`, `west`, `north`, `south`)에 부여된 재질 이름.
- `max_order`: 반사 이미지 소스 계산 차수 (기본 7~10).
- `use_ray_tracing`: 하이브리드 시뮬레이션(Ray Tracing) 사용 여부.

#### 🎙️ `meta['mic_config']` (마이크 배열 상세)
- **`relative_positions`**: 중심점 기준 마이크들의 상대 좌표 (3xN). **이것이 실제 마이크 위치 메타데이터입니다.**
- `name`: 설정 이름 (`Default_4Mic_Glasses`).
- `use_bcm`: 골전도 센서 포함 여부.
- `bcm_rel_pos`: 중심점 기준 골전도 센서의 상대 좌표.
- `bcm_cutoff_hz`: 골전도 저역 통과 필터 컷오프 주파수.
- `bcm_noise_attenuation_db`: 골전도 센서에서 소음이 감쇄되는 정도.

### 🔊 2. `data['source_info']` (음원 정보)
음원들은 리스트 형태로 저장되며, 각 항목은 위치와 타입을 가집니다.
```python
# [
#   {"pos": [x, y, z], "type": "target"}, # 사용자 입(Mouth) 위치
#   {"pos": [x, y, z], "type": "noise"},  # 환경 소음원 1
#   ...
# ]
```

## 🎨 3. 시각화 (Visualization)

생성된 RIR의 방 구조와 파형을 시각적으로 확인하려면 전용 도구(`scripts/visualize_rirs.py`)를 사용하세요.

### 단일 파일 확인
특정 RIR 파일을 지정하여 이미지(`.png`)로 뽑아냅니다.
```bash
uv run python scripts/visualize_rirs.py data/rirs/rir_00001.pkl
```
*결과물: `data/rirs/rir_00001.png` 생성*

### 대량 확인 (Batch Processing)
폴더 전체를 지정하면 폴더 내의 모든 `.pkl` 파일을 이미지로 변환하여 `viz/` 하위 폴더에 저장합니다.
```bash
uv run python scripts/visualize_rirs.py data/rirs
```
*결과물: `data/rirs/viz/` 폴더에 모든 이미지 저장*

### 시각화 내용
- **Room Layout (2D/3D)**: 방의 형태와 마이크, 타겟, 노이즈 음원의 위치.
- **RIR Waveforms**: 각 마이크 채널별, 음원별 임펄스 응답 파형.

---

## 🗄️ 4. 데이터베이스 등록 (Indexing)

이 모든 정보 중 핵심 스탯은 `manage_db.py`를 통해 DB에 자동 추출되어 저장됩니다.

```bash
uv run python scripts/manage_db.py rir --path data/rirs
```

**DB(`rirfile` 테이블)에 추출되는 항목:**
- `room_type`, `num_noise`, `num_mic`, `num_bcm`, `rt60`

---

## ⚡ 4. 팁 (Tips)

*   **마이크 위치 커스텀**: `src/simulation/config.py`에서 `relative_positions`를 수정하면 즉시 시뮬레이션에 반영되며, 이 정보는 생성된 모든 `.pkl` 파일의 메타데이터에 기록됩니다.
*   **재질 확인**: `room_config['materials']`에 기록된 재질 이름은 `pyroomacoustics`의 라이브러리 명칭과 매칭됩니다.
