# 데이터베이스 관리 가이드

> 수십만 개의 오디오 파일 경로와 메타정보를 SQLite DB에 인덱싱하여 고속 조회합니다.
> 파일을 복사하거나 이동할 필요 없이, **경로만 등록**하는 방식입니다.

---

## 개요

```
data/metadata.db  ← SQLite 단일 파일
├── speechfile    ← 음성 데이터 경로 + 메타정보
├── noisefile     ← 소음 데이터 경로 + 메타정보
└── rirfile       ← RIR 데이터 경로 + 메타정보
```

모든 DB 작업은 `scripts/manage_db.py`를 통해 CLI로 수행합니다.

---

## 1. 현황 확인

```bash
uv run python scripts/manage_db.py stats
```

출력 예시:

```
=== Database Statistics ===
[Speech]  train: 45,231  val: 5,654  test: 5,654  total: 56,539
[Noise]   train: 8,420   val: 1,052  test: 1,052  total: 10,524
[RIR]     train: 4,000   val:   500  test:   500  total:  5,000
```

---

## 2. 데이터 등록

### 음성(Speech) 등록

```bash
uv run python scripts/manage_db.py speech \
    --path data/speech/KsponSpeech \
    --dataset KsponSpeech \
    --language ko
```

| 옵션 | 필수 | 설명 |
|---|---|---|
| `--path` | ✅ | 음성 파일이 있는 폴더 경로 (하위 폴더 포함 재귀 탐색) |
| `--dataset` | ✅ | 데이터셋 이름 (임의 지정, 필터링에 활용) |
| `--language` | | 언어 코드 (예: `ko`, `en`) |
| `--speaker` | | 화자 ID (생략 시 폴더명에서 자동 추론) |

### 소음(Noise) 등록

```bash
uv run python scripts/manage_db.py noise \
    --path data/noise/traffic \
    --dataset TrafficNoise \
    --category "교통소음" \
    --sub "도로"
```

| 옵션 | 필수 | 설명 |
|---|---|---|
| `--path` | ✅ | 소음 파일이 있는 폴더 경로 |
| `--dataset` | ✅ | 데이터셋 이름 |
| `--category` | | 대분류 (예: "교통소음", "공장소음") |
| `--sub` | | 소분류 (예: "도로", "프레스기") |

### RIR 등록

```bash
uv run python scripts/manage_db.py rir \
    --path data/rirs \
    --dataset SimRIR_v1
```

`.pkl` 파일에서 `room_type`, `rt60`, `num_mic`, `num_bcm` 등을 자동 추출하여 DB에 저장합니다.

---

## 3. Train / Val / Test 분할

### 기본 분할 (8:1:1)

```bash
uv run python scripts/manage_db.py realloc --type speech --ratio 0.8 0.1 0.1
uv run python scripts/manage_db.py realloc --type noise  --ratio 0.8 0.1 0.1
uv run python scripts/manage_db.py realloc --type rir    --ratio 0.8 0.1 0.1
```

### 고정 분할 정책 (중요)

`realloc`은 **이미 `val` 또는 `test`로 지정된 데이터를 절대 `train`으로 돌리지 않습니다.**

```
기존: [val: 500개] [test: 500개] [train: 4,000개]
새 데이터 5,000개 추가 후 realloc 실행:
결과: [val: 500개(동일)] [test: 500개(동일)] [train: 9,000개]
```

이 덕분에 데이터셋을 확장하더라도 **기존 검증/테스트 결과의 신뢰성이 유지**됩니다.

---

## 4. 기타 유틸리티

### 경로 동기화 (파일 이동 후)

파일을 다른 위치로 옮겼을 때 DB 경로를 업데이트합니다.

```bash
uv run python scripts/manage_db.py sync
```

### 중복 방지

`manage_db.py`는 이미 등록된 경로는 자동으로 건너뜁니다. 같은 명령을 두 번 실행해도 중복 등록되지 않습니다.

---

## 5. Python API로 직접 사용

스크립트나 Jupyter Notebook에서 DB를 직접 조회·수정할 수 있습니다.

```python
from src.db import create_db_engine, DatabaseManager

engine = create_db_engine("data/metadata.db")
manager = DatabaseManager(engine)

# 소음 데이터 등록
manager.index_noise(
    root_dir="data/noise/new_dataset",
    dataset_name="NewNoise_v2",
    category="생활소음",
    sub_category="청소기"
)

# 음성 데이터 등록
manager.index_speech(
    root_dir="data/speech/new_speaker",
    dataset_name="CustomVoice",
    language="ko"
)
```

---

## 6. DB 스키마

### `speechfile` 테이블

| 컬럼 | 타입 | 설명 | 예시 |
|---|---|---|---|
| `id` | int | 고유 번호 (PK) | 1 |
| `path` | str | 파일 절대 경로 (Unique) | `/data/speech/ks_001.wav` |
| `dataset_name` | str | 데이터셋 이름 | `KsponSpeech` |
| `speaker` | str | 화자 ID | `Kspon_0001` |
| `language` | str | 언어 코드 | `ko` |
| `duration_sec` | float | 오디오 길이 (초) | 4.52 |
| `sample_rate` | int | 샘플 레이트 (Hz) | 16000 |
| `split` | str | 분할 영역 | `train` / `val` / `test` |

### `noisefile` 테이블

| 컬럼 | 타입 | 설명 | 예시 |
|---|---|---|---|
| `id` | int | 고유 번호 (PK) | 1 |
| `path` | str | 파일 절대 경로 (Unique) | `/data/noise/car.wav` |
| `dataset_name` | str | 데이터셋 이름 | `TrafficNoise` |
| `category` | str | 대분류 | `교통소음` |
| `sub_category` | str | 소분류 | `도로` |
| `duration_sec` | float | 오디오 길이 (초) | 10.0 |
| `sample_rate` | int | 샘플 레이트 (Hz) | 16000 |
| `split` | str | 분할 영역 | `train` |

### `rirfile` 테이블

| 컬럼 | 타입 | 설명 | 예시 |
|---|---|---|---|
| `id` | int | 고유 번호 (PK) | 1 |
| `path` | str | 파일 절대 경로 (Unique) | `/data/rirs/rir_001.pkl` |
| `dataset_name` | str | 데이터셋 이름 | `SimRIR_v1` |
| `room_type` | str | 방 형태 | `shoebox` |
| `num_noise` | int | 노이즈 소스 개수 | 4 |
| `num_mic` | int | 마이크 개수 | 4 |
| `num_bcm` | int | 골전도 센서 개수 | 1 |
| `rt60` | float | 잔향 시간 (초) | 0.35 |
| `split` | str | 분할 영역 | `train` |

---

## 7. 심볼릭 링크 활용 (대용량 데이터)

데이터가 외부 드라이브나 NAS에 있는 경우, 심볼릭 링크로 연결합니다.

```bash
# 외부 드라이브의 데이터를 프로젝트 내부에서 참조
ln -s /mnt/d/BigNoiseDataset data/noise/big_noise

# DB에 등록 (링크 경로 사용)
uv run python scripts/manage_db.py noise \
    --path data/noise/big_noise \
    --dataset BigNoise
```

---

## 8. FAQ

**Q. 파일을 추가했는데 DB에 반영이 안 됩니다.**

동일 명령을 다시 실행하면 새로 추가된 파일만 등록됩니다.

```bash
uv run python scripts/manage_db.py noise --path data/noise/traffic --dataset TrafficNoise
```

**Q. `metadata.db`를 실수로 삭제했습니다.**

걱정 없습니다. 파일 자체는 그대로이므로 `manage_db.py` 명령들을 순서대로 다시 실행하면 DB가 재생성됩니다. `realloc`까지 실행해야 split이 지정됩니다.

**Q. 특정 데이터셋만 학습에 사용하고 싶습니다.**

현재는 DB 전체가 사용됩니다. 특정 데이터셋만 사용하려면 `SEDataModule`의 `init_args`에 필터 옵션을 추가하거나, 해당 데이터셋만 등록한 별도 DB를 만들어 `db_path`를 지정합니다.

**Q. 샘플 레이트가 다른 파일이 섞여 있습니다.**

`SEDataModule`의 `target_sr: 16000` 설정에 의해 DataLoader 시점에서 자동으로 리샘플링됩니다. 단, 디스크 I/O가 늘어날 수 있으므로 사전에 변환해두는 것을 권장합니다.
