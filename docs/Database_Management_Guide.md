# 🗄️ 데이터베이스 관리 가이드 (Database Management Guide)

이 문서는 본 프로젝트에서 사용하는 **SQLite 데이터베이스(`data/metadata.db`)**를 관리하는 방법을 아주 상세히 설명합니다.

우리는 수십만 개의 오디오 파일 경로와 메타정보(길이, 화자 등)를 DB에 저장해두고, 학습 시에 이를 고속으로 조회하여 사용합니다.

---

## 🚀 1. 가장 쉬운 사용법 (CLI 도구)

복잡한 코드 없이, 터미널 명령어로 데이터를 추가할 수 있습니다. 스크립트 위치는 `scripts/manage_db.py`입니다.

### 📊 데이터베이스 현황 확인 (Stats)
현재 DB에 저장된 파일 개수와 카테고리별 분포, **샘플 레이트 현황**을 한눈에 확인할 수 있습니다.
```bash
uv run python scripts/manage_db.py stats
```

### 🔄 경로 자동 동기화 (Sync)
`.wav` 파일을 `.npy`로 변환했거나 파일 위치를 옮겼을 때, DB에 저장된 경로를 실제 파일과 일치하도록 자동 업데이트합니다.
```bash
uv run python scripts/manage_db.py sync
```

### 🔊 노이즈 데이터 추가하기 (가장 중요)

새로운 노이즈 데이터를 다운로드 받거나 녹음했다면, 다음 명령어로 DB에 등록하세요.

**기본 명령어 포맷:**
```bash
uv run python scripts/manage_db.py noise --path "[폴더 경로]" --category "[카테고리]" --sr 16000
```

**실전 예시:**
만약 D 드라이브에 있는 `Living_Noise`라는 폴더를 추가하고 싶다면?

1. **(권장) 심볼릭 링크 생성**: 먼저 프로젝트 안으로 연결합니다.
   ```bash
   ln -s "/mnt/d/New_Noise_Data" data/raw/noise/new_living_noise
   ```

2. **DB에 등록**:
   ```bash
   uv run python scripts/manage_db.py noise \
       --path data/raw/noise/new_living_noise \
       --category living
   ```

3. **확인**: 등록이 완료되면 "Successfully added X noise files." 메시지가 뜹니다.

### 📂 폴더 구조가 모호할 경우 (Sub-category 수동 지정)

기본적으로 소분류(`sub_category`)는 데이터가 들어있는 **바로 위 폴더 이름**을 자동으로 사용합니다. 만약 폴더 이름이 모호하거나(`wav/`, `data/` 등), 직접 이름을 지정하고 싶다면 `--sub` 옵션을 사용하세요.

```bash
# --sub 또는 --sub_category 옵션 사용
uv run python scripts/manage_db.py noise \
    --path data/raw/noise/flat_folder \
    --category urban \
    --sub traffic  # 모든 파일의 소분류를 'traffic'으로 고정
```

만약 폴더 깊이가 깊어서 바로 위 폴더가 아닌 **더 상위 폴더** 이름을 쓰고 싶다면 `--sub_depth`를 쓰세요.
```bash
# --sub_depth 2: 두 단계 위 폴더 이름을 소분류로 사용
uv run python scripts/manage_db.py noise ... --sub_depth 2
```

> **💡 참고:** 이미 등록된 파일은 자동으로 건너뛰므로(Duplicate Check), 명령어를 여러 번 실행해도 안전합니다.

---

### 🗣️ 음성(Speech) 데이터 추가하기

KsponSpeech 같은 대용량 음성 데이터를 추가할 때 사용합니다.

```bash
```bash
uv run python scripts/manage_db.py speech \
    --path data/raw/speech/train \
    --dataset KsponSpeech \
    --sr 16000
```

*   `--eval`: 평가용 데이터라면 이 플래그를 붙여주세요.
    ```bash
    # 평가 데이터 등록 예시
    uv run python scripts/manage_db.py speech \
        --path data/raw/speech/eval \
        --dataset KsponSpeech \
        --eval
    ```

---

### 🏛️ RIR (공간 음향) 데이터 추가하기

시뮬레이션으로 생성된 RIR 파일(`.wav` 또는 `.pkl`)이 있는 폴더를 통째로 등록합니다.

```bash
uv run python scripts/manage_db.py rir --path data/rirs
```

---

## 💻 2. 파이썬 코드에서 사용하기 (API)

Jupyter Notebook이나 다른 스크립트에서 직접 DB를 조작하고 싶다면 `DatabaseManager`를 import해서 쓰세요.

```python
from src.db import create_db_engine, DatabaseManager

# 1. DB 연결 (엔진 생성)
# 파일이 없으면 자동으로 생성됩니다.
engine = create_db_engine("data/metadata.db")

# 2. 매니저 초기화
manager = DatabaseManager(engine)

# 3. 데이터 등록
# 노이즈 추가
manager.index_noise(
    root_dir="data/raw/noise/server_room", 
    category="machine"
)

# 음성 추가
manager.index_speech(
    root_dir="data/raw/speech/new_speaker",
    dataset_name="MyCustomVoice"
)
```

---

## 🏗️ 3. 데이터 구조 (Schema)

DB 내부가 어떻게 생겼는지 궁금하시다면 참고하세요. (`src/data/models.py`에 정의됨)

### Table: `speechfile`
| 필드명 | 설명 | 예시 |
| :--- | :--- | :--- |
| `id` | 고유 번호 | 1 |
| `path` | 파일 절대 경로 (Unique) | `/home/user/data/speech/file.wav` |
| `dataset_name` | 데이터셋 이름 | `KsponSpeech` |
| `speaker_id` | 화자 식별자 (폴더명 추론) | `KsponSpeech_0001` |
| `duration_sec` | 오디오 길이 (초) | 4.52 |
| `sample_rate` | 샘플 레이트 (Hz) | 16000 |
| `is_eval` | 평가 데이터 여부 | `False` (0) |

### Table: `noisefile`
| 필드명 | 설명 | 예시 |
| :--- | :--- | :--- |
| `id` | 고유 번호 | 1 |
| `path` | 파일 절대 경로 (Unique) | `/home/user/data/noise/car.wav` |
| `category` | 대분류 (사용자 입력) | `urban` |
| `sub_category` | 소분류 (폴더명 추론) | `traffic` |
| `duration_sec` | 오디오 길이 (초) | 10.0 |
| `sample_rate` | 샘플 레이트 (Hz) | 16000 |

### Table: `rirfile`
| 필드명 | 설명 | 예시 |
| :--- | :--- | :--- |
| `id` | 고유 번호 | 1 |
| `path` | 파일 절대 경로 (Unique) | `/home/user/data/rirs/rir_00001.pkl` |
| `room_type` | 방 형태 | `shoebox`, `l_shape`, `polygon` |
| `num_noise` | 시뮬레이션된 **노이즈 소스 개수** | 4 |
| `num_mic` | 시뮬레이션된 **마이크 개수** (Air) | 4 |
| `num_bcm` | 시뮬레이션된 **골전도 센서 개수** | 1 |
| `rt60` | 잔향 시간 (초) | 0.35 |

---

## ❓ 자주 묻는 질문 (FAQ)

**Q. 노이즈 폴더에 파일 하나만 추가했습니다. 전체를 다시 인덱싱해야 하나요?**  
A. 네, 그냥 `manage_db.py` 명령어를 똑같이 다시 실행하시면 됩니다.  
내부에서 **"이미 DB에 있는 경로는 무시(Skip)"**하도록 짜여 있어서, **새로 추가된 파일만 쏙 골라서** 1초 만에 등록됩니다.

**Q. DB 파일을 실수로 지웠습니다!**  
A. 걱정 마세요. `data/metadata.db` 파일은 언제든 다시 만들 수 있습니다.  
그냥 위의 명령어들을 다시 실행하면 파일들을 싹 훑어서 DB를 새로 구축해 줍니다.

---

## 🔍 5. SQLModel 퀵 시트 (Common Usage)

학습 코드나 분석 스크립트에서 데이터를 조회할 때 자주 사용하는 패턴들입니다.

### 세션 및 엔진 준비
```python
from sqlmodel import Session, select, func
from src.db import create_db_engine
from src.data.models import SpeechFile, NoiseFile, RIRFile

engine = create_db_engine("data/metadata.db")
```

### 데이터 조회 (Read)
```python
with Session(engine) as session:
    # 1. 전체 조회
    all_speech = session.exec(select(SpeechFile)).all()
    
    # 2. 필터링 (Where)
    # 예: 특정 데이터셋의 평가 데이터만 가져오기
    eval_files = session.exec(
        select(SpeechFile).where(
            SpeechFile.dataset_name == "KsponSpeech",
            SpeechFile.is_eval == True
        )
    ).all()
    
    # 3. 단일 항목 조회 (First)
    first_rir = session.exec(select(RIRFile)).first()
```

### 통계 및 정렬 (Stats & Sort)
```python
with Session(engine) as session:
    # 4. 개수 세기 (Count)
    total_noise = session.exec(select(func.count(NoiseFile.id))).one()
    
    # 5. 정렬 및 개수 제한 (Order by & Limit)
    # 예: 가장 긴 노이즈 10개 가져오기
    longest_noise = session.exec(
        select(NoiseFile)
        .order_by(NoiseFile.duration_sec.desc())
        .limit(10)
    ).all()
    
    # 6. 랜덤 샘플링 (SQLite 특화)
    random_sample = session.exec(
        select(SpeechFile).order_by(func.random()).limit(1)
    ).first()
```

### 데이터 추가/삭제 (Create/Delete)
```python
with Session(engine) as session:
    # 추가
    # new_item = SpeechFile(...)
    # session.add(new_item)
    
    # 삭제 (조회 후 삭제)
    # target = session.exec(select(SpeechFile).where(...)).first()
    # if target:
    #     session.delete(target)
    
    session.commit() # 변경사항 저장
```
