import os
from pathlib import Path
from typing import Optional, List, Dict
import soundfile as sf
import numpy as np
from sqlmodel import Session, select, func
from tqdm import tqdm
from src.data.models import SpeechFile, NoiseFile, RIRFile

class DatabaseManager:
    """
    고속 학습을 위한 음성, 노이즈, RIR 메타데이터 통합 관리 클래스.
    SQLite 데이터베이스를 기반으로 인덱싱, 경로 동기화 및 통계 기능을 제공합니다.
    """
    def __init__(self, engine):
        """
        Args:
            engine: SQLModel/SQLAlchemy 데이터베이스 엔진 객체
        """
        self.engine = engine

    def index_speech(self, root_dir: str, dataset_name: str, speaker: Optional[str] = None, language: str = "ko", split: str = "train", sample_rate: int = 16000):
        """
        입력 디렉토리를 스캔하여 음성 파일(WAV, NPY)의 메타데이터를 DB에 인덱싱합니다.
        
        Args:
            root_dir: 음성 데이터 최상위 디렉토리
            dataset_name: 데이터셋 식별 명칭 (예: 'LibriSpeech')
            speaker: 화자 식별자 (None인 경우 폴더이름 사용)
            language: 언어 (ko, en 등)
            split: 데이터 분할 종류 ('train', 'val', 'test')
            sample_rate: .npy 파일의 실제 샘플 레이트 (길이 계산용)
        """
        root_dir = Path(root_dir)
        if not root_dir.exists():
            print(f"Error: Directory {root_dir} does not exist.")
            return

        print(f"Indexing speech from {root_dir} (Dataset: {dataset_name}, Split: {split}, Lang: {language})...")
        files = list(root_dir.rglob("*.wav")) + list(root_dir.rglob("*.npy"))
        
        with Session(self.engine) as session:
            new_entries = []
            for p in tqdm(files, desc="Indexing Speech"):
                path_str = str(p.absolute())
                
                if session.exec(select(SpeechFile).where(SpeechFile.path == path_str)).first():
                    continue

                try:
                    duration, actual_sr = self._get_audio_info(p, sample_rate)
                    final_speaker = speaker if speaker else p.parent.name
                    
                    speech = SpeechFile(
                        path=path_str,
                        dataset_name=dataset_name,
                        speaker=final_speaker,
                        language=language,
                        duration_sec=duration,
                        sample_rate=actual_sr,
                        split=split
                    )
                    new_entries.append(speech)
                except Exception as e:
                    print(f"Warning: Failed to process {p}: {e}")

            self._commit_batches(session, new_entries)

    def index_noise(self, root_dir: str, dataset_name: str, category: Optional[str] = None, sub_category: Optional[str] = None, sub_depth: int = 1, sample_rate: int = 16000, split: str = "train"):
        """
        잡음 데이터를 데이터베이스에 인덱싱합니다.
        
        Args:
            root_dir: 잡음 데이터 최상위 디렉토리
            dataset_name: 데이터셋 명칭
            category: 대분류 (None인 경우 폴더명 자동 파싱 시도)
            sub_category: 소분류 명칭 (None인 경우 폴더명 자동 파싱 시도)
            sub_depth: 자동 파싱 시 소분류 추출을 위한 디렉토리 계층 깊이
            sample_rate: .npy 파일의 샘플 레이트
            split: 데이터 분할 종류 ('train', 'val', 'test')
        """
        root_dir = Path(root_dir)
        if not root_dir.exists():
            print(f"Error: Directory {root_dir} does not exist.")
            return

        print(f"Indexing noise from {root_dir} (Dataset: {dataset_name}, Split: {split})...")
        files = list(root_dir.rglob("*.wav")) + list(root_dir.rglob("*.npy"))

        with Session(self.engine) as session:
            new_entries = []
            for p in tqdm(files, desc="Indexing Noise"):
                path_str = str(p.absolute())

                if session.exec(select(NoiseFile).where(NoiseFile.path == path_str)).first():
                    continue

                try:
                    duration, actual_sr = self._get_audio_info(p, sample_rate)
                    
                    # Manual labels have priority
                    if category and sub_category:
                        final_cat, final_sub = category, sub_category
                    else:
                        folder_name = p.parents[sub_depth - 1].name
                        # Handle TS_/VS_ prefix pattern as fallback
                        if (folder_name.startswith("VS_") or folder_name.startswith("TS_")) and "_" in folder_name:
                            parts = folder_name.split("_")
                            if len(parts) >= 3:
                                final_cat = parts[1].split(".")[-1] if "." in parts[1] else parts[1]
                                final_sub = parts[2].split(".")[-1] if "." in parts[2] else parts[2]
                            else:
                                final_cat = category or folder_name
                                final_sub = sub_category or "unknown"
                        else:
                            final_cat = category or folder_name
                            final_sub = sub_category or "unknown"
                    
                    noise = NoiseFile(
                        path=path_str,
                        dataset_name=dataset_name,
                        category=final_cat,
                        sub_category=final_sub,
                        duration_sec=duration,
                        sample_rate=actual_sr,
                        split=split
                    )
                    new_entries.append(noise)
                except Exception as e:
                    print(f"Warning: Failed to process {p}: {e}")

            self._commit_batches(session, new_entries)

    def reallocate_splits(self, table_type: str, ratios: tuple = (0.8, 0.1, 0.1)):
        """
        데이터베이스에 이미 등록된 파일들을 지정된 비율(Train:Val:Test)로 무작위 재배치합니다.
        
        Args:
            table_type: 'speech' 또는 'noise'
            ratios: (Train, Val, Test) 비율 합이 1.0이어야 함
        """
        assert sum(ratios) == 1.0, "Ratios must sum to 1.0"
        
        if table_type == 'speech':
            model = SpeechFile
        elif table_type == 'noise':
            model = NoiseFile
        elif table_type == 'rir':
            model = RIRFile
        else:
            raise ValueError(f"Unknown table_type: {table_type}")
        print(f"Reallocating {table_type} splits with ratios {ratios}...")
        
        from sqlmodel import func
        with Session(self.engine) as session:
            # 1. 고정 분할 정책: 이미 val, test인 데이터는 건드리지 않고 train인 데이터만 대상으로 재배치
            # 이를 통해 데이터 추가 시 기존 검증/테스트 데이터셋의 오염(Leakage)을 방지함
            total_count = session.exec(select(func.count(model.id))).one()
            
            # 현재 train 상태인 아이템들만 가져옴
            train_items = session.exec(select(model).where(model.split == "train")).all()
            np.random.shuffle(train_items)

            # 목표 수량 계산 (전체 데이터 대비 비율)
            target_val_total = int(total_count * ratios[1])
            target_test_total = int(total_count * ratios[2])

            # 이미 할당된 수량 확인
            current_val_count = session.exec(select(func.count(model.id)).where(model.split == "val")).one()
            current_test_count = session.exec(select(func.count(model.id)).where(model.split == "test")).one()

            # 새로 추가할 수량
            needed_val = max(0, target_val_total - current_val_count)
            needed_test = max(0, target_test_total - current_test_count)

            print(f"Current split: val={current_val_count}, test={current_test_count} (Total items in DB: {total_count})")
            print(f"Targeting total: val={target_val_total}, test={target_test_total}")
            print(f"Moving {needed_val} items to val, {needed_test} items to test from current train set...")

            # 순서대로 할당
            for i, item in enumerate(train_items):
                if i < needed_val:
                    item.split = "val"
                elif i < needed_val + needed_test:
                    item.split = "test"
                else:
                    break
                session.add(item)
            
            session.commit()
            print(f"Successfully reallocated {table_type} splits. (val: +{needed_val}, test: +{needed_test})")

    def index_rirs(self, root_dir: str, dataset_name: str = "unknown", split: str = "train", sample_rate: int = 16000):
        """
        RIR 데이터(.pkl 또는 .wav)를 스캔하여 메타데이터를 인덱싱합니다.
        """
        root_dir = Path(root_dir)
        if not root_dir.exists():
            print(f"Error: Directory {root_dir} does not exist.")
            return

        print(f"Indexing RIRs from {root_dir} (Dataset: {dataset_name}, Split: {split}, SR: {sample_rate})...")
        files = list(root_dir.rglob("*.pkl")) + list(root_dir.rglob("*.wav"))

        with Session(self.engine) as session:
            new_entries = []
            for p in tqdm(files, desc="Indexing RIRs"):
                path_str = str(p.absolute())
                if session.exec(select(RIRFile).where(RIRFile.path == path_str)).first():
                    continue

                try:
                    rir = self._parse_rir(p)
                    # Override with explicit metadata
                    rir.dataset_name = dataset_name
                    rir.split = split
                    rir.sample_rate = sample_rate
                    new_entries.append(rir)
                except Exception as e:
                    print(f"Warning: Failed to process {p}: {e}")

            self._commit_batches(session, new_entries)

    def get_stats(self) -> Dict:
        """
        데이터베이스에 저장된 전체 데이터 현황(데이터셋별, 카테고리별 수량 등)을 요약하여 반환합니다.
        """
        with Session(self.engine) as session:
            stats = {
                "speech": {
                    "total": session.exec(select(func.count(SpeechFile.id))).one(),
                    "npy": session.exec(select(func.count(SpeechFile.id)).where(SpeechFile.path.like("%.npy"))).one(),
                    "sample_rates": [dict(sr=row[0], count=row[1]) for row in session.exec(select(SpeechFile.sample_rate, func.count(SpeechFile.id)).group_by(SpeechFile.sample_rate)).all()],
                    "datasets": [dict(name=row[0], count=row[1]) for row in session.exec(select(SpeechFile.dataset_name, func.count(SpeechFile.id)).group_by(SpeechFile.dataset_name)).all()]
                },
                "noise": {
                    "total": session.exec(select(func.count(NoiseFile.id))).one(),
                    "npy": session.exec(select(func.count(NoiseFile.id)).where(NoiseFile.path.like("%.npy"))).one(),
                    "sample_rates": [dict(sr=row[0], count=row[1]) for row in session.exec(select(NoiseFile.sample_rate, func.count(NoiseFile.id)).group_by(NoiseFile.sample_rate)).all()],
                    "categories": [dict(name=row[0], count=row[1]) for row in session.exec(select(NoiseFile.category, func.count(NoiseFile.id)).group_by(NoiseFile.category)).all()]
                },
                "rir": {
                    "total": session.exec(select(func.count(RIRFile.id))).one(),
                    "duration_stats": [dict(duration=row[0], count=row[1]) for row in session.exec(select(RIRFile.duration_sec, func.count(RIRFile.id)).group_by(RIRFile.duration_sec)).all()],
                    "types": [dict(name=row[0], count=row[1]) for row in session.exec(select(RIRFile.room_type, func.count(RIRFile.id)).group_by(RIRFile.room_type)).all()]
                }
            }
        return stats

    def sync_paths(self):
        """
        데이터베이스의 파일 경로를 동기화합니다. 
        동일 경로에 .wav 대신 .npy 파일이 존재하는 경우 이를 우선적으로 사용하도록 업데이트합니다.
        """
        print("Synchronizing database paths (.wav -> .npy)...")
        with Session(self.engine) as session:
            updated = 0
            # Check Speech
            for item in session.exec(select(SpeechFile).where(SpeechFile.path.like("%.wav"))).all():
                if self._check_and_update_path(item): updated += 1
            
            # Check Noise
            for item in session.exec(select(NoiseFile).where(NoiseFile.path.like("%.wav"))).all():
                if self._check_and_update_path(item): updated += 1
            
            if updated > 0:
                session.commit()
                print(f"Successfully synchronized {updated} paths.")
            else:
                print("Database is already in sync.")

    def _get_audio_info(self, path: Path, sr_hint: int) -> tuple[float, int]:
        """
        오디오 파일의 지속 시간(초)과 샘플 레이트를 추출합니다.
        .npy 포맷은 메모리 형상 정보를 활용하여 초고속으로 파싱합니다.
        """
        if path.suffix.lower() == ".wav":
            info = sf.info(path)
            return info.duration, info.samplerate
        elif path.suffix.lower() == ".npy":
            data = np.load(path, mmap_mode='r')
            return data.shape[0] / sr_hint, sr_hint
        return 0.0, sr_hint

    def _check_and_update_path(self, item) -> bool:
        """
        파일의 존재 여부를 확인하고, .wav 파일이 유실되었으나 .npy가 존재하는 경우 경로를 업데이트합니다.
        """
        wav_p = Path(item.path)
        npy_p = wav_p.with_suffix(".npy")
        if not wav_p.exists() and npy_p.exists():
            item.path = str(npy_p.absolute())
            return True
        return False

    def _commit_batches(self, session: Session, entries: List, batch_size: int = 1000):
        """
        대량의 데이터를 배치 단위로 나누어 트랜잭션을 커밋합니다. DB 부하를 최소화합니다.
        """
        if not entries:
            print("추가할 데이터가 없습니다.")
            return
        
        for i in range(0, len(entries), batch_size):
            session.add_all(entries[i:i + batch_size])
            session.commit()
        print(f"성공적으로 {len(entries)}개의 데이터를 추가했습니다.")

    def _parse_rir(self, p: Path) -> RIRFile:
        """
        RIR 파일(.pkl 또는 .wav)을 파싱하여 RIRFile 모델 객체를 생성합니다.
        """
        room_type = "unknown"; num_noise = 0; num_mic = 4; num_bcm = 1; rt60 = None; duration_sec = 1.0
        if p.suffix == '.pkl':
            import pickle
            with open(p, 'rb') as f: data = pickle.load(f)
            meta = data.get('meta', {})
            room_type = meta.get('room_config', {}).get('room_type', 'unknown')
            num_mic = meta.get('mic_config', {}).get('relative_positions').shape[1] if meta.get('mic_config', {}).get('relative_positions') is not None else 0
            num_bcm = 1 if meta.get('mic_config', {}).get('use_bcm', False) else 0
            num_noise = sum(1 for s in data.get('source_info', []) if s['type'] == 'noise')
            rt60 = meta.get('rt60')
            # Extract duration if exists, otherwise fallback to RIR tensor length
            duration_sec = meta.get('rir_len_sec')
            if duration_sec is None and data.get('rirs'):
                duration_sec = len(data['rirs'][0][0]) / meta.get('fs', 16000)
        else:
            parts = p.stem.split('_')
            room_type = parts[1] if len(parts) >= 2 else "unknown"
            for part in parts:
                if part.startswith("rt"): rt60 = float(part[2:])
                if part.startswith("n") and part[1:].isdigit(): num_noise = int(part[1:])
                if part.startswith("len"): duration_sec = float(part[3:])
        
        return RIRFile(
            path=str(p.absolute()), 
            room_type=room_type, 
            num_noise=num_noise, 
            num_mic=num_mic, 
            num_bcm=num_bcm, 
            rt60=rt60,
            duration_sec=duration_sec or 1.0
        )
