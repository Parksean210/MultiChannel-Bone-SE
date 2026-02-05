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
    Unified Database API for Speech, Noise, and RIR management.
    """
    def __init__(self, engine):
        self.engine = engine

    def index_speech(self, root_dir: str, dataset_name: str, is_eval: bool = False, sample_rate: int = 16000):
        """
        Index speech files (WAV or NPY) into the database.
        Calculates duration automatically.
        """
        root_dir = Path(root_dir)
        if not root_dir.exists():
            print(f"Error: Directory {root_dir} does not exist.")
            return

        print(f"Indexing speech from {root_dir} (Dataset: {dataset_name}, SR: {sample_rate})...")
        files = list(root_dir.rglob("*.wav")) + list(root_dir.rglob("*.npy"))
        
        with Session(self.engine) as session:
            new_entries = []
            for p in tqdm(files, desc="Indexing Speech"):
                path_str = str(p.absolute())
                
                # Double-check duplicates
                if session.exec(select(SpeechFile).where(SpeechFile.path == path_str)).first():
                    continue

                try:
                    duration, actual_sr = self._get_audio_info(p, sample_rate)
                    speaker_id = p.parent.name 
                    
                    speech = SpeechFile(
                        path=path_str,
                        dataset_name=dataset_name,
                        speaker_id=speaker_id,
                        duration_sec=duration,
                        sample_rate=actual_sr,
                        is_eval=is_eval
                    )
                    new_entries.append(speech)
                except Exception as e:
                    print(f"Warning: Failed to process {p}: {e}")

            self._commit_batches(session, new_entries)

    def index_noise(self, root_dir: str, category: str, sub_category: Optional[str] = None, sub_depth: int = 1, sample_rate: int = 16000):
        """
        Index noise files from a directory into the database.
        
        Args:
            root_dir: Root directory of noise data.
            category: Main category (urban, living, etc.)
            sub_category: Fixed sub-category name. If None, inferred via sub_depth.
            sub_depth: How many levels up from the file to pick the sub-category name (default: 1).
            sample_rate: Sample rate (for .npy duration calculation).
        """
        root_dir = Path(root_dir)
        if not root_dir.exists():
            print(f"Error: Directory {root_dir} does not exist.")
            return

        print(f"Indexing noise from {root_dir} (Category: {category}, Sub: {sub_category or f'Depth={sub_depth}'}, SR: {sample_rate})...")
        files = list(root_dir.rglob("*.wav")) + list(root_dir.rglob("*.npy"))

        with Session(self.engine) as session:
            new_entries = []
            for p in tqdm(files, desc="Indexing Noise"):
                path_str = str(p.absolute())

                if session.exec(select(NoiseFile).where(NoiseFile.path == path_str)).first():
                    continue

                try:
                    duration, actual_sr = self._get_audio_info(p, sample_rate)
                    
                    # Flexible Sub-category Logic
                    if sub_category:
                        final_sub = sub_category
                    else:
                        # Go up sub_depth levels
                        try:
                            final_sub = p.parents[sub_depth - 1].name
                        except (IndexError, AttributeError):
                            final_sub = "unknown"
                    
                    noise = NoiseFile(
                        path=path_str,
                        category=category,
                        sub_category=final_sub,
                        duration_sec=duration,
                        sample_rate=actual_sr
                    )
                    new_entries.append(noise)
                except Exception as e:
                    print(f"Warning: Failed to process {p}: {e}")

            self._commit_batches(session, new_entries)

    def index_rirs(self, root_dir: str):
        """
        Index RIR files (.pkl or .wav) into the database.
        """
        root_dir = Path(root_dir)
        if not root_dir.exists():
            return

        print(f"Indexing RIRs from {root_dir}...")
        files = list(root_dir.rglob("*.pkl")) + list(root_dir.rglob("*.wav"))

        with Session(self.engine) as session:
            new_entries = []
            for p in tqdm(files, desc="Indexing RIRs"):
                path_str = str(p.absolute())
                if session.exec(select(RIRFile).where(RIRFile.path == path_str)).first():
                    continue

                try:
                    rir = self._parse_rir(p)
                    new_entries.append(rir)
                except Exception as e:
                    print(f"Warning: Failed to process {p}: {e}")

            self._commit_batches(session, new_entries)

    def get_stats(self) -> Dict:
        """
        Returns a comprehensive summary of the database contents.
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
                    "types": [dict(name=row[0], count=row[1]) for row in session.exec(select(RIRFile.room_type, func.count(RIRFile.id)).group_by(RIRFile.room_type)).all()]
                }
            }
        return stats

    def sync_paths(self):
        """
        Automatically updates .wav paths to .npy if the original file is missing but .npy exists.
        Useful after bulk conversion.
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
        """Returns (duration_sec, sample_rate)"""
        if path.suffix.lower() == ".wav":
            info = sf.info(path)
            return info.duration, info.samplerate
        elif path.suffix.lower() == ".npy":
            data = np.load(path, mmap_mode='r')
            return data.shape[0] / sr_hint, sr_hint
        return 0.0, sr_hint

    def _check_and_update_path(self, item) -> bool:
        wav_p = Path(item.path)
        npy_p = wav_p.with_suffix(".npy")
        if not wav_p.exists() and npy_p.exists():
            item.path = str(npy_p.absolute())
            return True
        return False

    def _commit_batches(self, session, entries, batch_size=1000):
        if not entries:
            print("No new entries to add.")
            return
        
        for i in range(0, len(entries), batch_size):
            session.add_all(entries[i:i + batch_size])
            session.commit()
        print(f"Successfully added {len(entries)} files.")

    def _parse_rir(self, p: Path) -> RIRFile:
        # Internal parser for RIR files (logic relocated from manager.py)
        room_type = "unknown"; num_noise = 0; num_mic = 4; num_bcm = 1; rt60 = None
        if p.suffix == '.pkl':
            import pickle
            with open(p, 'rb') as f: data = pickle.load(f)
            meta = data.get('meta', {})
            room_type = meta.get('room_config', {}).get('room_type', 'unknown')
            num_mic = meta.get('mic_config', {}).get('relative_positions').shape[1] if meta.get('mic_config', {}).get('relative_positions') is not None else 0
            num_bcm = 1 if meta.get('mic_config', {}).get('use_bcm', False) else 0
            num_noise = sum(1 for s in data.get('source_info', []) if s['type'] == 'noise')
            rt60 = meta.get('rt60')
        else:
            parts = p.stem.split('_')
            room_type = parts[1] if len(parts) >= 2 else "unknown"
            for part in parts:
                if part.startswith("rt"): rt60 = float(part[2:])
                if part.startswith("n") and part[1:].isdigit(): num_noise = int(part[1:])
        
        return RIRFile(path=str(p.absolute()), room_type=room_type, num_noise=num_noise, num_mic=num_mic, num_bcm=num_bcm, rt60=rt60)
