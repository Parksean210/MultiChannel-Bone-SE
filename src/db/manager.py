import os
from pathlib import Path
from typing import Optional, List
import soundfile as sf
from sqlmodel import Session, select
from tqdm import tqdm
from src.data.models import SpeechFile, NoiseFile, RIRFile

class DatabaseManager:
    def __init__(self, engine):
        self.engine = engine

    def index_speech(self, root_dir: str, dataset_name: str, is_eval: bool = False):
        """
        Index speech files from a directory into the database.
        """
        root_dir = Path(root_dir)
        if not root_dir.exists():
            print(f"Error: Directory {root_dir} does not exist.")
            return

        print(f"Indexing speech from {root_dir} (Dataset: {dataset_name})...")
        files = list(root_dir.rglob("*.wav"))
        
        with Session(self.engine) as session:
            new_entries = []
            for p in tqdm(files, desc="Processing Speech"):
                path_str = str(p.absolute())
                
                # Check for duplicates
                statement = select(SpeechFile).where(SpeechFile.path == path_str)
                results = session.exec(statement)
                if results.first():
                    continue

                try:
                    info = sf.info(p)
                    # Extract speaker_id if possible (assuming KsponSpeech format: KsponSpeech_01/KsponSpeech_0001/...)
                    # Fallback to parent folder name
                    speaker_id = p.parent.name 
                    
                    speech = SpeechFile(
                        path=path_str,
                        dataset_name=dataset_name,
                        speaker_id=speaker_id,
                        duration_sec=info.duration,
                        is_eval=is_eval
                    )
                    new_entries.append(speech)
                except Exception as e:
                    print(f"Warning: Failed to process {p}: {e}")

            if new_entries:
                session.add_all(new_entries)
                session.commit()
                print(f"Successfully added {len(new_entries)} speech files.")
            else:
                print("No new speech files to add.")

    def index_noise(self, root_dir: str, category: str, sub_category: Optional[str] = None):
        """
        Index noise files from a directory.
        Category: e.g., 'urban', 'living', 'traffic'
        Sub-category: Optional manual override. If None, inferred from parent directory.
        """
        root_dir = Path(root_dir)
        if not root_dir.exists():
            print(f"Error: Directory {root_dir} does not exist.")
            return

        print(f"Indexing noise from {root_dir} (Category: {category}, Sub: {sub_category or 'Auto'})...")
        files = list(root_dir.rglob("*.wav"))

        with Session(self.engine) as session:
            new_entries = []
            for p in tqdm(files, desc="Processing Noise"):
                path_str = str(p.absolute())

                # Check for duplicates
                statement = select(NoiseFile).where(NoiseFile.path == path_str)
                results = session.exec(statement)
                if results.first():
                    continue

                try:
                    info = sf.info(p)
                    # Sub-category override or auto-infer
                    final_sub = sub_category if sub_category else p.parent.name
                    
                    noise = NoiseFile(
                        path=path_str,
                        category=category,
                        sub_category=final_sub,
                        duration_sec=info.duration
                    )
                    new_entries.append(noise)
                except Exception as e:
                    print(f"Warning: Failed to process {p}: {e}")

            if new_entries:
                session.add_all(new_entries)
                session.commit()
                print(f"Successfully added {len(new_entries)} noise files.")
            else:
                print("No new noise files to add.")

    def index_rirs(self, root_dir: str):
        """
        Index RIR files from a directory.
        """
        root_dir = Path(root_dir)
        if not root_dir.exists():
            print(f"Error: Directory {root_dir} does not exist.")
            return

        print(f"Indexing RIRs from {root_dir}...")
        # Support both .wav and .pkl
        files = list(root_dir.rglob("*.pkl")) + list(root_dir.rglob("*.wav"))

        with Session(self.engine) as session:
            new_entries = []
            for p in tqdm(files, desc="Processing RIRs"):
                path_str = str(p.absolute())

                # Check for duplicates
                statement = select(RIRFile).where(RIRFile.path == path_str)
                results = session.exec(statement)
                if results.first():
                    continue

                try:
                    room_type = "unknown"
                    config_n = "unknown"
                    rt60 = None

                    if p.suffix == '.pkl':
                        import pickle
                        with open(p, 'rb') as f:
                            data = pickle.load(f)
                        
                        # Extract Metadata from pickle
                        meta = data.get('meta', {})
                        room_config = meta.get('room_config', {})
                        room_type = room_config.get('room_type', 'unknown')
                        
                        # Extract Mic/BCM/Noise Stats
                        mic_config = meta.get('mic_config', {})
                        # relative_positions schema is (3, N)
                        rel_pos = mic_config.get('relative_positions')
                        num_mic = rel_pos.shape[1] if rel_pos is not None else 0
                        num_bcm = 1 if mic_config.get('use_bcm', False) else 0
                        
                        source_info = data.get('source_info', [])
                        num_noise = sum(1 for s in source_info if s['type'] == 'noise')
                        
                        rt60 = meta.get('rt60')
                    
                    else:
                        # Fallback for WAV files (limited info)
                        room_type = "unknown"
                        num_noise = 0
                        num_mic = 4 # Default assumption
                        num_bcm = 1
                        rt60 = None
                        
                        parts = p.stem.split('_')
                        if len(parts) >= 2:
                            room_type = parts[1]
                        for part in parts:
                            if part.startswith("rt"):
                                try:
                                    rt60 = float(part[2:])
                                except:
                                    pass
                            if part.startswith("n") and part[1:].isdigit():
                                num_noise = int(part[1:])
                    
                    if rt60 is not None:
                        rt60 = float(rt60)

                    rir = RIRFile(
                        path=path_str,
                        room_type=room_type,
                        num_noise=num_noise,
                        num_mic=num_mic,
                        num_bcm=num_bcm,
                        rt60=rt60
                    )
                    new_entries.append(rir)
                except Exception as e:
                    print(f"Warning: Failed to process {p}: {e}")

            if new_entries:
                session.add_all(new_entries)
                session.commit()
                print(f"Successfully added {len(new_entries)} RIR files.")
            else:
                print("No new RIR files to add.")
