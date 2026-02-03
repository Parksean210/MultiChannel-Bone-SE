from typing import Optional
from sqlmodel import Field, SQLModel, create_engine, Session

class SpeechFile(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    path: str = Field(index=True, unique=True)
    dataset_name: str = Field(index=True)
    speaker_id: Optional[str] = Field(default=None, index=True)
    duration_sec: float
    is_eval: bool = Field(default=False, index=True)

class NoiseFile(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    path: str = Field(index=True, unique=True)
    category: str = Field(index=True)  # urban, living, etc.
    sub_category: Optional[str] = Field(default=None, index=True) # traffic, construction, etc.
    duration_sec: float

class RIRFile(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    path: str = Field(index=True, unique=True)
    room_type: str = Field(index=True)     # shoebox, l_shape, polygon
    num_noise: int = Field(default=0, index=True) # 노이즈 개수
    num_mic: int = Field(default=4, index=True)   # 마이크로폰 개수 (Air)
    num_bcm: int = Field(default=1, index=True)   # 골전도 센서 개수
    rt60: Optional[float] = Field(default=None, index=True) # Reverberation time

