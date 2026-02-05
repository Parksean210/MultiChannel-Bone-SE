from typing import Optional
from sqlmodel import Field, SQLModel, create_engine, Session

class SpeechFile(SQLModel, table=True):
    """음성 데이터(Speech) 메타데이터 모델"""
    id: Optional[int] = Field(default=None, primary_key=True)
    path: str = Field(index=True, unique=True)        # 파일 절대 경로
    dataset_name: str = Field(index=True)             # 데이터셋 명칭 (KsponSpeech 등)
    speaker: Optional[str] = Field(default=None, index=True) # 화자 ID/이름
    language: Optional[str] = Field(default="ko", index=True) # 언어 (ko, en 등)
    duration_sec: float                               # 파일 길이 (초)
    sample_rate: int                                  # 샘플 레이트 (16000 등)
    split: str = Field(default="train", index=True)   # 데이터 분할 (train, val, test)

class NoiseFile(SQLModel, table=True):
    """잡음 데이터(Noise) 메타데이터 모델"""
    id: Optional[int] = Field(default=None, primary_key=True)
    path: str = Field(index=True, unique=True)        # 파일 절대 경로
    dataset_name: str = Field(index=True)             # 데이터셋 명칭 (136-2.극한 소음 등)
    category: str = Field(index=True)                 # 대분류 (urban, living 등)
    sub_category: Optional[str] = Field(default=None, index=True) # 소분류 (traffic 등)
    duration_sec: float                               # 파일 길이 (초)
    sample_rate: int                                  # 샘플 레이트 (16000 등)
    split: str = Field(default="train", index=True)   # 데이터 분할 (train, val, test)

class RIRFile(SQLModel, table=True):
    """방 임헐스 응답(RIR) 메타데이터 모델"""
    id: Optional[int] = Field(default=None, primary_key=True)
    path: str = Field(index=True, unique=True)        # 설정 파일(.pkl) 또는 오디오 경로
    dataset_name: str = Field(default="unknown", index=True) # 데이터셋 명칭
    room_type: str = Field(index=True)                # 방 구조 (shoebox, l_shape 등)
    num_noise: int = Field(default=0, index=True)     # 포함된 잡음 소스 개수
    num_mic: int = Field(default=4, index=True)       # 공기 전도 마이크로폰 개수
    num_bcm: int = Field(default=1, index=True)       # 골전도 센서 게수
    duration_sec: float = Field(default=1.0, index=True) # RIR 길이 (초)
    rt60: Optional[float] = Field(default=None, index=True) # 잔향 시간 (RT60)
    sample_rate: int = Field(default=16000, index=True) # 샘플 레이트
    split: str = Field(default="train", index=True)   # 데이터 분할 (train, val, test)
