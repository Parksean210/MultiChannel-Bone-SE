# src/simulation/config.py
from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict, Union, Optional

@dataclass
class MicArrayConfig:
    """AR Glass 마이크 및 골전도 센서 설정"""
    # 기본 공기 전도 마이크 4개 (Air Microphones)
    relative_positions: np.ndarray = field(default_factory=lambda: np.array([
        [-0.07, 0.04, 0.0], [ 0.07, 0.04, 0.0],
        [-0.07, -0.04, 0.0], [ 0.07, -0.04, 0.0]
    ]).T)
    name: str = "Default_4Mic_Glasses"
    
    # 🦴 BCM (골전도) 설정
    use_bcm: bool = True
    bcm_rel_pos: List[float] = field(default_factory=lambda: [-0.07, 0.0, 0.0])
    bcm_cutoff_hz: float = 500.0 
    bcm_noise_attenuation_db: float = 20.0

    # 🌪️ Robustness (강인성)
    perturb_pos_std: Optional[float] = None 
    perturb_gain_std: Optional[float] = None 

@dataclass
class RoomConfig:
    """방 설정 (생성 시점에 결정됨)"""
    room_type: str = "shoebox"
    dimensions: Optional[List[float]] = None # Shoebox용
    corners: Optional[np.ndarray] = None     # Polygon용 (2xN)
    height: float = 3.0
    # 재질 정보
    materials: Dict[str, Union[float, str]] = field(default_factory=dict)
    
    # Simulation Config (Hybrid: ISM + Ray Tracing)
    max_order: int = 7 # Early reflection (ISM) order - 10차까지 상향 조정
    use_ray_tracing: bool = True
    ray_tracing_receiver_radius: float = 0.5
    ray_tracing_n_rays: int = 10000 
    ray_tracing_energy_thres: float = 1e-7