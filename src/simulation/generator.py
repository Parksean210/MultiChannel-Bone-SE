# src/simulation/generator.py
import pyroomacoustics as pra
import random
import numpy as np
import pickle
import os
from dataclasses import asdict
from typing import List, Tuple
from .config import RoomConfig, MicArrayConfig

class RandomRoomGenerator:
    def __init__(self):
        # 다양한 재질 풀 (pyroomacoustics 테이블에서 가져옴)
        self.floor_materials = [
            "carpet_cotton", "carpet_tufted_9.5mm", "carpet_thin", "carpet_hairy",
            "linoleum_on_concrete", "concrete_floor", "marble_floor", 
            "stage_floor", "audience_floor"
        ]
        self.wall_materials = [
            # 흡음 낮음 (반사 높음)
            "brickwork", "rough_concrete", "smooth_brickwork_flush_pointing",
            "ceramic_tiles", "limestone_wall", "glass_window", "glass_3mm",
            # 흡음 중간
            "gypsum_board", "plywood_thin", "wooden_lining", "wood_16mm",
            "plasterboard", "facing_brick",
            # 흡음 높음
            "curtains_cotton_0.5", "curtains_velvet", "curtains_fabric_folded",
            "rockwool_50mm_80kgm3", "mineral_wool_50mm_70kgm3",
            # 숫자로 직접 지정 (흡음계수)
            0.1, 0.15, 0.2, 0.25, 0.3
        ]
        self.ceiling_materials = [
            "plasterboard", "hard_surface", "ceiling_plasterboard",
            "ceiling_fibre_absorber", "ceiling_fissured_tile",
            "ceiling_perforated_gypsum_board", "ceiling_melamine_foam",
            0.1, 0.2, 0.3, 0.4
        ]
    
    def _get_random_materials(self, is_shoebox=True):
        """바닥, 천장, 벽면 재질을 각각 랜덤하게 선택"""
        floor = random.choice(self.floor_materials)
        ceiling = random.choice(self.ceiling_materials)
        
        if is_shoebox:
            # Shoebox는 동서남북 벽을 따로 설정 가능
            walls = [random.choice(self.wall_materials) for _ in range(4)]
            return {
                "floor": floor, "ceiling": ceiling,
                "east": walls[0], "west": walls[1], "north": walls[2], "south": walls[3]
            }
        else:
            # Polygon은 벽면 전체 통일 (구현 편의상)
            wall = random.choice(self.wall_materials)
            return {"floor": floor, "ceiling": ceiling, "walls": wall}
    
    def generate_random_shoebox(self):
        dims = [random.uniform(3, 8), random.uniform(3, 8), random.uniform(2.4, 4.0)]
        return RoomConfig("shoebox", dimensions=dims, height=dims[2], materials=self._get_random_materials(True))

    def generate_random_l_shape(self):
        Lx, Ly = random.uniform(4, 9), random.uniform(4, 9)
        min_thick = 1.5
        Wx, Wy = random.uniform(min_thick, Lx-min_thick), random.uniform(min_thick, Ly-min_thick)
        corners = np.array([[0, 0], [Lx, 0], [Lx, Wy], [Wx, Wy], [Wx, Ly], [0, Ly]]).T
        return RoomConfig("l_shape", corners=corners, height=random.uniform(2.4, 4.0), materials=self._get_random_materials(False))

    def generate_random_polygon(self, n_vertices_range=(5, 7)):
        n_vertices = random.randint(*n_vertices_range)
        angles = np.sort(np.random.uniform(0, 2 * np.pi, n_vertices))
        radii = np.random.uniform(2.0, 7.0, n_vertices)
        corners = np.vstack((radii * np.cos(angles), radii * np.sin(angles)))
        corners[0] -= np.min(corners[0]) # 양수 좌표 보정
        corners[1] -= np.min(corners[1])
        return RoomConfig("polygon", corners=corners, height=random.uniform(2.4, 4.0), materials=self._get_random_materials(False))

class RIRGenerator:
    def __init__(self, fs: int = 16000, rir_len_sec: float = 1.0):
        self.fs = fs
        self.rir_len_sec = rir_len_sec
        self.room = None
        self.room_config = None
        self.mic_config = None
        self.mic_pos = None  # 마이크 절대 위치 저장용
        self.source_info = []

    def _parse_materials(self, mat_config):
        return {k: pra.Material(v) if isinstance(v, str) else pra.Material(float(v)) for k, v in mat_config.items()}

    def create_room(self, config: RoomConfig):
        # 이전 상태 초기화 (데이터 누적 방지)
        self.source_info = []
        self.mic_pos = None
        
        self.room_config = config
        mats = self._parse_materials(config.materials)
        
        if config.room_type == "shoebox":
            self.room = pra.ShoeBox(config.dimensions, fs=self.fs, materials=mats, max_order=config.max_order)
        else:
            # Polygon: 2D 벽 생성 -> 3D 확장 (Extrude)
            # Extrude 시 materials={"ceiling": ..., "floor": ...} 형태로 전달
            self.room = pra.Room.from_corners(config.corners, fs=self.fs, materials=mats["walls"], max_order=config.max_order)
            self.room.extrude(config.height, materials={"ceiling": mats["ceiling"], "floor": mats["floor"]})
        
        # Ray Tracing 설정 (Hybrid Simulation)
        if hasattr(config, 'use_ray_tracing') and config.use_ray_tracing:
            self.room.set_ray_tracing(
                receiver_radius=config.ray_tracing_receiver_radius,
                n_rays=config.ray_tracing_n_rays,
                energy_thres=config.ray_tracing_energy_thres
            )

    def _get_random_safe_point(self, height_range=(0.5, 2.0), margin=0.3):
        """방 내부의 안전한 랜덤 좌표 (x, y, z) 반환 (Rejection Sampling)"""
        if self.room is None: raise ValueError("Room not created")
        
        # Bounding Box 계산
        if self.room_config.room_type == 'shoebox':
            dims = self.room_config.dimensions
            x_min, x_max = 0, dims[0]
            y_min, y_max = 0, dims[1]
            z_ceil = dims[2]
        else:
            corners = self.room_config.corners
            x_min, x_max = np.min(corners[0]), np.max(corners[0])
            y_min, y_max = np.min(corners[1]), np.max(corners[1])
            z_ceil = self.room_config.height
            
        z_min = max(0.1, height_range[0])
        z_max = min(z_ceil - 0.1, height_range[1])
        
        for _ in range(100): # 최대 100번 시도
            x = random.uniform(x_min + margin, x_max - margin)
            y = random.uniform(y_min + margin, y_max - margin)
            z = random.uniform(z_min, z_max)
            
            if self.room.is_inside([x, y, z]):
                return [x, y, z]
        
        raise ValueError("Failed to find a safe point inside the room.")

    def add_ar_glasses_randomly(self, config: MicArrayConfig):
        """AR 글래스를 방 안 랜덤한 위치/높이에 배치 (재시도 로직 포함)"""
        self.mic_config = config
        
        for _ in range(20): # 위치 선정 최대 20회 재시도
            try:
                # 1. 안전한 랜덤 위치 (높이 1.0 ~ 1.8m)
                center_pos = self._get_random_safe_point(height_range=(1.0, 1.8), margin=0.5)
                rotation_deg = random.uniform(0, 360)
                
                # 2. 회전 및 배치
                theta = np.radians(rotation_deg)
                R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
                
                air_mics_abs = R @ config.relative_positions + np.array(center_pos)[:, None]
                final_mics_pos = air_mics_abs
                
                if config.use_bcm:
                    bcm_abs = R @ np.array(config.bcm_rel_pos).reshape(3, 1) + np.array(center_pos)[:, None]
                    final_mics_pos = np.hstack([air_mics_abs, bcm_abs])

                # 위치 퍼터베이션 (강인성)
                if config.perturb_pos_std and config.perturb_pos_std > 0:
                    final_mics_pos += np.random.normal(0, config.perturb_pos_std, size=final_mics_pos.shape)
                
                # 3. 각 마이크 위치가 방 안에 있는지 체크
                is_valid = True
                for mic_idx in range(final_mics_pos.shape[1]):
                    mic_pos = final_mics_pos[:, mic_idx]
                    if not self.room.is_inside(mic_pos):
                        is_valid = False
                        break
                
                if is_valid:
                    self.room.add_microphone_array(final_mics_pos)
                    self.mic_pos = final_mics_pos # (3, num_mics)
                    return center_pos, R
            
            except ValueError:
                continue # _get_random_safe_point 실패 시 재시도
        
        raise ValueError("Failed to place AR glasses safely inside the room after 20 attempts.")

    def add_target_source(self, glasses_center, rotation_matrix, offset=(0.1, 0.0, -0.05)):
        """Target(입) 배치: 안경 기준 상대 좌표로 고정"""
        offset_vec = np.array(offset).reshape(3, 1)
        source_pos = (rotation_matrix @ offset_vec).flatten() + np.array(glasses_center)
        
        if not self.room.is_inside(source_pos.reshape(3, 1)):
             raise ValueError("Target source placed outside.")
             
        self.room.add_source(source_pos)
        self.source_info.append({"pos": source_pos, "type": "target"})

    def add_noise_sources_randomly(self, num_sources_range=(1, 3)):
        """노이즈 소스를 방 안 랜덤 위치에 N개 배치"""
        num_sources = random.randint(*num_sources_range)
        for i in range(num_sources):
            # 바닥~천장 전체 범위
            pos = self._get_random_safe_point(height_range=(0.2, self.room_config.height - 0.2), margin=0.3)
            self.room.add_source(pos)
            self.source_info.append({"pos": pos, "type": "noise"})
        return num_sources

    def generate_and_save(self, filename: str):
        self.room.compute_rir()
        # RT60 측정 (측정값 우선, 실패 시 이론값 fallback)
        rt60_measured, rt60_theory = None, None
        try:
            rt60_measured = self.room.measure_rt60(mic=0, source=0)
        except Exception:
            pass
        try:
            rt60_theory = self.room.rt60_theory()
        except Exception:
            pass
        rt60 = rt60_measured if rt60_measured is not None else rt60_theory

        # RIR Truncation: 사용자가 지정한 길이로 제한 (기본 1.0초)
        max_rir_len = int(self.rir_len_sec * self.fs)
        rirs = [[rir[:max_rir_len] for rir in mic_rirs] for mic_rirs in self.room.rir]

        # RIR Normalization: 전체 채널/소스 중 global peak로 normalize
        global_peak = 0.0
        for mic_rirs in rirs:
            for rir in mic_rirs:
                if len(rir) == 0: continue
                peak = np.max(np.abs(rir))
                if peak > global_peak:
                    global_peak = peak
        
        if global_peak > 0:
            normalized_rirs = [[rir / global_peak for rir in mic_rirs] for mic_rirs in rirs]
        else:
            normalized_rirs = rirs

        data = {
            "meta": {
                "fs": self.fs, 
                "rt60": rt60,  # RT60 추가
                "room_config": asdict(self.room_config), 
                "mic_config": asdict(self.mic_config),
                "mic_pos": self.mic_pos, # 검증용 절대 위치
                "rir_gain": global_peak,  # 원본 복원용 gain 저장
            },
            "source_info": self.source_info,
            "rirs": normalized_rirs
        }
        with open(filename, 'wb') as f: pickle.dump(data, f)