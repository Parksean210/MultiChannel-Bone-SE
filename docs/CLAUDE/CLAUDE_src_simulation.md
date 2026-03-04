# src/simulation/ - Room Impulse Response Simulation

## Overview
pyroomacoustics 기반의 가상 음향 환경 시뮬레이션. 다양한 방 형태/재질에서 RIR을 생성하여 .pkl로 저장한다.
학습 시에는 사용하지 않으며, `scripts/generate_rir_bank.py`를 통해 오프라인으로 사전 생성한다.

## Files
| File | Class | Description |
|---|---|---|
| `config.py` | `MicArrayConfig`, `RoomConfig` | 마이크 배열 및 방 설정 데이터클래스 |
| `generator.py` | `RandomRoomGenerator`, `RIRGenerator` | 랜덤 방 생성 + RIR 계산/저장 |

## MicArrayConfig
- 기본 4채널 공기전도 마이크 (AR 글래스 형태, 좌우 대칭 배치)
- BCM (골전도 센서): 사용 여부, 상대 위치, cutoff 주파수(500Hz), 노이즈 감쇄(20dB)
- 강인성 옵션: 위치/게인 퍼터베이션 표준편차

## RoomConfig
- 방 유형: `shoebox`, `l_shape`, `polygon`
- Hybrid 시뮬레이션: ISM (max_order=7) + Ray Tracing (10000 rays)
- 재질: pyroomacoustics Material 테이블 기반 랜덤 선택

## RIR Generation Pipeline
```
RandomRoomGenerator
  -> generate_random_shoebox/l_shape/polygon()
  -> RoomConfig 반환

RIRGenerator
  -> create_room(config)         # pyroomacoustics Room 생성
  -> add_ar_glasses_randomly()   # 랜덤 위치에 마이크 배열 배치 (20회 재시도)
  -> add_target_source()         # 입 위치 고정 (안경 기준 상대 좌표)
  -> add_noise_sources_randomly()# 1~8개 노이즈 소스 랜덤 배치
  -> generate_and_save()         # RIR 계산 + RT60 측정 + .pkl 저장
```

## RIR .pkl Format
```python
{
    "meta": {fs, rt60, room_config, mic_config, mic_pos, rir_len_sec, rir_gain},
    "source_info": [{pos, type: "target"/"noise"}, ...],
    "rirs": [[rir_array per source] per mic]  # Normalized by global peak
}
```
