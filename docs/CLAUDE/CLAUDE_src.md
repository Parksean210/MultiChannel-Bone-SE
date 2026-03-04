# src/ - Core Source Code

## Overview
프레임워크의 핵심 소스 코드. PyTorch Lightning 기반으로 구성되며, 모델(Pure PyTorch)과 학습 시스템(Lightning)이 명확히 분리되어 있다.

## Module Dependency Graph
```
main.py (LightningCLI)
  -> modules/se_module.py (SEModule: LightningModule)
       -> models/ (BaseSEModel -> ICConvTasNet 등)
       -> modules/losses.py (CompositeLoss)
  -> data/datamodule.py (SEDataModule: LightningDataModule)
       -> data/dataset.py (SpatialMixingDataset)
            -> data/models.py (SQLModel ORM)
            -> db/engine.py (SQLite 엔진)
  -> callbacks/ (GPU 모니터, 오디오 저장)
  -> simulation/ (RIR 생성, 오프라인 전용)
```

## Sub-packages
| Package | Role | Layer |
|---|---|---|
| `models/` | 순수 PyTorch 모델 아키텍처 | Pure PyTorch |
| `modules/` | 학습 루프, 손실함수, GPU 합성 | Lightning |
| `data/` | Dataset, DataModule, ORM 모델 | Lightning + SQLModel |
| `db/` | SQLite 엔진 생성 및 CRUD 관리 | SQLModel |
| `callbacks/` | GPU 모니터링, 추론 결과 저장 | Lightning Callback |
| `simulation/` | pyroomacoustics 기반 RIR 생성 | Offline Utility |

## Key Patterns
- YAML `class_path`를 통해 모델/손실함수를 런타임에 동적 로딩 (LightningCLI)
- 데이터 합성은 DataLoader(CPU)에서 원본 로드 -> SEModule(GPU)에서 FFT Convolution 수행
- 모든 평가지표(SI-SDR, SDR, STOI, PESQ)는 Channel 0 기준으로 계산
