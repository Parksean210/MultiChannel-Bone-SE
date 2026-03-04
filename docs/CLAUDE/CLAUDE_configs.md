# configs/ - Experiment Configuration (YAML)

## Overview
LightningCLI가 읽는 YAML 설정 파일. 모델, 손실함수, 데이터, 트레이너를 모두 YAML에서 선언적으로 관리한다.

## Files
| File | Model | Purpose |
|---|---|---|
| `ic_conv_tasnet.yaml` | `ICConvTasNet` | 메인 실험 설정 (5채널, Gradient Checkpointing) |
| `ic_conv_tasnet_bcm_off.yaml` | `ICConvTasNet` | BCM ablation (4채널, BCM 채널 제외) |
| `baseline.yaml` | `DummyModel` | 파이프라인 검증용 (Identity 모델) |
| `ic_mamba2_bcm_guide_v2.yaml` | `ICMamba2BCMGuideV2` | Mamba2 + FiLM 컨디셔닝 실험 (**dev/mamba-integration** 전용) |
| `ic_mamba2_ft.yaml` | `ICMamba2FT` | ICMamba2BCMGuideV2 파인튜닝 (**dev/mamba-integration** 전용) |
| `ic_mamba.yaml` | `ICMamba` | Mamba1 SSM 기반 모델 (**dev/mamba-integration** 전용) |
| `ic_mamba_bcm_off.yaml` | `ICMamba` | Mamba1 BCM ablation (**dev/mamba-integration** 전용) |
| `spatialnet.yaml` | `SpatialNet` | 주파수 도메인 SpatialNet 실험 |
| `spatialnet_fsq_split.yaml` | `SpatialNetFSQSplit` | FSQ Split Computing 실험 (96kbps) |

## YAML Structure
```yaml
seed_everything: 42          # 재현성 시드

data:
  class_path: src.data.SEDataModule
  init_args: {db_path, batch_size, num_workers, target_sr}

model:
  class_path: src.modules.se_module.SEModule
  init_args:
    model:                   # Pure PyTorch 모델 (class_path로 동적 로딩)
      class_path: src.models.ICConvTasNet
      init_args: {...}
    loss:                    # 손실함수 (class_path로 동적 로딩)
      class_path: src.modules.losses.CompositeLoss
      init_args: {alpha: 0.1}
    optimizer_config: {lr, weight_decay}
    scheduler_config: {warmup_epochs, min_lr}

trainer:
  default_root_dir: "results"
  max_epochs, accelerator, devices, strategy, precision, gradient_clip_val
  logger: MLFlowLogger
  callbacks: [ModelCheckpoint, GPUStatsMonitor, AudioPredictionWriter, EarlyStopping]
```

## CLI Override Examples
```bash
# 배치 사이즈 변경
--data.init_args.batch_size 8

# BCM 제외 (4채널)
--model.model.init_args.in_channels 4

# 실험 이름 변경
--trainer.logger.init_args.run_name "Experiment_A"

# 체크포인트에서 재개
--ckpt_path path/to/checkpoint.ckpt
```

## Notes
- `val_check_interval: 7857`: 스텝 단위 검증 주기 (대규모 데이터셋에서 에폭 내 검증 수행)
- **strategy 선택 기준**: `use_checkpoint=True`인 Conv-TasNet 계열은 `"ddp_find_unused_parameters_true"` 필수 (gradient checkpointing이 unused params 감지를 유발). Mamba 계열은 반드시 `"ddp"` 사용 (`find_unused_parameters=True`와 Mamba 충돌)
- `precision: 16-mixed`: Mixed Precision으로 메모리/속도 최적화
- Mamba 계열 config는 `mamba-ssm`, `causal-conv1d` 별도 설치 후 사용 가능
