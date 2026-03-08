import os

import torch
from lightning.pytorch.cli import LightningCLI

# RTX 30xx/40xx Tensor Core 활성화 (성능 향상, 정밀도 미세 손실)
torch.set_float32_matmul_precision('high')

# WSL2: synchronous CUDA execution to prevent async error contamination in Triton kernels.
# Safe to leave on; negligible overhead compared to Mamba2 computation.
# On real Linux (supercomputer), Triton works fine without this.
if os.environ.get('CUDA_LAUNCH_BLOCKING') is None:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# results/ 디렉토리 자동 생성 (슈퍼컴 등 초기 환경에서 sqlite DB 생성 실패 방지)
os.makedirs("results", exist_ok=True)
# Note: SEModule and SEDataModule are loaded dynamically via YAML class_path


# WSL2 Triton autotuning patch: catch CUDA errors during kernel benchmarking
# Triton's do_bench() fails on WSL2 with 'CUDA error: unknown error' during sync.
# This patch skips failed configs (returns inf) instead of crashing.
try:
    import triton.runtime.autotuner as _triton_at
    _orig_bench = _triton_at.Autotuner._bench
    def _safe_bench(self, *args, **kwargs):
        try:
            return _orig_bench(self, *args, **kwargs)
        except Exception:
            return float('inf')
    _triton_at.Autotuner._bench = _safe_bench
except Exception:
    pass  # triton not installed, ignore

def cli_main():
    # 순수 LightningCLI 사용 (설정은 YAML에서 관리)
    LightningCLI(
        save_config_callback=None,
    )

if __name__ == "__main__":
    cli_main()
