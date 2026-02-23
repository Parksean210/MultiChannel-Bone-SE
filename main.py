import os
import torch
from lightning.pytorch.cli import LightningCLI

# RTX 30xx/40xx Tensor Core 활성화 (성능 향상, 정밀도 미세 손실)
torch.set_float32_matmul_precision('high')

# results/ 디렉토리 자동 생성 (슈퍼컴 등 초기 환경에서 sqlite DB 생성 실패 방지)
os.makedirs("results", exist_ok=True)
# Note: SEModule and SEDataModule are loaded dynamically via YAML class_path

def cli_main():
    # 순수 LightningCLI 사용 (설정은 YAML에서 관리)
    LightningCLI(
        save_config_callback=None,
    )

if __name__ == "__main__":
    cli_main()
