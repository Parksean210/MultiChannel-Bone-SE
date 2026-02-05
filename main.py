import lightning as L
from lightning.pytorch.cli import LightningCLI

def cli_main():
    """
    프로젝트의 메인 진입점(Entry Point).
    LightningCLI를 사용하여 YAML 설정 파일을 기반으로 모델 학습 및 평가를 자동으로 제어합니다.
    """
    cli = LightningCLI(
        save_config_callback=None,
    )

if __name__ == "__main__":
    cli_main()
