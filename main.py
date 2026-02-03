import lightning as L
from lightning.pytorch.cli import LightningCLI

def cli_main():
    # Model과 DataModule을 자동으로 연결해주는 사령관 역할
    # 이 한 줄로 YAML 설정을 읽고 학습(fit), 테스트(test) 등을 수행합니다.
    cli = LightningCLI(
        save_config_callback=None,
    )

if __name__ == "__main__":
    cli_main()
