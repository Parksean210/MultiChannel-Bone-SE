import mlflow
from lightning.pytorch.cli import LightningCLI
# Note: SEModule and SEDataModule are loaded dynamically via YAML class_path

def cli_main():
    # MLflow 시스템 메트릭 로깅 활성화
    mlflow.enable_system_metrics_logging()
    
    # 순수 LightningCLI 사용 (설정은 YAML에서 관리)
    LightningCLI(
        save_config_callback=None,
    )

if __name__ == "__main__":
    cli_main()
