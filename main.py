import lightning as L
from lightning.pytorch.cli import LightningCLI
import mlflow

def cli_main():
    """
    프로젝트의 메인 진입점.
    """
    # MLFlow 전용 System Metrics 탭 활성화
    # 1. 트래킹 서버 경로 설정 (YAML과 일치하게)
    mlflow.set_tracking_uri("file:./mlruns")
    # 2. 시스템 메트릭 기능을 전역으로 켬
    mlflow.enable_system_metrics_logging()
    
    cli = LightningCLI(
        save_config_callback=None,
    )

if __name__ == "__main__":
    cli_main()
