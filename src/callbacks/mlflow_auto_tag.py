import os
import sys
import subprocess
import mlflow
from lightning.pytorch.callbacks import Callback


class MLflowAutoTagCallback(Callback):
    """
    MLflow run에 실험 메타데이터를 자동으로 태깅하고 config를 아티팩트로 저장합니다.
    YAML callbacks 리스트에 추가하면 별도 코드 수정 없이 동작합니다.

    자동으로 기록하는 정보:
        Tags:
            - git_commit: 현재 커밋 해시 (재현성)
            - git_dirty: 미커밋 변경사항 존재 여부
            - model_type: 모델 클래스명 (ICConvTasNet, ICMamba 등)
            - in_channels: 입력 채널 수
            - target_type: aligned_dry / spatialized
            - sample_rate: 샘플 레이트
        Artifacts:
            - config/: 실행에 사용된 YAML 파일들
    """

    def setup(self, trainer, pl_module, stage):
        if stage != "fit":
            return
        if trainer.global_rank != 0:
            return
        if not hasattr(trainer, 'logger') or not hasattr(trainer.logger, 'run_id'):
            return

        run_id = trainer.logger.run_id
        tracking_uri = getattr(trainer.logger, '_tracking_uri', None) \
                    or getattr(trainer.logger, 'tracking_uri', None)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        with mlflow.start_run(run_id=run_id):
            self._log_git_tags()
            self._log_model_tags(pl_module)
            self._log_config_artifacts()

    def _log_git_tags(self):
        """git commit hash와 dirty 상태를 태그로 기록합니다."""
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            mlflow.set_tag("git_commit", commit)

            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            mlflow.set_tag("git_dirty", "true" if status else "false")
        except Exception:
            mlflow.set_tag("git_commit", "unknown")

    def _log_model_tags(self, pl_module):
        """모델 구조 정보를 태그로 기록합니다."""
        mlflow.set_tag("model_type", type(pl_module.model).__name__)
        mlflow.set_tag("in_channels", str(pl_module.model.in_channels))
        mlflow.set_tag("target_type", pl_module.target_type)
        mlflow.set_tag("sample_rate", str(pl_module.sample_rate))

    def _log_config_artifacts(self):
        """sys.argv에서 --config로 지정된 YAML 파일들을 아티팩트로 저장합니다."""
        args = sys.argv[1:]
        for i, arg in enumerate(args):
            if arg in ("--config", "-c") and i + 1 < len(args):
                path = args[i + 1]
            elif arg.startswith("--config="):
                path = arg.split("=", 1)[1]
            else:
                continue
            if os.path.exists(path):
                mlflow.log_artifact(path, artifact_path="config")
