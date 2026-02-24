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
        client = trainer.logger.experiment  # This is the MlflowClient

        # 전역 mlflow 상태를 건드리지 않고, 이미 초기화된 클라이언트를 직접 사용하여 태깅합니다.
        self._log_git_tags(client, run_id)
        self._log_model_tags(client, run_id, pl_module)
        self._log_config_artifacts(client, run_id)

    def _log_git_tags(self, client, run_id):
        """git commit hash와 dirty 상태를 태그로 기록합니다."""
        try:
            # check_output returns bytes in Python 3, so we need to decode
            commit_bytes = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL
            )
            commit = commit_bytes.decode('utf-8').strip() if isinstance(commit_bytes, bytes) else str(commit_bytes).strip()
            client.set_tag(run_id, "git_commit", commit)

            status_bytes = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL
            )
            status = status_bytes.decode('utf-8').strip() if isinstance(status_bytes, bytes) else str(status_bytes).strip()
            client.set_tag(run_id, "git_dirty", "true" if status else "false")
        except Exception:
            client.set_tag(run_id, "git_commit", "unknown")

    def _log_model_tags(self, client, run_id, pl_module):
        """모델 구조 정보를 태그로 기록합니다."""
        client.set_tag(run_id, "model_type", type(pl_module.model).__name__)
        client.set_tag(run_id, "in_channels", str(pl_module.model.in_channels))
        client.set_tag(run_id, "target_type", pl_module.target_type)
        client.set_tag(run_id, "sample_rate", str(pl_module.sample_rate))

    def _log_config_artifacts(self, client, run_id):
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
                client.log_artifact(run_id, path, artifact_path="config")
