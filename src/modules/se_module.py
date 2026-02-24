import torch
import lightning as L
import mlflow
from typing import Optional, Dict, Any
import os
import tempfile

from src.utils.synthesis import create_bcm_kernel, apply_spatial_synthesis
from src.utils.metrics import create_metric_suite, compute_and_log_metrics
from src.utils.audio_io import (
    prepare_audio_for_saving,
    build_metadata_filename,
    create_spectrogram_image,
    save_audio_file,
)


class SEModule(L.LightningModule):
    """
    Speech Enhancement 기능을 수행하는 PyTorch Lightning 모듈.
    모델 학습/검증 루프 제어, 최적화(Optimization), 지표 로깅 및 GPU 기반 데이터 합성을 담당합니다.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 loss: torch.nn.Module,
                 optimizer_config: Dict[str, Any] = {"lr": 1e-3, "weight_decay": 0.0},
                 target_type: str = "spatialized",
                 sample_rate: int = 16000,
                 num_val_samples_to_log: int = 4):
        """
        Args:
            model: 음성 향상을 수행할 신경망 모델 (BaseSEModel 자식 클래스)
            loss: 손실 함수 (예: CompositeLoss)
            optimizer_config: 학습률(lr) 및 가중치 감쇠(weight_decay) 설정
            target_type: 정답 데이터 종류 ("spatialized": 잔향 포함, "aligned_dry": 잔향 제거)
            sample_rate: 오디오 샘플 레이트
            num_val_samples_to_log: 검증 시 로깅할 오디오 샘플 수
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'loss'])

        self.model = model
        self.loss = loss
        self.optimizer_config = optimizer_config
        self.target_type = target_type
        self.sample_rate = sample_rate
        self.num_val_samples_to_log = num_val_samples_to_log

        # BCM 커널 캐싱 (매 스텝 재생성 방지)
        bcm_kernel = create_bcm_kernel(cutoff_hz=500.0, sample_rate=sample_rate, num_taps=101)
        self.register_buffer('bcm_kernel', bcm_kernel)

        # Metrics (torchmetrics 객체 - DDP 분산 집계 지원)
        metrics = create_metric_suite(sample_rate)
        self.si_sdr = metrics["si_sdr"]
        self.sdr = metrics["sdr"]
        self.stoi = metrics["stoi"]
        self.pesq = metrics["pesq"]

    def on_save_checkpoint(self, checkpoint: Dict) -> None:
        """체크포인트에 모델 클래스명과 init args를 저장하여 config 없이 복원 가능하게 합니다."""
        import inspect
        checkpoint["model_class_name"] = type(self.model).__name__
        sig = inspect.signature(type(self.model).__init__)
        checkpoint["model_init_args"] = {
            k: getattr(self.model, k)
            for k in sig.parameters
            if k != 'self' and hasattr(self.model, k)
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. 모델의 in_channels에 맞게 자동 슬라이싱합니다.
        Input: (B, C, T) -> Output: (B, C, T)
        """
        return self.model(x[:, :self.model.in_channels, :].contiguous())

    def _select_target(self, batch: Dict) -> torch.Tensor:
        """배치에서 타겟 텐서를 선택합니다 (Channel 0 기준)."""
        if self.target_type == "aligned_dry":
            return batch['aligned_dry'][:, 0:1, :]
        return batch['clean'][:, 0:1, :]

    def _apply_gpu_synthesis(self, batch: Dict) -> Dict:
        """GPU 기반 실시간 공간 합성. 캐싱된 BCM 커널을 사용합니다."""
        return apply_spatial_synthesis(batch, bcm_kernel=self.bcm_kernel, sample_rate=self.sample_rate)

    def training_step(self, batch, batch_idx):
        """학습 단계 루프."""
        batch = self._apply_gpu_synthesis(batch)
        target = self._select_target(batch)
        est_clean = self(batch['noisy'])

        loss = self.loss(est_clean[:, 0:1, :], target)

        batch_size = batch['noisy'].shape[0]
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        """검증 단계 루프."""
        batch = self._apply_gpu_synthesis(batch)
        target = self._select_target(batch)
        est_clean = self(batch['noisy'])
        est_ch0 = est_clean[:, 0:1, :]

        loss = self.loss(est_ch0, target)
        batch_size = batch['noisy'].shape[0]
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        # 메트릭 일괄 계산 및 로깅
        metric_suite = {"si_sdr": self.si_sdr, "sdr": self.sdr, "stoi": self.stoi, "pesq": self.pesq}
        compute_and_log_metrics(self, metric_suite, est_ch0, target, prefix="val",
                                batch_size=batch_size, sync_dist=True)

        # 에폭의 첫 번째 배치에 대해 오디오 샘플 로깅
        if batch_idx == 0:
            target_to_log = batch['aligned_dry'] if self.target_type == "aligned_dry" else batch['clean']
            self.log_audio_samples(batch['noisy'], target_to_log, est_clean, batch=batch)

    def test_step(self, batch, batch_idx):
        """테스트 단계 루프."""
        batch = self._apply_gpu_synthesis(batch)
        target = batch['aligned_dry'][:, 0:1, :]
        est_clean = self(batch['noisy'])
        est_ch0 = est_clean[:, 0:1, :]

        batch_size = batch['noisy'].shape[0]
        metric_suite = {"si_sdr": self.si_sdr, "sdr": self.sdr, "stoi": self.stoi, "pesq": self.pesq}
        results = compute_and_log_metrics(self, metric_suite, est_ch0, target, prefix="test",
                                          batch_size=batch_size, sync_dist=False)
        return {f"test_{k}": v for k, v in results.items()}

    def on_test_epoch_end(self):
        """테스트 완료 후 MLflow에 최종 집계 메트릭을 summary로 기록합니다."""
        if not self.trainer.is_global_zero:
            return
        if not (self.logger and "MLFlowLogger" in str(type(self.logger))):
            return

        tracking_uri = getattr(self.logger, '_tracking_uri', None) \
                    or getattr(self.logger, 'tracking_uri', None)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        with mlflow.start_run(run_id=self.logger.run_id):
            for name, metric_fn in [
                ("si_sdr", self.si_sdr),
                ("sdr", self.sdr),
                ("stoi", self.stoi),
                ("pesq", self.pesq),
            ]:
                try:
                    val = metric_fn.compute()
                    mlflow.log_metric(f"test_{name}_final", val.item())
                except Exception:
                    pass

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """추론 단계 루프."""
        batch = self._apply_gpu_synthesis(batch)
        est_clean = self(batch['noisy'])

        if self.target_type == "aligned_dry":
            target = batch['aligned_dry']
        else:
            target = batch['clean']

        return {
            "noisy": batch['noisy'],
            "enhanced": est_clean,
            "target": target,
            "speech_id": batch.get("speech_id"),
            "noise_ids": batch.get("noise_ids"),
            "rir_id": batch.get("rir_id"),
            "snr": batch.get("snr"),
        }

    def log_audio_samples(self, noisy: torch.Tensor, clean: torch.Tensor,
                          est_clean: torch.Tensor, batch: Optional[Dict] = None):
        """추론 결과를 MLflow/Tensorboard에 오디오 형태로 로깅합니다."""
        num_samples = min(noisy.shape[0], self.num_val_samples_to_log)

        # Channel 0, CPU 전송
        noisy_cpu = noisy[:num_samples, 0].detach().cpu()
        clean_cpu = clean[:num_samples, 0].detach().cpu()
        est_cpu = est_clean[:num_samples, 0].detach().cpu()

        # TensorBoard
        if self.logger and hasattr(self.logger.experiment, 'add_audio'):
            for i in range(num_samples):
                self.logger.experiment.add_audio(f'sample_{i}/Noisy_Mic0', noisy_cpu[i], self.global_step, sample_rate=self.sample_rate)
                self.logger.experiment.add_audio(f'sample_{i}/Enhanced_Mic0', est_cpu[i], self.global_step, sample_rate=self.sample_rate)
                self.logger.experiment.add_audio(f'sample_{i}/Target_Mic0', clean_cpu[i], self.global_step, sample_rate=self.sample_rate)

        # MLflow
        elif self.logger and "MLFlowLogger" in str(type(self.logger)) and self.trainer.is_global_zero:
            run_id = self.logger.run_id
            for i in range(num_samples):
                # 메타데이터 파일명 생성
                if batch is not None and "speech_id" in batch:
                    base_name = build_metadata_filename(
                        speech_id=batch["speech_id"][i],
                        noise_ids=batch["noise_ids"][i],
                        rir_id=batch["rir_id"][i],
                        snr=batch["snr"][i],
                    )
                else:
                    base_name = f"sample_{i}"

                with tempfile.TemporaryDirectory() as tmp_dir:
                    for name, signal in [("Noisy", noisy_cpu[i]), ("Enhanced", est_cpu[i]), ("Target", clean_cpu[i])]:
                        # WAV 저장
                        wav_name = f"{name}_{base_name}_step{self.global_step}.wav"
                        wav_path = os.path.join(tmp_dir, wav_name)
                        save_audio_file(wav_path, signal.unsqueeze(0), self.sample_rate, channel=0)
                        self.logger.experiment.log_artifact(run_id, wav_path, artifact_path=f"audio_samples/sample_{i}")

                        # 스펙트로그램 저장
                        spec_name = f"{name}_spec_{base_name}_step{self.global_step}.png"
                        spec_path = os.path.join(tmp_dir, spec_name)
                        create_spectrogram_image(
                            signal, self.sample_rate, n_fft=512,
                            title=f"{name} Spectrogram ({base_name}, Step {self.global_step})",
                            save_path=spec_path,
                        )
                        self.logger.experiment.log_artifact(run_id, spec_path, artifact_path=f"audio_samples/sample_{i}")

    def configure_optimizers(self):
        """AdamW + ReduceLROnPlateau 설정."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_config.get('lr', 1e-3),
            weight_decay=self.optimizer_config.get('weight_decay', 0.0)
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
