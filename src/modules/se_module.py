import torch
import lightning as L
from typing import Optional, Dict, Any
import torchaudio
import torchaudio.functional as F_audio
import torch.nn.functional as F

class SEModule(L.LightningModule):
    """
    Speech Enhancement 기능을 수행하는 PyTorch Lightning 모듈.
    모델 학습/검증 루프 제어, 최적화(Optimization), 지표 로깅 및 GPU 기반 데이터 합성을 담당합니다.
    """
    def __init__(self, 
                 model: torch.nn.Module, 
                 loss: torch.nn.Module, 
                 optimizer_config: Dict[str, Any] = {"lr": 1e-3, "weight_decay": 0.0}):
        """
        Args:
            model: 음성 향상을 수행할 신경망 모델
            loss: 손실 함수 (예: CompositeLoss)
            optimizer_config: 학습률(lr) 및 가중치 감쇠(weight_decay) 설정
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'loss'])
        
        self.model = model
        self.loss = loss
        self.optimizer_config = optimizer_config
        
        # 검증(Validation) 시 MLflow/Tensorboard에 로깅할 샘플 수 제한
        self.num_val_samples_to_log = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass calling the underlying model.
        Input: (B, C, T) -> Output: (B, C, T)
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        학습 단계 루프. GPU 기반 실시간 합성을 수행한 후 전방향 연산 및 손실을 계산합니다.
        """
        # 1. GPU 기반 실시간 공간 합성 (Data Augmentation 효과)
        batch = self._apply_gpu_synthesis(batch)
        
        noisy = batch['noisy']
        # Target: 기본적으로 잔향이 포함된 스피치(Reverberant Speech)를 사용
        clean = batch['clean'] 
        
        # 모델 전방향 연산 (Forward Pass)
        est_clean = self(noisy)
        
        # 손실 함수 계산
        loss = self.loss(est_clean, clean)
        
        # 지표 로깅
        batch_size = noisy.shape[0]
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def _apply_gpu_synthesis(self, batch: Dict):
        """
        DataLoader에서 전달받은 원본 데이터를 바탕으로 GPU 상에서 공간 합성을 수행합니다.
        FFT Convolution을 통해 대규모 배치를 고속으로 처리합니다.
        """
        raw_speech = batch['raw_speech']   # Shape: (B, T)
        raw_noises = batch['raw_noises']   # Shape: (B, S-1, T)
        rir_tensor = batch['rir_tensor']   # Shape: (B, M, S, L)
        snr = batch['snr']                 # Shape: (B,)
        
        B, M, S, L = rir_tensor.shape
        T = raw_speech.shape[-1]
        device = raw_speech.device

        # 1. 스피치 소스 공간화 (Source Index 0)
        # (Batch, 1, Time) * (Batch, Mic, RIR_Len) -> (Batch, Mic, Time)
        speech_rir = rir_tensor[:, :, 0, :]
        speech_mc = F_audio.fftconvolve(raw_speech.unsqueeze(1), speech_rir, mode="full")
        speech_mc = speech_mc[:, :, :T] # 원본 길이로 절삭

        # 2. 노이즈 소스 공간화 및 누적 (Source Index 1..S-1)
        noise_mc_total = torch.zeros_like(speech_mc)
        for k in range(1, S):
            noise_rir = rir_tensor[:, :, k, :]
            noise_wav = raw_noises[:, k-1, :]
            
            # 노이즈별 RIR을 적용하여 다채널 잡음 생성
            noise_spatialized = F_audio.fftconvolve(noise_wav.unsqueeze(1), noise_rir, mode="full")
            noise_mc_total += noise_spatialized[:, :, :T]

        # 3. 골전도 센서(BCM) 모델링 (마지막 마이크 채널 대상)
        use_bcm = batch['mic_config']['use_bcm'][0]
        if use_bcm:
             # GPU 연산 최적화를 위해 무거운 IIR 대신 단순 FIR 필터 적용
             cutoff = batch['mic_config']['bcm_cutoff_hz'][0]
             box_len = int(16000 / cutoff)
             kernel = torch.ones((1, 1, box_len), device=device) / box_len
             
             # 마지막 채널(BCM)에 대해 저대역 통과 필터링 수행
             bcm_speech = speech_mc[:, -1:]
             bcm_speech_padded = F.pad(bcm_speech, (box_len // 2, box_len // 2), mode='reflect')
             speech_mc[:, -1:] = F.conv1d(bcm_speech_padded, kernel)[:, :, :T]
             
             bcm_noise = noise_mc_total[:, -1:]
             bcm_noise_padded = F.pad(bcm_noise, (box_len // 2, box_len // 2), mode='reflect')
             noise_mc_total[:, -1:] = F.conv1d(bcm_noise_padded, kernel)[:, :, :T]
             
             # BCM 센서의 잡음 감쇄 특성 반영
             atten_db = batch['mic_config']['bcm_noise_attenuation_db'][0]
             atten_factor = 10 ** (-atten_db / 20.0)
             noise_mc_total[:, -1] *= atten_factor

        # 4. SNR 스케일링 (에어 채널의 RMS 기준)
        air_idx = slice(0, M-1) if use_bcm else slice(0, M)
        clean_rms = torch.sqrt(torch.mean(speech_mc[:, air_idx, :]**2, dim=(1, 2)) + 1e-8)
        noise_rms = torch.sqrt(torch.mean(noise_mc_total[:, air_idx, :]**2, dim=(1, 2)) + 1e-8)
        
        # 목표 SNR 달성을 위한 노이즈 가중치 산출
        target_factor = (clean_rms / (10**(snr/20))) / (noise_rms + 1e-8)
        noise_mc_total *= target_factor.view(B, 1, 1)

        # 5. 최종 혼합 (Clean + Scaled Noise)
        noisy_mc = speech_mc + noise_mc_total
        
        batch['noisy'] = noisy_mc
        batch['clean'] = speech_mc 
        return batch

    def validation_step(self, batch, batch_idx):
        batch = self._apply_gpu_synthesis(batch)
        noisy = batch['noisy']
        clean = batch['clean']
        
        est_clean = self(noisy)
        loss = self.loss(est_clean, clean)
        
        # Log validation loss
        batch_size = noisy.shape[0]
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        # Log Audio Samples (Only for the first batch of the epoch)
        if batch_idx == 0:
            self.log_audio_samples(noisy, clean, est_clean)
            
    def log_audio_samples(self, noisy, clean, est_clean):
        """
        Log audio examples to MLflow/Tensorboard.
        Logs the first N samples from the batch.
        """
        # Ensure we don't log too many
        num_samples = min(noisy.shape[0], self.num_val_samples_to_log)
        
        # Get Logger (MLflow or Tensorboard)
        # Note: Implementation depends on the logger type. 
        # Here we assume MLflow is mostly used via artifacts or standard Lightning logger integration.
        
        # Move to CPU for logging
        noisy = noisy[:num_samples].detach().cpu()
        clean = clean[:num_samples].detach().cpu()
        est_clean = est_clean[:num_samples].detach().cpu()
        
        # If using TensorBoard (default in Lightning)
        if self.logger and hasattr(self.logger.experiment, 'add_audio'):
            for i in range(num_samples):
                self.logger.experiment.add_audio(f'sample_{i}/Noisy', noisy[i], self.global_step, sample_rate=16000)
                self.logger.experiment.add_audio(f'sample_{i}/Enhanced', est_clean[i], self.global_step, sample_rate=16000)
                self.logger.experiment.add_audio(f'sample_{i}/Clean', clean[i], self.global_step, sample_rate=16000)
        
        # If using MLflowLogger
        elif self.logger and "MLFlowLogger" in str(type(self.logger)):
             # Lightning's MLflowLogger doesn't have direct add_audio.
             # We might need custom artifact logging, but for now we skip complex workaround 
             # and rely on the UI looking for saved files if we implemented saving to disk.
             # For simplicity, we assume Tensorboard or just skip if not available.
             pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.optimizer_config.get('lr', 1e-3),
            weight_decay=self.optimizer_config.get('weight_decay', 0.0)
        )
        
        # Optional: Add Scheduler (ReduceLROnPlateau is common for SE)
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
