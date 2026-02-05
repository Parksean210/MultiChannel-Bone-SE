import torch
import lightning as L
from typing import Optional, Dict, Any
import torchaudio
import torchaudio.functional as F_audio
import torch.nn.functional as F
import os
import tempfile
import soundfile as sf
import matplotlib.pyplot as plt
import io
import torchaudio

class SEModule(L.LightningModule):
    """
    Speech Enhancement 기능을 수행하는 PyTorch Lightning 모듈.
    모델 학습/검증 루프 제어, 최적화(Optimization), 지표 로깅 및 GPU 기반 데이터 합성을 담당합니다.
    """
    def __init__(self, 
                 model: torch.nn.Module, 
                 loss: torch.nn.Module, 
                 optimizer_config: Dict[str, Any] = {"lr": 1e-3, "weight_decay": 0.0},
                 target_type: str = "spatialized", # "spatialized" or "aligned_dry"
                 sample_rate: int = 16000,
                 num_val_samples_to_log: int = 4):
        """
        Args:
            model: 음성 향상을 수행할 신경망 모델
            loss: 손실 함수 (예: CompositeLoss)
            optimizer_config: 학습률(lr) 및 가중치 감쇠(weight_decay) 설정
            target_type: 정답 데이터 종류 ("spatialized": 잔향 포함, "aligned_dry": 잔향 제거)
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'loss'])
        
        self.model = model
        self.loss = loss
        self.optimizer_config = optimizer_config
        self.target_type = target_type
        self.sample_rate = sample_rate
        self.num_val_samples_to_log = num_val_samples_to_log

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
        
        # Target selection (Channel 0 기준)
        if self.target_type == "aligned_dry":
            target = batch['aligned_dry'][:, 0:1, :] # [B, 1, T]
        else:
            target = batch['clean'][:, 0:1, :]      # [B, 1, T]
        
        # 모델 전방향 연산 (Forward Pass)
        est_clean = self(noisy) # [B, M, T]
        
        # 0번 채널에 대해서만 손실 함수 계산 (안경 전면 마이크 기준)
        loss = self.loss(est_clean[:, 0:1, :], target)
        
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
            # 500Hz Cutoff로 Sinc FIR 로우패스 필터 생성 (BCM 특성)
            cutoff = 500.0
            num_taps = 101 # 필터 길이 (홀수 권장)
            t = torch.arange(num_taps, device=device) - (num_taps - 1) / 2
            
            # Sinc 함수 생성
            fc = cutoff / self.sample_rate
            sinc = torch.sinc(2 * fc * t)
            
            # Hanning 윈도우 적용하여 Sidelobe 억제 (버터워스보다 깔끔한 차단 특성)
            window = torch.hann_window(num_taps, periodic=False, device=device)
            kernel = (sinc * window).view(1, 1, -1)
            kernel = kernel / kernel.sum() # Normalize
            
            # 마지막 채널(BCM)에 대해 저대역 통과 필터링 수행
            bcm_speech = speech_mc[:, -1:]
            bcm_speech_padded = F.pad(bcm_speech, (num_taps // 2, num_taps // 2), mode='reflect')
            speech_mc[:, -1:] = F.conv1d(bcm_speech_padded, kernel)[:, :, :T]
            
            bcm_noise = noise_mc_total[:, -1:]
            bcm_noise_padded = F.pad(bcm_noise, (num_taps // 2, num_taps // 2), mode='reflect')
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
        
        # 6. 정렬된 드라이 음성 생성 (Dereverberation 타겟용)
        # Reference Mic (0번)의 RIR Peak를 찾아 원본 음성을 시간 정렬합니다.
        ref_rir = rir_tensor[:, 0, 0, :] # (B, L)
        peak_indices = torch.argmax(torch.abs(ref_rir), dim=-1) # (B,)
        
        aligned_dry = torch.zeros((B, 1, T), device=device)
        for b in range(B):
            peak = peak_indices[b]
            if peak < T:
                aligned_dry[b, 0, peak:] = raw_speech[b, :T-peak]
        
        batch['noisy'] = noisy_mc
        batch['clean'] = speech_mc 
        batch['aligned_dry'] = aligned_dry
        return batch

    def validation_step(self, batch, batch_idx):
        """
        검증 단계 루프. 모델의 성능을 평가하고 주기적으로 오디오 샘플을 로깅합니다.
        """
        batch = self._apply_gpu_synthesis(batch)
        noisy = batch['noisy']
        
        # Target selection (Channel 0 기준)
        if self.target_type == "aligned_dry":
            target = batch['aligned_dry'][:, 0:1, :]
        else:
            target = batch['clean'][:, 0:1, :]
        
        est_clean = self(noisy)
        loss = self.loss(est_clean[:, 0:1, :], target)
        
        # 검증 손실 로깅
        batch_size = noisy.shape[0]
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        # 에폭의 첫 번째 배치에 대해 오디오 샘플을 시각화/청취용으로 로깅
        if batch_idx == 0:
            target_to_log = batch['aligned_dry'] if self.target_type == "aligned_dry" else batch['clean']
            self.log_audio_samples(noisy, target_to_log, est_clean)
            
    def log_audio_samples(self, noisy: torch.Tensor, clean: torch.Tensor, est_clean: torch.Tensor):
        """
        추론 결과를 MLflow/Tensorboard에 오디오 형태로 로깅합니다.
        배치 내의 첫 N개 샘플에 대해 원본, 정답(Clean), 추론 결과를 기록합니다.
        """
        # 로깅할 샘플 수 제한
        num_samples = min(noisy.shape[0], self.num_val_samples_to_log)
        
        # Move to CPU for logging (0번 채널만 로깅)
        noisy = noisy[:num_samples, 0].detach().cpu()
        clean = clean[:num_samples, 0].detach().cpu()
        est_clean = est_clean[:num_samples, 0].detach().cpu()
        
        # If using TensorBoard (default in Lightning)
        if self.logger and hasattr(self.logger.experiment, 'add_audio'):
            for i in range(num_samples):
                self.logger.experiment.add_audio(f'sample_{i}/Noisy_Mic0', noisy[i], self.global_step, sample_rate=self.sample_rate)
                self.logger.experiment.add_audio(f'sample_{i}/Enhanced_Mic0', est_clean[i], self.global_step, sample_rate=self.sample_rate)
                self.logger.experiment.add_audio(f'sample_{i}/Target_Mic0', clean[i], self.global_step, sample_rate=self.sample_rate)
        
        # If using MLflowLogger
        elif self.logger and "MLFlowLogger" in str(type(self.logger)):
            # MLflowLogger doesn't have add_audio, use log_artifact instead
            
            run_id = self.logger.run_id
            for i in range(num_samples):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    # Save audio to temporary file
                    for name, signal in [("Noisy", noisy[i]), ("Enhanced", est_clean[i]), ("Target", clean[i])]:
                        # 1. Save Audio Artifact
                        wav_path = os.path.join(tmp_dir, f"{name}_step{self.global_step}.wav")
                        # Explicitly convert to float32 for soundfile support
                        sf.write(wav_path, signal.to(torch.float32).numpy(), self.sample_rate)
                        self.logger.experiment.log_artifact(run_id, wav_path, artifact_path=f"audio_samples/sample_{i}")

                        # 2. Save Spectrogram Image Artifact
                        plt.figure(figsize=(10, 4))
                        # Use torchaudio for spectrogram calculation
                        n_fft = 512
                        specgram = torchaudio.transforms.Spectrogram(n_fft=n_fft)(signal)
                        spec_db = torchaudio.functional.amplitude_to_DB(specgram, 1.0, 1e-10, 80.0).numpy()
                        
                        # Map bins to Frequency (0 to fs/2)
                        f_max = self.sample_rate / 2000.0 # to kHz
                        plt.imshow(spec_db, aspect='auto', origin='lower', cmap='viridis',
                                   extent=[0, signal.shape[-1]/self.sample_rate, 0, f_max])
                        
                        plt.title(f"{name} Spectrogram (Step {self.global_step})")
                        plt.xlabel("Time (s)")
                        plt.ylabel("Frequency (kHz)")
                        plt.colorbar(format='%+2.0f dB')
                        plt.tight_layout()
                        
                        spec_path = os.path.join(tmp_dir, f"{name}_spec_step{self.global_step}.png")
                        plt.savefig(spec_path)
                        plt.close()
                        self.logger.experiment.log_artifact(run_id, spec_path, artifact_path=f"audio_samples/sample_{i}")

    def configure_optimizers(self):
        """
        최적화 알고리즘 및 학습률 스케줄러(Scheduler)를 설정합니다.
        기본적으로 AdamW 최적화기와 ReduceLROnPlateau 스케줄러를 사용합니다.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.optimizer_config.get('lr', 1e-3),
            weight_decay=self.optimizer_config.get('weight_decay', 0.0)
        )
        
        # 성능 정체 시 학습률을 감소시키는 스케줄러 설정
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
