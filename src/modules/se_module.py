import torch
import lightning as L
from typing import Optional, Dict, Any
import torchaudio
import torchaudio.functional as F_audio
import torch.nn.functional as F

class SEModule(L.LightningModule):
    """
    LightningModule for Speech Enhancement.
    Controls the Training/Validation loop, Optimization, and Logging.
    """
    def __init__(self, 
                 model: torch.nn.Module, 
                 loss: torch.nn.Module, 
                 optimizer_config: Dict[str, Any] = {"lr": 1e-3, "weight_decay": 0.0}):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'loss'])
        
        self.model = model
        self.loss = loss
        self.optimizer_config = optimizer_config
        
        # For validation logging limits
        self.num_val_samples_to_log = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass calling the underlying model.
        Input: (B, C, T) -> Output: (B, C, T)
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # 1. Synthesis on GPU
        batch = self._apply_gpu_synthesis(batch)
        
        noisy = batch['noisy']
        clean = batch['clean'] # This is Clean-Spatialized (Reverb) or Aligned-Dry based on choice
                               # For now, let's use the Reverberant Speech as target, 
                               # but often Aligned-Dry is preferred.
        
        # Forward
        est_clean = self(noisy)
        
        # Loss
        loss = self.loss(est_clean, clean)
        
        # Log
        batch_size = noisy.shape[0]
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def _apply_gpu_synthesis(self, batch):
        """
        Performs Spatial Mixing on GPU. 
        Input batch contains: raw_speech, raw_noises, rir_tensor, snr, etc.
        """
        raw_speech = batch['raw_speech']   # (B, T)
        raw_noises = batch['raw_noises']   # (B, S-1, T)
        rir_tensor = batch['rir_tensor']   # (B, M, S, L)
        snr = batch['snr']                 # (B,)
        
        B, M, S, L = rir_tensor.shape
        T = raw_speech.shape[-1]
        device = raw_speech.device

        # 1. Process Speech (Source Index 0)
        # (B, 1, T) * (B, M, L) -> (B, M, T)
        speech_rir = rir_tensor[:, :, 0, :]
        speech_mc = F_audio.fftconvolve(raw_speech.unsqueeze(1), speech_rir, mode="full")
        speech_mc = speech_mc[:, :, :T] # Truncate

        # 2. Process Noises (Source Index 1..S-1)
        # Accumulate noise on GPU
        noise_mc_total = torch.zeros_like(speech_mc)
        num_sources = batch['num_sources'] # List of ints
        
        for k in range(1, S):
            # We only process if k < num_sources for that sample
            # But for simplicity in Batch mode, we can convolve all and mask or 
            # just rely on the fact that padded RIRs are zero.
            # If rir_tensor[:, :, k, :] is zeros, convolution is zero.
            noise_rir = rir_tensor[:, :, k, :]
            noise_wav = raw_noises[:, k-1, :]
            
            noise_spatialized = F_audio.fftconvolve(noise_wav.unsqueeze(1), noise_rir, mode="full")
            noise_mc_total += noise_spatialized[:, :, :T]

        # 3. BCM Modeling on GPU (Last Channel)
        # Note: mic_config is a dict of lists here due to default_collate
        use_bcm = batch['mic_config']['use_bcm'][0] # Assuming same config for batch
        if use_bcm:
             # Use a simple FIR lowpass instead of unstable Biquad IIR on GPU
             cutoff = batch['mic_config']['bcm_cutoff_hz'][0]
             
             # For 500Hz cutoff at 16kHz, a simple moving average or a small sinc kernel works.
             # Here we apply a simple exponential moving average or just skip IIR.
             # Let's use a simple FIR filter by convolving with a boxcar (moving average)
             # box_len = fs // cutoff -> 16000 // 500 = 32 samples
             box_len = int(16000 / cutoff)
             kernel = torch.ones((1, 1, box_len), device=device) / box_len
             
             # Pad and convolve last channel
             bcm_speech = speech_mc[:, -1:]
             bcm_speech_padded = F.pad(bcm_speech, (box_len // 2, box_len // 2), mode='reflect')
             speech_mc[:, -1:] = F.conv1d(bcm_speech_padded, kernel)[:, :, :T]
             
             bcm_noise = noise_mc_total[:, -1:]
             bcm_noise_padded = F.pad(bcm_noise, (box_len // 2, box_len // 2), mode='reflect')
             noise_mc_total[:, -1:] = F.conv1d(bcm_noise_padded, kernel)[:, :, :T]
             
             # Apply noise attenuation to BCM
             atten_db = batch['mic_config']['bcm_noise_attenuation_db'][0]
             atten_factor = 10 ** (-atten_db / 20.0)
             noise_mc_total[:, -1] *= atten_factor

        # 4. SNR Scaling
        # Calculate RMS on air channels (indices 0..M-2 if BCM, else 0..M-1)
        air_idx = slice(0, M-1) if use_bcm else slice(0, M)
        
        # (B, M_air, T) -> (B,)
        clean_rms = torch.sqrt(torch.mean(speech_mc[:, air_idx, :]**2, dim=(1, 2)) + 1e-8)
        noise_rms = torch.sqrt(torch.mean(noise_mc_total[:, air_idx, :]**2, dim=(1, 2)) + 1e-8)
        
        # target_noise_rms = clean_rms / (10**(snr/20))
        # factor = target_noise_rms / noise_rms
        target_factor = (clean_rms / (10**(snr/20))) / (noise_rms + 1e-8)
        
        # Apply factor: (B, 1, 1)
        noise_mc_total *= target_factor.view(B, 1, 1)

        # 5. Final Mix
        noisy_mc = speech_mc + noise_mc_total
        
        batch['noisy'] = noisy_mc
        batch['clean'] = speech_mc # Using reverberant speech as target
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
