import os
import random
import concurrent.futures
import torch
import torch.nn.functional as F
import numpy as np
from pesq import pesq
from typing import Optional

from .se_module import SEModule


class MetricGAN_SEModule(SEModule):
    """
    SEModule을 상속받아 MetricGAN 학습 로직을 추가한 LightningModule.

    변경점 vs SEModule:
      - automatic_optimization = False  (D/G 교대 수동 업데이트)
      - training_step: PESQ async overlap 구조
          Forward → submit PESQ (background) → G update → collect PESQ → D update
          G update GPU 연산과 PESQ CPU 연산을 overlap하여 D update 대기 최소화.
      - configure_optimizers: G/D 별도 optimizer + scheduler
      - on_train_epoch_end: 스케줄러 수동 step

    YAML 사용법:
        model:
          class_path: src.modules.metricgan_module.MetricGAN_SEModule
          init_args:
            adv_weight: 0.1
            pesq_async: true      # PESQ를 G update와 병렬 실행 (권장)
            model:
              class_path: src.models.LABNet
              ...
            discriminator:
              class_path: src.models.discriminator.MetricDiscriminator
              init_args: {ndf: 16}
            loss:
              class_path: src.modules.losses.LABNetLoss
              ...
            optimizer_config:
              lr: 5e-4
              d_lr: 1e-4
              weight_decay: 1e-2
            scheduler_config:
              warmup_epochs: 5
              min_lr: 1e-6
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss: torch.nn.Module,
        discriminator: torch.nn.Module,
        adv_weight: float = 0.1,
        random_channel_aug: bool = True,
        pesq_async: bool = True,
        **kwargs,
    ):
        super().__init__(model=model, loss=loss, **kwargs)

        self.automatic_optimization = False
        self.discriminator = discriminator
        self.adv_weight = adv_weight
        self.random_channel_aug = random_channel_aug
        self.pesq_async = pesq_async

        # pesq C 확장은 GIL을 해제하므로 ThreadPoolExecutor로 진정한 병렬성 확보.
        # joblib 대비 process spawn 오버헤드 없이 동등한 처리량.
        self._pesq_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    def _get_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._pesq_executor is None:
            n_workers = max(4, os.cpu_count() or 4)
            self._pesq_executor = concurrent.futures.ThreadPoolExecutor(max_workers=n_workers)
        return self._pesq_executor

    def on_train_end(self):
        if self._pesq_executor is not None:
            self._pesq_executor.shutdown(wait=False)
            self._pesq_executor = None
        super().on_train_end()

    # ── PESQ ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_pesq(clean: np.ndarray, est: np.ndarray, sr: int = 16000) -> float:
        try:
            return pesq(sr, clean, est, "wb")
        except Exception:
            return -1.0

    def batch_pesq(self, clean_batch: np.ndarray, est_batch: np.ndarray):
        """
        배치 PESQ 병렬 계산.

        ThreadPoolExecutor 사용: pesq C 확장이 GIL을 해제하므로
        process spawn 오버헤드 없이 joblib와 동등한 병렬성 확보.

        -1(silent/오류) 샘플은 개별 필터링하여 나머지만 사용.
        전체 배치가 모두 -1인 경우에만 (None, None, None) 반환.

        Returns:
            (pesq_scores, scaled_scores, valid_mask) — valid 샘플만, FloatTensor (N,)
            또는 (None, None, None) — 전체 배치가 silent인 경우
        """
        sr = self.sample_rate
        executor = self._get_executor()
        futures = [executor.submit(self._calc_pesq, c, n, sr)
                   for c, n in zip(clean_batch, est_batch)]
        scores = np.array([f.result() for f in futures], dtype=np.float32)

        valid_mask = scores != -1.0
        if not valid_mask.any():
            return None, None, None

        valid_scores = scores[valid_mask]
        scaled = np.clip((valid_scores - 1.0) / 3.5, 0.0, 1.0)
        return torch.FloatTensor(valid_scores), torch.FloatTensor(scaled), valid_mask

    # ── Target 스펙트럼 계산 헬퍼 ─────────────────────────────────────────────

    def _compress_spec(self, wav: torch.Tensor):
        """
        (B, 1, T) waveform → (tgt_mag (B,T,F), tgt_spec (B,T,F) complex)
        power-compressed magnitude와 complex spectrum 동시 반환.
        원본 ABNet forward의 tgt_spec = power_compress(stft(tgt)) 와 동일.
        """
        c = self.loss.compress_factor
        raw_spec = self.model.stft(wav)                     # (B, 1, F, T_spec)
        raw_spec = raw_spec.squeeze(1).permute(0, 2, 1)    # (B, T, F) complex
        mag = raw_spec.abs().clamp(min=1e-8).pow(c)
        # ABNet과 동일: power_compress 후 .angle()로 phase 유도
        # (raw_spec.angle() 직접 사용 시 ±π 경계 bin에서 2π float32 오차 발생)
        phase = torch.complex(mag * raw_spec.angle().cos(),
                              mag * raw_spec.angle().sin()).angle()
        spec_compressed = torch.complex(mag * phase.cos(), mag * phase.sin())
        return mag, spec_compressed

    # ── D loss ───────────────────────────────────────────────────────────────

    def _cal_d_loss(self, tgt_mag: torch.Tensor, est_mag: torch.Tensor,
                    pesq_scores, scaled_scores, valid_mask):
        """
        Discriminator loss. PESQ 결과는 training_step에서 async로 미리 계산되어 전달됨.

          valid 샘플 있으면:
            d_loss = MSE(D(tgt,tgt)[-1], 1)                        — 전체 배치 real pair
                   + MSE(D(tgt[v],est[v].detach())[-1], scaled[v]) — valid만 fake pair
          없으면 (전체 silent):
            d_loss = MSE(D(tgt,tgt)[-1], 1)  — real pair만

        Args:
            tgt_mag:       (B, T, F) power-compressed target magnitude
            est_mag:       (B, T, F) power-compressed estimated magnitude
            pesq_scores:   FloatTensor (N_valid,) 또는 None
            scaled_scores: FloatTensor (N_valid,) [0,1] 또는 None
            valid_mask:    ndarray bool (B,) 또는 None
        Returns:
            d_loss (scalar), pesq_mean (float or None)
        """
        device = tgt_mag.device

        real = self.discriminator(tgt_mag, tgt_mag)
        d_loss = F.mse_loss(real[-1], real[-1].new_ones(real[-1].shape))

        if scaled_scores is not None:
            valid_idx = torch.from_numpy(valid_mask).to(device)
            scaled_scores = scaled_scores.to(device)
            fake = self.discriminator(tgt_mag[valid_idx], est_mag[valid_idx].detach())
            d_loss = d_loss + F.mse_loss(fake[-1].flatten(), scaled_scores)
            pesq_mean = pesq_scores.mean().item()
        else:
            pesq_mean = None

        return d_loss, pesq_mean

    # ── G loss ───────────────────────────────────────────────────────────────

    def _cal_g_loss(self, tgt_mag: torch.Tensor, tgt_spec: torch.Tensor,
                    est_mag: torch.Tensor, est_spec: torch.Tensor):
        """
        Generator loss.
          base_loss = λ_mag * MSE(est_mag, tgt_mag)
                    + λ_comp * (MSE(est_spec.real, tgt_spec.real)
                               + MSE(est_spec.imag, tgt_spec.imag))
          g_loss    = base_loss + adv_weight * MSE(D(tgt_mag, est_mag)[-1], 1)
        """
        loss_mag  = F.mse_loss(est_mag, tgt_mag)
        loss_comp = (F.mse_loss(est_spec.real, tgt_spec.real) +
                     F.mse_loss(est_spec.imag, tgt_spec.imag))
        base_loss = self.loss.lambda_mag * loss_mag + self.loss.lambda_comp * loss_comp

        fake = self.discriminator(tgt_mag, est_mag)
        adv_loss = F.mse_loss(fake[-1], fake[-1].new_ones(fake[-1].shape))
        return base_loss + self.adv_weight * adv_loss, base_loss, adv_loss

    # ── training_step ────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()

        batch = self._apply_gpu_synthesis(batch)
        target = self._select_target(batch)   # (B, 1, T)
        noisy  = batch['noisy']               # (B, C, T)

        if self.random_channel_aug:
            n_ch = random.randint(1, noisy.size(1))
            noisy_input = noisy[:, :n_ch].contiguous()
        else:
            noisy_input = noisy

        # ── Forward pass ─────────────────────────────────────────────────────
        result    = self.model.forward_with_intermediates(noisy_input)
        est_clean = result['est_wav']   # (B, 1, T)
        est_mag   = result['est_mag']   # (B, T, F)
        est_spec  = result['est_spec']  # (B, T, F)

        with torch.no_grad():
            tgt_mag, tgt_spec = self._compress_spec(target)

        # ── PESQ async submit ─────────────────────────────────────────────────
        # Forward 직후 PESQ를 background thread에 즉시 submit.
        # 이후 G update GPU 연산(~수백 ms)과 PESQ CPU 연산이 동시에 실행됨.
        tgt_np = target.squeeze(1).detach().cpu().numpy()
        est_np = est_clean.squeeze(1).detach().cpu().numpy()
        if self.pesq_async:
            pesq_future = self._get_executor().submit(self.batch_pesq, tgt_np, est_np)

        # ── Generator 업데이트 (PESQ CPU와 병렬 실행) ────────────────────────
        g_opt.zero_grad()
        g_loss, base_loss, adv_loss = self._cal_g_loss(tgt_mag, tgt_spec,
                                                        est_mag, est_spec)
        self.manual_backward(g_loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        g_opt.step()

        # ── PESQ 결과 수집 ────────────────────────────────────────────────────
        # G update가 완료된 시점에 collect.
        # G update 시간(GPU) ≈ PESQ 시간(CPU) → 대부분 대기 없이 즉시 수집.
        if self.pesq_async:
            pesq_scores, scaled_scores, valid_mask = pesq_future.result()
        else:
            pesq_scores, scaled_scores, valid_mask = self.batch_pesq(tgt_np, est_np)

        # ── Discriminator 업데이트 ────────────────────────────────────────────
        d_opt.zero_grad()
        d_loss, pesq_mean = self._cal_d_loss(tgt_mag, est_mag,
                                             pesq_scores, scaled_scores, valid_mask)
        self.manual_backward(d_loss)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=5.0)
        d_opt.step()

        # ── 로깅 ─────────────────────────────────────────────────────────────
        bs = noisy.size(0)
        log_dict = {
            'train_g_loss':    g_loss,
            'train_base_loss': base_loss,
            'train_adv_loss':  adv_loss,
            'train_d_loss':    d_loss,
        }
        if pesq_mean is not None:
            log_dict['train_pesq'] = pesq_mean

        self.log_dict(log_dict, on_step=True, on_epoch=True,
                      prog_bar=True, batch_size=bs)

    # ── configure_optimizers ─────────────────────────────────────────────────

    def configure_optimizers(self):
        lr   = self.optimizer_config.get('lr', 5e-4)
        d_lr = self.optimizer_config.get('d_lr', 1e-4)
        wd   = self.optimizer_config.get('weight_decay', 1e-2)

        g_opt = torch.optim.AdamW(self.model.parameters(),         lr=lr,   weight_decay=wd)
        d_opt = torch.optim.AdamW(self.discriminator.parameters(), lr=d_lr, weight_decay=wd)

        warmup = self.scheduler_config.get('warmup_epochs', 5)
        min_lr = self.scheduler_config.get('min_lr', 1e-6)
        T_max  = (self.trainer.max_epochs or 100) - warmup

        def make_sch(opt):
            return torch.optim.lr_scheduler.SequentialLR(
                opt,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        opt, start_factor=1e-2, end_factor=1.0, total_iters=warmup),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt, T_max=max(1, T_max), eta_min=min_lr),
                ],
                milestones=[warmup],
            )

        return [g_opt, d_opt], [make_sch(g_opt), make_sch(d_opt)]

    # ── on_train_epoch_end ───────────────────────────────────────────────────

    def on_train_epoch_end(self):
        for sch in self.lr_schedulers():
            sch.step()
