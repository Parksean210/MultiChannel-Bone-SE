"""
손실 함수 유닛 테스트 (tests/test_losses.py)

대상: src/modules/losses.py
  - SISDRLoss
  - STFTLoss
  - MultiResolutionSTFTLoss
  - CompositeLoss
"""
import pytest
import torch

from src.modules.losses import (
    SISDRLoss,
    STFTLoss,
    MultiResolutionSTFTLoss,
    CompositeLoss,
)

B, T = 2, 16000


# ─────────────────────────────────────────────
# SISDRLoss
# ─────────────────────────────────────────────
class TestSISDRLoss:
    def test_output_is_scalar(self):
        """손실값이 스칼라 텐서여야 함"""
        loss_fn = SISDRLoss()
        loss = loss_fn(torch.randn(B, T), torch.randn(B, T))
        assert loss.shape == torch.Size([])

    def test_perfect_prediction_is_negative(self):
        """
        완벽한 예측 (pred == target) → SI-SDR = +∞ → 손실 = -∞ 방향.
        적어도 0보다 작아야 함.
        """
        loss_fn = SISDRLoss()
        signal = torch.randn(B, T)
        loss = loss_fn(signal, signal)
        assert loss.item() < 0

    def test_perfect_less_than_random(self):
        """완벽한 예측의 손실 < 무작위 예측의 손실"""
        torch.manual_seed(0)
        loss_fn = SISDRLoss()
        signal = torch.randn(B, T)
        noise = torch.randn(B, T)

        perfect_loss = loss_fn(signal, signal)
        random_loss = loss_fn(noise, signal)
        assert perfect_loss < random_loss

    def test_shape_mismatch_raises(self):
        """shape 불일치 시 RuntimeError"""
        loss_fn = SISDRLoss()
        with pytest.raises(RuntimeError):
            loss_fn(torch.randn(2, 100), torch.randn(2, 200))

    def test_multichannel_input(self):
        """(B, C, T) 멀티채널 입력도 처리 가능해야 함"""
        loss_fn = SISDRLoss()
        preds = torch.randn(B, 3, T)
        targets = torch.randn(B, 3, T)
        loss = loss_fn(preds, targets)
        assert loss.shape == torch.Size([])

    def test_differentiable(self):
        """역전파 가능해야 함 (requires_grad)"""
        loss_fn = SISDRLoss()
        preds = torch.randn(B, T, requires_grad=True)
        targets = torch.randn(B, T)
        loss = loss_fn(preds, targets)
        loss.backward()
        assert preds.grad is not None

    def test_scale_invariant(self):
        """
        SI-SDR은 스케일에 불변해야 함.
        pred를 2배 스케일해도 손실이 동일해야 함.
        """
        torch.manual_seed(1)
        loss_fn = SISDRLoss()
        preds = torch.randn(B, T)
        targets = torch.randn(B, T)

        loss1 = loss_fn(preds, targets)
        loss2 = loss_fn(preds * 2.0, targets)
        assert torch.allclose(loss1, loss2, atol=1e-4)


# ─────────────────────────────────────────────
# STFTLoss
# ─────────────────────────────────────────────
class TestSTFTLoss:
    @pytest.fixture
    def stft_loss(self):
        return STFTLoss(fft_size=512, hop_size=120, win_length=400)

    def test_output_is_scalar(self, stft_loss):
        loss = stft_loss(torch.randn(B, T), torch.randn(B, T))
        assert loss.shape == torch.Size([])

    def test_nonnegative(self, stft_loss):
        """STFT 손실은 항상 0 이상"""
        loss = stft_loss(torch.randn(B, T), torch.randn(B, T))
        assert loss.item() >= 0

    def test_identical_signals_lower(self, stft_loss):
        """동일 신호 손실 < 다른 신호 손실"""
        torch.manual_seed(2)
        signal = torch.randn(B, T)
        loss_identical = stft_loss(signal, signal)
        loss_random = stft_loss(torch.randn(B, T), signal)
        assert loss_identical < loss_random

    def test_multichannel_input(self, stft_loss):
        """(B, C, T) 입력도 처리 가능해야 함"""
        preds = torch.randn(B, 3, T)
        targets = torch.randn(B, 3, T)
        loss = stft_loss(preds, targets)
        assert loss.shape == torch.Size([])

    def test_differentiable(self, stft_loss):
        preds = torch.randn(B, T, requires_grad=True)
        targets = torch.randn(B, T)
        loss = stft_loss(preds, targets)
        loss.backward()
        assert preds.grad is not None


# ─────────────────────────────────────────────
# MultiResolutionSTFTLoss
# ─────────────────────────────────────────────
class TestMultiResolutionSTFTLoss:
    @pytest.fixture
    def mrstft_loss(self):
        return MultiResolutionSTFTLoss(
            fft_sizes=[512, 1024],
            hop_sizes=[120, 240],
            win_lengths=[400, 800],
        )

    def test_output_is_scalar(self, mrstft_loss):
        loss = mrstft_loss(torch.randn(B, T), torch.randn(B, T))
        assert loss.shape == torch.Size([])

    def test_nonnegative(self, mrstft_loss):
        loss = mrstft_loss(torch.randn(B, T), torch.randn(B, T))
        assert loss.item() >= 0

    def test_number_of_stft_losses(self, mrstft_loss):
        """내부에 fft_sizes만큼의 STFTLoss가 있어야 함"""
        assert len(mrstft_loss.stft_losses) == 2

    def test_size_mismatch_raises(self):
        """fft_sizes, hop_sizes, win_lengths 길이가 다르면 AssertionError"""
        with pytest.raises(AssertionError):
            MultiResolutionSTFTLoss(
                fft_sizes=[512, 1024, 2048],
                hop_sizes=[120, 240],     # 길이 불일치
                win_lengths=[400, 800, 1600],
            )

    def test_differentiable(self, mrstft_loss):
        preds = torch.randn(B, T, requires_grad=True)
        targets = torch.randn(B, T)
        mrstft_loss(preds, targets).backward()
        assert preds.grad is not None


# ─────────────────────────────────────────────
# CompositeLoss
# ─────────────────────────────────────────────
class TestCompositeLoss:
    def test_output_is_scalar(self):
        loss_fn = CompositeLoss(alpha=0.1)
        loss = loss_fn(torch.randn(B, T), torch.randn(B, T))
        assert loss.shape == torch.Size([])

    def test_alpha_zero_equals_sisdr(self):
        """alpha=0 이면 CompositeLoss == SISDRLoss"""
        torch.manual_seed(3)
        preds = torch.randn(B, T)
        targets = torch.randn(B, T)

        composite = CompositeLoss(alpha=0.0)(preds, targets)
        sisdr = SISDRLoss()(preds, targets)
        assert torch.allclose(composite, sisdr, atol=1e-5)

    def test_alpha_scaling_changes_loss(self):
        """alpha가 크면 STFT 손실 기여가 늘어 전체 손실이 달라짐"""
        torch.manual_seed(4)
        preds = torch.randn(B, T)
        targets = torch.randn(B, T)

        loss_small = CompositeLoss(alpha=0.0)(preds, targets)
        loss_large = CompositeLoss(alpha=1.0)(preds, targets)
        assert loss_small.item() != pytest.approx(loss_large.item())

    def test_differentiable(self):
        loss_fn = CompositeLoss(alpha=0.1)
        preds = torch.randn(B, T, requires_grad=True)
        targets = torch.randn(B, T)
        loss_fn(preds, targets).backward()
        assert preds.grad is not None
