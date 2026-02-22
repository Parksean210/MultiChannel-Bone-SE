"""
BaseSEModel 유닛 테스트 (tests/test_models.py)

대상: src/models/base.py
  - __init__ (in_channels, window 등록)
  - stft / istft (복소수 스펙트로그램 변환 + 역변환)
  - to_frames (overlap-based 프레임 분할)
  - forward (NotImplementedError)

모든 테스트는 CPU에서 동작합니다.
"""
import pytest
import torch

from src.models.base import BaseSEModel

B = 2   # batch size
C = 5   # mic channels
T = 16000


@pytest.fixture
def model():
    """기본 설정 BaseSEModel"""
    return BaseSEModel(in_channels=C, n_fft=512, hop_length=256, win_length=512)


@pytest.fixture
def model_small():
    """빠른 테스트용 소형 모델"""
    return BaseSEModel(in_channels=2, n_fft=128, hop_length=64, win_length=128)


# ─────────────────────────────────────────────
# __init__ / 속성
# ─────────────────────────────────────────────
class TestBaseSEModelInit:
    def test_in_channels_attribute(self, model):
        """in_channels 속성이 올바르게 저장되어야 함"""
        assert model.in_channels == C

    def test_window_registered_as_buffer(self, model):
        """window가 register_buffer로 등록되어 state_dict에 포함되어야 함"""
        buffers = dict(model.named_buffers())
        assert "window" in buffers

    def test_window_shape(self, model):
        """window shape == (win_length,)"""
        assert model.window.shape == (model.win_length,)

    def test_hann_window_default(self, model):
        """
        기본 윈도우는 Hann: 양 끝 ≈ 0, 중앙 ≈ 1.
        torch.hann_window(periodic=True, default) 사용 시 마지막 값이
        정확히 0이 아닐 수 있으므로 atol=1e-4로 확인.
        """
        w = model.window
        assert abs(w[0].item()) < 1e-4      # 양 끝 ≈ 0
        assert abs(w[-1].item()) < 1e-4
        assert w[len(w) // 2].item() > 0.9  # 중앙 ≈ 1

    def test_hamming_window(self):
        m = BaseSEModel(window_type="hamming")
        assert m.window.shape[0] == m.win_length

    def test_rect_window(self):
        m = BaseSEModel(window_type="rect")
        assert torch.all(m.window == 1.0)

    def test_invalid_window_type_raises(self):
        """잘못된 window_type → ValueError"""
        with pytest.raises(ValueError, match="Unsupported window_type"):
            BaseSEModel(window_type="blackman")

    def test_forward_not_implemented(self, model):
        """자식 클래스에서 forward 미구현 시 NotImplementedError"""
        x = torch.randn(B, C, T)
        with pytest.raises(NotImplementedError):
            model(x)


# ─────────────────────────────────────────────
# stft
# ─────────────────────────────────────────────
class TestSTFT:
    def test_output_shape(self, model):
        """stft 출력 shape: (B, C, F, T_frames) where F = n_fft/2 + 1"""
        x = torch.randn(B, C, T)
        spec = model.stft(x)
        assert spec.dim() == 4
        assert spec.shape[0] == B
        assert spec.shape[1] == C
        assert spec.shape[2] == model.n_fft // 2 + 1

    def test_output_is_complex(self, model):
        """출력이 복소수 텐서여야 함"""
        x = torch.randn(B, C, T)
        spec = model.stft(x)
        assert spec.is_complex()

    def test_no_nan(self, model):
        x = torch.randn(B, C, T)
        spec = model.stft(x)
        assert not torch.isnan(spec.real).any()
        assert not torch.isnan(spec.imag).any()

    def test_different_signals_differ(self, model):
        """서로 다른 신호의 스펙트로그램은 달라야 함"""
        torch.manual_seed(0)
        x1 = torch.randn(B, C, T)
        x2 = torch.randn(B, C, T)
        spec1 = model.stft(x1)
        spec2 = model.stft(x2)
        assert not torch.allclose(spec1, spec2)


# ─────────────────────────────────────────────
# istft (역변환 + 라운드트립)
# ─────────────────────────────────────────────
class TestISTFT:
    def test_output_shape(self, model):
        """istft 출력 shape: (B, C, T)"""
        x = torch.randn(B, C, T)
        spec = model.stft(x)
        reconstructed = model.istft(spec, length=T)
        assert reconstructed.shape == (B, C, T)

    def test_stft_istft_roundtrip(self, model):
        """
        stft → istft 라운드트립: 원본 신호와 충분히 가까워야 함.
        center=True, Hann window → COLA 조건 만족 → 거의 완벽한 복원.
        """
        torch.manual_seed(5)
        x = torch.randn(B, C, T)
        spec = model.stft(x)
        reconstructed = model.istft(spec, length=T)
        assert torch.allclose(x, reconstructed, atol=1e-4), \
            f"STFT 라운드트립 오차: {(x - reconstructed).abs().max().item():.6f}"

    def test_istft_length_matches(self, model):
        """length 파라미터가 출력 길이를 정확히 제어해야 함"""
        x = torch.randn(B, C, T)
        spec = model.stft(x)
        for target_len in [T, T // 2, T - 100]:
            out = model.istft(spec, length=target_len)
            assert out.shape[-1] == target_len


# ─────────────────────────────────────────────
# to_frames
# ─────────────────────────────────────────────
class TestToFrames:
    def test_output_ndim(self, model):
        """출력이 4D: (B, C, NumFrames, WinLength)"""
        x = torch.randn(B, C, T)
        frames = model.to_frames(x)
        assert frames.dim() == 4

    def test_output_batch_channel(self, model):
        """배치·채널 차원이 유지되어야 함"""
        x = torch.randn(B, C, T)
        frames = model.to_frames(x)
        assert frames.shape[0] == B
        assert frames.shape[1] == C

    def test_win_length_last_dim(self, model):
        """마지막 차원 == win_length"""
        x = torch.randn(B, C, T)
        frames = model.to_frames(x)
        assert frames.shape[3] == model.win_length

    def test_no_nan(self, model):
        x = torch.randn(B, C, T)
        frames = model.to_frames(x)
        assert not torch.isnan(frames).any()

    def test_num_frames_positive(self, model):
        x = torch.randn(B, C, T)
        frames = model.to_frames(x)
        assert frames.shape[2] > 0

    def test_window_applied(self, model_small):
        """
        윈도우가 적용되었는지 간단한 확인:
        zero-padded 신호의 첫 프레임은 window 가중치에 의해 양 끝이 작아야 함.
        """
        B_, C_ = 1, 1
        x = torch.ones(B_, C_, 256)  # DC 신호
        frames = model_small.to_frames(x, center=False)
        frame0 = frames[0, 0, 0, :]  # (WinLength,)
        # Hann 윈도우가 적용되면 양 끝 값이 0에 가까움
        assert frame0[0].item() < 0.1
        assert frame0[-1].item() < 0.1
