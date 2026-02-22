"""
공유 픽스처 (tests/conftest.py)

모든 테스트에서 공통으로 사용하는 합성 텐서 픽스처.
실제 DB나 오디오 파일 없이 순수 CPU에서 동작합니다.
"""
import pytest
import torch

# ─────────────────────────────────────────────
# 전역 상수
# ─────────────────────────────────────────────
B = 2      # batch size
M = 5      # mic channels
S = 8      # total sources (1 speech + 7 noise)
T = 16000  # 1초 @ 16kHz
L = 512    # RIR length
SR = 16000


# ─────────────────────────────────────────────
# 오디오 픽스처
# ─────────────────────────────────────────────
@pytest.fixture
def raw_speech():
    """(B, T) 합성 음성 신호"""
    torch.manual_seed(42)
    return torch.randn(B, T) * 0.1


@pytest.fixture
def raw_noises():
    """(B, S-1, T) 합성 노이즈 신호"""
    torch.manual_seed(43)
    return torch.randn(B, S - 1, T) * 0.05


@pytest.fixture
def rir_tensor():
    """
    (B, M, S, L) 델타 함수 RIR.
    t=0에 1.0 → identity convolution (잔향 없음)
    """
    rir = torch.zeros(B, M, S, L)
    rir[:, :, :, 0] = 1.0
    return rir


@pytest.fixture
def snr_tensor():
    """(B,) 목표 SNR 텐서"""
    return torch.tensor([10.0, 5.0])


@pytest.fixture
def bcm_kernel():
    """create_bcm_kernel로 생성한 500Hz LPF 커널 (1, 1, 101)"""
    from src.utils.synthesis import create_bcm_kernel
    return create_bcm_kernel(cutoff_hz=500.0, sample_rate=SR, num_taps=101)


@pytest.fixture
def synthesis_batch(raw_speech, raw_noises, rir_tensor, snr_tensor):
    """apply_spatial_synthesis에 전달할 완전한 배치 딕셔너리"""
    return {
        "raw_speech": raw_speech,
        "raw_noises": raw_noises,
        "rir_tensor": rir_tensor,
        "snr": snr_tensor,
        "mic_config": {
            "use_bcm": True,
            "bcm_noise_attenuation_db": 20.0,
        },
    }


@pytest.fixture
def sinusoidal_speech():
    """
    440Hz 사인파 신호 (B=1, C=1, T) — 메트릭 테스트용.
    조용한 환경에서 예측 가능한 SI-SDR 값을 얻기 위해 사용.
    """
    t = torch.arange(T).float() / SR
    signal = torch.sin(2 * torch.pi * 440 * t) * 0.5
    return signal.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
