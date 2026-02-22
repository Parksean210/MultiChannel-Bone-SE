"""
메트릭 유닛 테스트 (tests/test_metrics.py)

대상: src/utils/metrics.py
  - create_metric_suite
  - compute_metrics

참고:
  - STOI / PESQ는 처리 시간이 비교적 길므로 SI-SDR / SDR 위주로 빠른 테스트 수행
  - PESQ는 간헐적 실패(코덱 이슈)가 있어 예외 처리가 구현되어 있음 → 별도 확인
"""
import pytest
import torch

from src.utils.metrics import create_metric_suite, compute_metrics

SR = 16000
T = SR  # 1초


def make_sinusoid(freq: float = 440.0, sr: int = SR) -> torch.Tensor:
    """440Hz 사인파 (1, 1, T) 텐서 생성"""
    t = torch.arange(T).float() / sr
    signal = torch.sin(2 * torch.pi * freq * t) * 0.5
    return signal.unsqueeze(0).unsqueeze(0)  # (1, 1, T)


# ─────────────────────────────────────────────
# create_metric_suite
# ─────────────────────────────────────────────
class TestCreateMetricSuite:
    def test_returns_all_keys(self):
        """si_sdr, sdr, stoi, pesq 키 모두 존재해야 함"""
        suite = create_metric_suite(SR)
        for key in ["si_sdr", "sdr", "stoi", "pesq"]:
            assert key in suite, f"키 '{key}'가 없습니다"

    def test_all_are_callable(self):
        """모든 메트릭 객체는 callable이어야 함"""
        suite = create_metric_suite(SR)
        for name, fn in suite.items():
            assert callable(fn), f"{name}은 callable이 아닙니다"

    def test_returns_dict(self):
        suite = create_metric_suite(SR)
        assert isinstance(suite, dict)

    def test_si_sdr_callable_with_tensors(self):
        """SI-SDR이 실제로 텐서를 처리할 수 있어야 함"""
        suite = create_metric_suite(SR)
        signal = make_sinusoid()
        result = suite["si_sdr"](signal, signal)
        assert result.shape == torch.Size([])  # 스칼라

    def test_sdr_callable_with_tensors(self):
        suite = create_metric_suite(SR)
        signal = make_sinusoid()
        result = suite["sdr"](signal, signal)
        assert result.shape == torch.Size([])


# ─────────────────────────────────────────────
# compute_metrics
# ─────────────────────────────────────────────
class TestComputeMetrics:
    def test_returns_dict(self):
        signal = make_sinusoid()
        result = compute_metrics(signal, signal, SR, metrics=["si_sdr"])
        assert isinstance(result, dict)
        assert "si_sdr" in result

    def test_values_are_float(self):
        """반환값이 Python float이어야 함"""
        signal = make_sinusoid()
        result = compute_metrics(signal, signal, SR, metrics=["si_sdr", "sdr"])
        for name, val in result.items():
            assert isinstance(val, float), f"{name}: float이 아닙니다"

    def test_si_sdr_perfect_high(self):
        """
        완벽한 예측 → SI-SDR이 매우 높아야 함 (> 30 dB).
        eps 처리로 +inf는 아니지만 충분히 큰 값이어야 함.
        """
        signal = make_sinusoid()
        result = compute_metrics(signal, signal, SR, metrics=["si_sdr"])
        assert result["si_sdr"] > 30.0, f"완벽한 예측 SI-SDR이 낮음: {result['si_sdr']:.1f}dB"

    def test_si_sdr_noisy_lower_than_clean(self):
        """노이즈 추가 시 SI-SDR이 낮아져야 함"""
        torch.manual_seed(0)
        clean = make_sinusoid()
        noisy = clean + torch.randn_like(clean) * 0.3

        clean_result = compute_metrics(clean, clean, SR, metrics=["si_sdr"])
        noisy_result = compute_metrics(noisy, clean, SR, metrics=["si_sdr"])
        assert clean_result["si_sdr"] > noisy_result["si_sdr"]

    def test_partial_metrics(self):
        """요청한 메트릭만 반환해야 함"""
        signal = make_sinusoid()
        result = compute_metrics(signal, signal, SR, metrics=["si_sdr", "sdr"])
        assert "si_sdr" in result
        assert "sdr" in result
        assert "stoi" not in result
        assert "pesq" not in result

    def test_accepts_1d_input(self):
        """(T,) 1D 입력도 자동으로 (1, 1, T)로 reshape되어야 함"""
        signal_1d = torch.sin(torch.arange(T).float() * 2 * torch.pi * 440 / SR) * 0.5
        result = compute_metrics(signal_1d, signal_1d, SR, metrics=["si_sdr"])
        assert "si_sdr" in result
        assert result["si_sdr"] > 30.0

    def test_accepts_2d_input(self):
        """(1, T) 2D 입력도 자동 reshape"""
        signal_2d = make_sinusoid().squeeze(0)  # (1, T)
        result = compute_metrics(signal_2d, signal_2d, SR, metrics=["si_sdr"])
        assert result["si_sdr"] > 30.0

    def test_accepts_3d_input(self):
        """(1, 1, T) 3D 입력 (기본 형식)"""
        signal = make_sinusoid()
        result = compute_metrics(signal, signal, SR, metrics=["si_sdr"])
        assert result["si_sdr"] > 30.0

    def test_unknown_metric_skipped(self):
        """존재하지 않는 메트릭은 결과에 포함되지 않아야 함"""
        signal = make_sinusoid()
        result = compute_metrics(signal, signal, SR, metrics=["si_sdr", "nonexistent_metric"])
        assert "si_sdr" in result
        assert "nonexistent_metric" not in result

    def test_si_sdr_range_reasonable(self):
        """SI-SDR은 일반적으로 -∞ ~ +∞지만, 실용 범위 (-30 ~ 50dB)를 확인"""
        torch.manual_seed(1)
        preds = torch.randn(1, 1, T)
        targets = torch.randn(1, 1, T)
        result = compute_metrics(preds, targets, SR, metrics=["si_sdr"])
        assert -50 < result["si_sdr"] < 50

    @pytest.mark.slow
    def test_stoi_range(self):
        """STOI는 0~1 범위. 완벽 예측 ≈ 1.0"""
        signal = make_sinusoid()
        result = compute_metrics(signal, signal, SR, metrics=["stoi"])
        assert "stoi" in result
        assert 0.0 <= result["stoi"] <= 1.0
