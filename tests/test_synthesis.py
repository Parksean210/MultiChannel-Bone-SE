"""
합성 파이프라인 유닛 테스트 (tests/test_synthesis.py)

대상: src/utils/synthesis.py
  - create_bcm_kernel
  - spatialize_sources
  - apply_bcm_modeling
  - scale_noise_to_snr
  - generate_aligned_dry
  - apply_spatial_synthesis

모든 테스트는 실제 DB 없이 합성 텐서만으로 CPU에서 동작합니다.
"""
import torch
import pytest

from src.utils.synthesis import (
    create_bcm_kernel,
    spatialize_sources,
    apply_bcm_modeling,
    scale_noise_to_snr,
    generate_aligned_dry,
    apply_spatial_synthesis,
)

B, M, S, T, L = 2, 5, 8, 16000, 512
SR = 16000


# ─────────────────────────────────────────────
# create_bcm_kernel
# ─────────────────────────────────────────────
class TestCreateBCMKernel:
    def test_output_shape(self, bcm_kernel):
        """커널 shape이 (1, 1, num_taps)인지 확인"""
        assert bcm_kernel.shape == (1, 1, 101)

    def test_sums_to_one(self, bcm_kernel):
        """정규화 확인: 커널 합 ≈ 1.0"""
        assert abs(bcm_kernel.sum().item() - 1.0) < 1e-5

    def test_dtype_float32(self, bcm_kernel):
        assert bcm_kernel.dtype == torch.float32

    def test_lowpass_property(self, bcm_kernel):
        """
        500Hz 커트오프 → 저주파 에너지 >> 고주파 에너지.
        FFT 주파수 응답으로 LPF 특성 검증.
        """
        kernel = bcm_kernel.squeeze()  # (101,)
        freq_response = torch.abs(torch.fft.rfft(kernel, n=2048))
        freqs = torch.fft.rfftfreq(2048, d=1.0 / SR)

        lo_energy = freq_response[freqs < 300].mean()
        hi_energy = freq_response[freqs > 2000].mean()

        # 고주파 에너지가 저주파 에너지의 10% 미만이어야 함
        assert hi_energy < lo_energy * 0.1

    def test_custom_num_taps(self):
        kernel = create_bcm_kernel(cutoff_hz=500.0, sample_rate=SR, num_taps=51)
        assert kernel.shape == (1, 1, 51)

    def test_no_nan(self, bcm_kernel):
        assert not torch.isnan(bcm_kernel).any()


# ─────────────────────────────────────────────
# spatialize_sources
# ─────────────────────────────────────────────
class TestSpatializeSources:
    def test_output_shapes(self, raw_speech, raw_noises, rir_tensor):
        """공간화 출력 shape: (B, M, T)"""
        speech_mc, noise_mc = spatialize_sources(raw_speech, raw_noises, rir_tensor)
        assert speech_mc.shape == (B, M, T)
        assert noise_mc.shape == (B, M, T)

    def test_no_nan(self, raw_speech, raw_noises, rir_tensor):
        speech_mc, noise_mc = spatialize_sources(raw_speech, raw_noises, rir_tensor)
        assert not torch.isnan(speech_mc).any(), "NaN in speech_mc"
        assert not torch.isnan(noise_mc).any(), "NaN in noise_mc"

    def test_return_individual_noise(self, raw_speech, raw_noises, rir_tensor):
        """return_individual_noise=True → 개별 노이즈 컴포넌트 리스트 반환"""
        speech_mc, noise_mc, comps = spatialize_sources(
            raw_speech, raw_noises, rir_tensor, return_individual_noise=True
        )
        assert isinstance(comps, list)
        assert len(comps) == S - 1  # S-1개의 노이즈 소스
        for c in comps:
            assert c.shape == (B, M, T)

    def test_individual_noise_sums_to_total(self, raw_speech, raw_noises, rir_tensor):
        """개별 노이즈들의 합 == 전체 노이즈"""
        _, noise_mc, comps = spatialize_sources(
            raw_speech, raw_noises, rir_tensor, return_individual_noise=True
        )
        noise_sum = sum(comps)
        assert torch.allclose(noise_mc, noise_sum, atol=1e-5)

    def test_delta_rir_identity(self, raw_speech, raw_noises, rir_tensor):
        """
        t=0 델타 RIR → 컨볼루션 = 항등 연산.
        모든 마이크 채널에서 speech_mc ≈ raw_speech.
        """
        speech_mc, _ = spatialize_sources(raw_speech, raw_noises, rir_tensor)
        for m in range(M):
            assert torch.allclose(speech_mc[:, m, :], raw_speech, atol=1e-4), \
                f"Channel {m}: delta RIR identity failed"


# ─────────────────────────────────────────────
# apply_bcm_modeling
# ─────────────────────────────────────────────
class TestApplyBCMModeling:
    def test_output_shapes_unchanged(self, bcm_kernel):
        """BCM 적용 후 shape이 변하지 않아야 함"""
        torch.manual_seed(0)
        speech_mc = torch.randn(B, M, T)
        noise_mc = torch.randn(B, M, T)
        speech_out, noise_out, _ = apply_bcm_modeling(speech_mc, noise_mc, bcm_kernel)
        assert speech_out.shape == (B, M, T)
        assert noise_out.shape == (B, M, T)

    def test_bcm_channel_lowpass(self, bcm_kernel):
        """
        마지막 채널(BCM)에 LPF 적용 확인.
        광대역 노이즈 입력 시 2kHz 이상 에너지가 90% 이상 감소해야 함.
        """
        torch.manual_seed(1)
        speech_mc = torch.randn(B, M, T) * 0.1
        noise_mc = torch.randn(B, M, T) * 0.05

        _, noise_out, _ = apply_bcm_modeling(speech_mc, noise_mc, bcm_kernel)

        fft_size = 1024
        freqs = torch.fft.rfftfreq(fft_size, d=1.0 / SR)
        hi_mask = freqs > 2000

        before_hi = torch.abs(torch.fft.rfft(noise_mc[0, -1, :fft_size])).float()[hi_mask].mean()
        after_hi = torch.abs(torch.fft.rfft(noise_out[0, -1, :fft_size])).float()[hi_mask].mean()

        assert after_hi < before_hi * 0.1, \
            f"BCM LPF 효과 부족: before={before_hi:.4f}, after={after_hi:.4f}"

    def test_non_bcm_channels_mostly_unchanged(self, bcm_kernel):
        """BCM 아닌 채널(0 ~ M-2)의 speech는 원본 그대로여야 함"""
        torch.manual_seed(2)
        speech_mc = torch.randn(B, M, T)
        noise_mc = torch.randn(B, M, T)
        speech_out, _, _ = apply_bcm_modeling(speech_mc, noise_mc, bcm_kernel)

        for ch in range(M - 1):
            assert torch.allclose(speech_out[:, ch, :], speech_mc[:, ch, :], atol=1e-6), \
                f"Channel {ch}: unexpected modification"

    def test_no_nan(self, bcm_kernel):
        torch.manual_seed(3)
        speech_mc = torch.randn(B, M, T)
        noise_mc = torch.randn(B, M, T)
        speech_out, noise_out, _ = apply_bcm_modeling(speech_mc, noise_mc, bcm_kernel)
        assert not torch.isnan(speech_out).any()
        assert not torch.isnan(noise_out).any()


# ─────────────────────────────────────────────
# scale_noise_to_snr
# ─────────────────────────────────────────────
class TestScaleNoiseToSNR:
    def test_snr_accuracy(self, raw_speech, raw_noises, rir_tensor):
        """SNR 스케일링 후 실제 SNR이 목표 SNR과 0.5dB 이내로 일치해야 함"""
        speech_mc, noise_mc = spatialize_sources(raw_speech, raw_noises, rir_tensor)
        target_snr = torch.tensor([10.0, 10.0])
        scaled_noise, _ = scale_noise_to_snr(speech_mc, noise_mc, target_snr)

        eps = 1e-8
        for b in range(B):
            speech_rms = torch.sqrt(torch.mean(speech_mc[b, 0, :] ** 2) + eps)
            noise_rms = torch.sqrt(torch.mean(scaled_noise[b, 0, :] ** 2) + eps)
            actual_snr = 20 * torch.log10(speech_rms / (noise_rms + eps))
            assert abs(actual_snr.item() - 10.0) < 0.5, \
                f"Batch {b}: SNR 오차 {abs(actual_snr.item() - 10.0):.2f}dB"

    def test_output_shape(self, raw_speech, raw_noises, rir_tensor):
        """스케일링 후 shape이 변하지 않아야 함"""
        speech_mc, noise_mc = spatialize_sources(raw_speech, raw_noises, rir_tensor)
        snr = torch.tensor([5.0, 5.0])
        scaled, _ = scale_noise_to_snr(speech_mc, noise_mc, snr)
        assert scaled.shape == noise_mc.shape

    def test_float64_snr_handled(self, raw_speech, raw_noises, rir_tensor):
        """
        float64 snr (collate 후 타입)이 RuntimeError 없이 처리되어야 함.
        내부에서 snr.float()로 변환하므로 dtype 충돌 없어야 함.
        """
        speech_mc, noise_mc = spatialize_sources(raw_speech, raw_noises, rir_tensor)
        snr_f64 = torch.tensor([10.0, 10.0], dtype=torch.float64)
        scaled, _ = scale_noise_to_snr(speech_mc, noise_mc, snr_f64)
        assert scaled.dtype == torch.float32

    def test_higher_snr_lower_noise(self, raw_speech, raw_noises, rir_tensor):
        """높은 SNR → 노이즈 에너지가 낮아야 함"""
        speech_mc, noise_mc = spatialize_sources(raw_speech, raw_noises, rir_tensor)

        scaled_lo, _ = scale_noise_to_snr(speech_mc, noise_mc, torch.tensor([0.0, 0.0]))
        scaled_hi, _ = scale_noise_to_snr(speech_mc, noise_mc, torch.tensor([20.0, 20.0]))

        lo_energy = (scaled_lo ** 2).mean()
        hi_energy = (scaled_hi ** 2).mean()
        assert hi_energy < lo_energy


# ─────────────────────────────────────────────
# generate_aligned_dry
# ─────────────────────────────────────────────
class TestGenerateAlignedDry:
    def test_output_shape(self, raw_speech, rir_tensor):
        """출력 shape: (B, 1, T)"""
        aligned = generate_aligned_dry(raw_speech, rir_tensor)
        assert aligned.shape == (B, 1, T)

    def test_no_nan(self, raw_speech, rir_tensor):
        aligned = generate_aligned_dry(raw_speech, rir_tensor)
        assert not torch.isnan(aligned).any()

    def test_delta_at_zero_identity(self, raw_speech, rir_tensor):
        """
        t=0 델타 RIR → peak_index=0 → 정렬 오프셋 없음.
        aligned_dry == raw_speech.
        """
        aligned = generate_aligned_dry(raw_speech, rir_tensor)
        assert torch.allclose(aligned[:, 0, :], raw_speech, atol=1e-5)

    def test_peak_delay_shifts_alignment(self, raw_speech):
        """
        RIR peak가 t=D에 있으면 aligned_dry는 D샘플 앞으로 정렬.
        즉 aligned_dry[b, 0, D:] == raw_speech[b, :T-D].
        """
        D = 100  # delay samples
        rir = torch.zeros(B, M, S, L)
        rir[:, :, :, D] = 1.0  # peak at index D

        aligned = generate_aligned_dry(raw_speech, rir)
        # aligned[b, 0, D:] should be raw_speech[b, :T-D]
        assert torch.allclose(aligned[:, 0, D:], raw_speech[:, :T - D], atol=1e-5)


# ─────────────────────────────────────────────
# apply_spatial_synthesis (통합)
# ─────────────────────────────────────────────
class TestApplySpatialSynthesis:
    def test_output_keys(self, synthesis_batch, bcm_kernel):
        """출력 배치에 noisy, clean, aligned_dry 키가 있어야 함"""
        out = apply_spatial_synthesis(synthesis_batch, bcm_kernel=bcm_kernel)
        assert "noisy" in out
        assert "clean" in out
        assert "aligned_dry" in out

    def test_output_shapes(self, synthesis_batch, bcm_kernel):
        out = apply_spatial_synthesis(synthesis_batch, bcm_kernel=bcm_kernel)
        assert out["noisy"].shape == (B, M, T)
        assert out["clean"].shape == (B, M, T)
        assert out["aligned_dry"].shape == (B, 1, T)

    def test_no_nan(self, synthesis_batch, bcm_kernel):
        out = apply_spatial_synthesis(synthesis_batch, bcm_kernel=bcm_kernel)
        for key in ["noisy", "clean", "aligned_dry"]:
            assert not torch.isnan(out[key]).any(), f"NaN in {key}"

    def test_without_bcm(self, synthesis_batch):
        """BCM 비활성화 시에도 정상 동작해야 함"""
        import copy
        batch = copy.deepcopy(synthesis_batch)
        batch["mic_config"] = {"use_bcm": False, "bcm_noise_attenuation_db": 20.0}
        out = apply_spatial_synthesis(batch, bcm_kernel=None)
        assert "noisy" in out
        assert not torch.isnan(out["noisy"]).any()

    def test_return_individual_noise(self, synthesis_batch, bcm_kernel):
        """return_individual_noise=True → noise_components 키 존재"""
        import copy
        batch = copy.deepcopy(synthesis_batch)
        out = apply_spatial_synthesis(batch, bcm_kernel=bcm_kernel, return_individual_noise=True)
        assert "noise_components" in out
        assert "noise_only" in out
