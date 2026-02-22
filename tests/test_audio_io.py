"""
오디오 I/O 유틸리티 유닛 테스트 (tests/test_audio_io.py)

대상: src/utils/audio_io.py
  - prepare_audio_for_saving
  - save_audio_file
  - build_metadata_filename
  - create_spectrogram_image (기본 동작만 확인)
"""
import os
import pytest
import numpy as np
import torch
import soundfile as sf

from src.utils.audio_io import (
    prepare_audio_for_saving,
    save_audio_file,
    build_metadata_filename,
    create_spectrogram_image,
)

SR = 16000
T = SR  # 1초


# ─────────────────────────────────────────────
# prepare_audio_for_saving
# ─────────────────────────────────────────────
class TestPrepareAudioForSaving:
    def test_output_shape_2d(self):
        """(C, T) → (T,) numpy 배열"""
        tensor = torch.randn(3, T)
        result = prepare_audio_for_saving(tensor, channel=0)
        assert result.shape == (T,)
        assert isinstance(result, np.ndarray)

    def test_output_shape_3d(self):
        """(B, C, T) → (T,): 첫 번째 배치, 지정 채널"""
        tensor = torch.randn(2, 3, T)
        result = prepare_audio_for_saving(tensor, channel=1)
        assert result.shape == (T,)

    def test_channel_selection(self):
        """채널 선택이 올바르게 동작해야 함"""
        ch0 = torch.zeros(3, T)
        ch1 = torch.ones(3, T)
        tensor = torch.stack([ch0[0], ch1[0], ch0[0]])  # (3, T): ch1만 1.0

        result_ch0 = prepare_audio_for_saving(tensor, channel=0)
        result_ch1 = prepare_audio_for_saving(tensor, channel=1)

        assert np.allclose(result_ch0, 0.0)
        assert np.allclose(result_ch1, 1.0)

    def test_nan_becomes_zero(self):
        """NaN → 0.0으로 대체"""
        tensor = torch.full((1, 100), float("nan"))
        result = prepare_audio_for_saving(tensor, channel=0)
        assert not np.isnan(result).any()
        assert np.allclose(result, 0.0)

    def test_inf_becomes_clamped(self):
        """Inf → 클리핑 후 ±1.0"""
        tensor = torch.tensor([[float("inf"), float("-inf"), 0.5]])
        result = prepare_audio_for_saving(tensor, channel=0)
        # nan_to_num converts inf → max float, then clamp clips to ±1
        assert result.max() <= 1.0
        assert result.min() >= -1.0

    def test_values_clamped_to_minus_one_plus_one(self):
        """±1.0 초과 값은 클리핑되어야 함"""
        tensor = torch.tensor([[2.5, -3.0, 0.5, -0.7]])
        result = prepare_audio_for_saving(tensor, channel=0)
        assert result.max() <= 1.0
        assert result.min() >= -1.0

    def test_dtype_is_float32(self):
        """출력은 항상 float32 numpy 배열"""
        tensor = torch.randn(1, 100, dtype=torch.float64)
        result = prepare_audio_for_saving(tensor, channel=0)
        assert result.dtype == np.float32

    def test_gradient_not_propagated(self):
        """detach()가 이루어지므로 gradient 없음"""
        tensor = torch.randn(1, 100, requires_grad=True)
        result = prepare_audio_for_saving(tensor, channel=0)
        assert isinstance(result, np.ndarray)  # numpy로 변환 완료


# ─────────────────────────────────────────────
# build_metadata_filename
# ─────────────────────────────────────────────
class TestBuildMetadataFilename:
    def test_basic_pattern(self):
        """기본 패턴: sid_{id}_nids_{ids}_rid_{id}_snr_{snr}dB"""
        name = build_metadata_filename(
            speech_id=1, noise_ids=[2, 3], rir_id=4, snr=10.0
        )
        assert "sid_1" in name
        assert "nids_2_3" in name
        assert "rid_4" in name
        assert "snr_10.0dB" in name

    def test_with_prefix(self):
        name = build_metadata_filename(
            speech_id=1, noise_ids=[2], rir_id=3, snr=5.0, prefix="train_"
        )
        assert name.startswith("train_")

    def test_with_suffix(self):
        name = build_metadata_filename(
            speech_id=1, noise_ids=[2], rir_id=3, snr=5.0, suffix=".wav"
        )
        assert name.endswith(".wav")

    def test_with_step(self):
        """step 번호가 파일명에 포함되어야 함"""
        name = build_metadata_filename(
            speech_id=1, noise_ids=[2], rir_id=3, snr=5.0, step=1000
        )
        assert "step1000" in name

    def test_filters_padding_minus_one(self):
        """-1 패딩은 파일명에서 제거되어야 함"""
        noise_ids = torch.tensor([5, -1, -1, -1, -1, -1, -1])
        name = build_metadata_filename(
            speech_id=1, noise_ids=noise_ids, rir_id=3, snr=5.0
        )
        assert "-1" not in name
        assert "5" in name

    def test_empty_noise_ids_after_filtering(self):
        """모두 -1인 경우 nids_ 뒤가 빈 문자열"""
        noise_ids = torch.tensor([-1, -1, -1])
        name = build_metadata_filename(
            speech_id=1, noise_ids=noise_ids, rir_id=3, snr=5.0
        )
        assert "nids_" in name  # 키는 존재
        assert "-1" not in name

    def test_tensor_ids_accepted(self):
        """텐서 타입 ID를 정상 처리해야 함"""
        name = build_metadata_filename(
            speech_id=torch.tensor(1),
            noise_ids=torch.tensor([2, 3]),
            rir_id=torch.tensor(4),
            snr=torch.tensor(10.0),
        )
        assert "sid_1" in name
        assert "rid_4" in name

    def test_negative_snr(self):
        """음수 SNR도 정상적으로 파일명에 포함"""
        name = build_metadata_filename(
            speech_id=1, noise_ids=[2], rir_id=3, snr=-5.0
        )
        assert "snr_-5.0dB" in name

    def test_returns_string(self):
        name = build_metadata_filename(
            speech_id=1, noise_ids=[2], rir_id=3, snr=5.0
        )
        assert isinstance(name, str)


# ─────────────────────────────────────────────
# save_audio_file
# ─────────────────────────────────────────────
class TestSaveAudioFile:
    def test_creates_file_from_tensor(self, tmp_path):
        """텐서 입력으로 WAV 파일 생성"""
        path = str(tmp_path / "test.wav")
        audio = torch.randn(1, T)  # (C, T)
        save_audio_file(path, audio, sample_rate=SR, channel=0)
        assert os.path.exists(path)

    def test_creates_file_from_numpy(self, tmp_path):
        """numpy 배열 입력으로 WAV 파일 생성"""
        path = str(tmp_path / "test_np.wav")
        audio = np.random.randn(T).astype(np.float32)
        save_audio_file(path, audio, sample_rate=SR)
        assert os.path.exists(path)

    def test_file_readable_with_correct_sr(self, tmp_path):
        """저장된 파일이 올바른 샘플 레이트로 읽혀야 함"""
        path = str(tmp_path / "test.wav")
        audio = torch.randn(1, T)
        save_audio_file(path, audio, sample_rate=SR, channel=0)

        data, sr = sf.read(path)
        assert sr == SR

    def test_file_correct_length(self, tmp_path):
        """저장된 파일 길이가 T 샘플이어야 함"""
        path = str(tmp_path / "test.wav")
        audio = torch.randn(1, T)
        save_audio_file(path, audio, sample_rate=SR, channel=0)

        data, _ = sf.read(path)
        assert data.shape == (T,)

    def test_creates_parent_directories(self, tmp_path):
        """존재하지 않는 상위 폴더를 자동 생성해야 함"""
        path = str(tmp_path / "sub" / "nested" / "test.wav")
        audio = np.zeros(1000, dtype=np.float32)
        save_audio_file(path, audio, sample_rate=SR)
        assert os.path.exists(path)

    def test_values_are_preserved(self, tmp_path):
        """
        저장-읽기 후 값이 보존되어야 함.
        WAV 기본 포맷은 16-bit PCM → 양자화 오차 ≈ 1/32768 ≈ 3e-5.
        따라서 atol=1e-3으로 확인.
        """
        path = str(tmp_path / "test.wav")
        audio = np.array([0.1, -0.2, 0.3, -0.4, 0.5], dtype=np.float32)
        save_audio_file(path, audio, sample_rate=SR)

        data, _ = sf.read(path)
        assert np.allclose(data, audio, atol=1e-3)


# ─────────────────────────────────────────────
# create_spectrogram_image (기본 동작)
# ─────────────────────────────────────────────
class TestCreateSpectrogramImage:
    def test_returns_figure(self):
        """matplotlib Figure 객체를 반환해야 함"""
        import matplotlib.pyplot as plt
        signal = torch.randn(T)
        fig = create_spectrogram_image(signal, sample_rate=SR, n_fft=512, title="Test")
        assert isinstance(fig, plt.Figure)

    def test_saves_png_file(self, tmp_path):
        """save_path 지정 시 파일이 생성되어야 함"""
        path = str(tmp_path / "spec.png")
        signal = torch.randn(T)
        create_spectrogram_image(signal, sample_rate=SR, n_fft=512, save_path=path)
        assert os.path.exists(path)
