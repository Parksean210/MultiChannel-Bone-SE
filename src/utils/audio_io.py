import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import torchaudio
import torchaudio.functional
import os
from typing import Optional, Union, Any


def prepare_audio_for_saving(
    tensor: torch.Tensor,
    channel: int = 0,
) -> np.ndarray:
    """
    텐서를 WAV 저장용 numpy 배열로 변환합니다.
    NaN 제거, 클리핑(-1~1), float32 캐스팅을 수행합니다.

    Args:
        tensor: (C, T) 또는 (B, C, T). 3D면 첫 번째 배치 사용
        channel: 추출할 채널 인덱스

    Returns:
        (T,) numpy float32 array
    """
    if tensor.dim() == 3:
        tensor = tensor[0]
    wav = tensor[channel].detach().cpu().float()
    wav = torch.nan_to_num(wav, nan=0.0)
    wav = torch.clamp(wav, -1.0, 1.0)
    return wav.numpy()


def save_audio_file(
    path: str,
    audio: Union[torch.Tensor, np.ndarray],
    sample_rate: int = 16000,
    channel: int = 0,
) -> None:
    """
    오디오를 WAV 파일로 저장합니다.

    Args:
        path: 출력 파일 경로
        audio: (C, T) 텐서 또는 (T,) numpy 배열
        sample_rate: 샘플 레이트
        channel: 텐서인 경우 추출할 채널
    """
    if isinstance(audio, torch.Tensor):
        audio = prepare_audio_for_saving(audio, channel=channel)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    sf.write(path, audio, sample_rate)


def _to_python(v: Any) -> Any:
    """텐서/리스트를 순수 파이썬 타입으로 변환하는 헬퍼."""
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu()
        return v.item() if v.numel() == 1 else v.tolist()
    if isinstance(v, (list, tuple)):
        return [_to_python(x) for x in v]
    return v


def build_metadata_filename(
    speech_id: Any,
    noise_ids: Any,
    rir_id: Any,
    snr: float,
    prefix: str = "",
    suffix: str = "",
    step: Optional[int] = None,
) -> str:
    """
    메타데이터 기반 표준 파일명을 생성합니다.
    Pattern: {prefix}sid_{id}_nids_{ids}_rid_{id}_snr_{snr}dB{_step{N}}{suffix}

    Args:
        speech_id: 음성 파일 ID
        noise_ids: 노이즈 ID 리스트 또는 텐서 (-1 패딩은 자동 제거)
        rir_id: RIR 파일 ID
        snr: SNR (dB)
        prefix: 파일명 앞에 붙일 문자열
        suffix: 파일명 뒤에 붙일 문자열 (확장자 포함 가능)
        step: 학습 스텝 번호

    Returns:
        생성된 파일명 문자열
    """
    sid = _to_python(speech_id)
    rid = _to_python(rir_id)
    snr_val = _to_python(snr)

    # Noise IDs 처리 (패딩 -1 제거)
    nids_raw = _to_python(noise_ids)
    if isinstance(nids_raw, (list, tuple)):
        nids_filtered = [str(int(n)) for n in nids_raw if n != -1]
    else:
        nids_filtered = [str(int(nids_raw))] if nids_raw != -1 else []
    nids_str = "_".join(nids_filtered)

    snr_str = f"{snr_val:.1f}" if isinstance(snr_val, (int, float)) else str(snr_val)

    parts = []
    if prefix:
        parts.append(prefix)
    parts.append(f"sid_{sid}_nids_{nids_str}_rid_{rid}_snr_{snr_str}dB")
    if step is not None:
        parts[-1] += f"_step{step}"

    return "".join(parts) + suffix


def create_spectrogram_image(
    signal: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 512,
    title: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    스펙트로그램 시각화 이미지를 생성합니다.

    Args:
        signal: (T,) 1D CPU 텐서
        sample_rate: 샘플 레이트
        n_fft: FFT 포인트 수
        title: 그래프 제목
        save_path: 저장 경로 (None이면 저장하지 않음)

    Returns:
        matplotlib Figure 객체
    """
    fig = plt.figure(figsize=(10, 4))
    specgram = torchaudio.transforms.Spectrogram(n_fft=n_fft)(signal)
    spec_db = torchaudio.functional.amplitude_to_DB(specgram, 1.0, 1e-10, 80.0).numpy()

    f_max = sample_rate / 2000.0  # kHz 단위
    plt.imshow(spec_db, aspect='auto', origin='lower', cmap='viridis',
               extent=[0, signal.shape[-1] / sample_rate, 0, f_max])
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (kHz)")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path)

    plt.close(fig)
    return fig
