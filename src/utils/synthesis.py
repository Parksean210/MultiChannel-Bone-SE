import torch
import torch.nn.functional as F
import torchaudio.functional as F_audio
from typing import Optional, Dict, Union


def create_bcm_kernel(
    cutoff_hz: float = 500.0,
    sample_rate: int = 16000,
    num_taps: int = 101,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    골전도 센서(BCM) 모델링용 sinc+hann 로우패스 FIR 커널을 생성합니다.
    SEModule.__init__에서 1회만 호출하여 register_buffer로 캐싱해야 합니다.

    Returns:
        kernel: (1, 1, num_taps) shape, F.conv1d 입력용
    """
    t = torch.arange(num_taps, device=device) - (num_taps - 1) / 2
    fc = cutoff_hz / sample_rate
    sinc = torch.sinc(2 * fc * t)
    window = torch.hann_window(num_taps, periodic=False, device=device)
    kernel = (sinc * window).view(1, 1, -1)
    kernel = kernel / kernel.sum()
    return kernel


def spatialize_sources(
    raw_speech: torch.Tensor,
    raw_noises: torch.Tensor,
    rir_tensor: torch.Tensor,
    return_individual_noise: bool = False,
) -> Union[tuple, tuple]:
    """
    FFT Convolution을 통해 음원을 공간화합니다.

    Args:
        raw_speech: (B, T) 원본 음성
        raw_noises: (B, S-1, T) 노이즈 원본들
        rir_tensor: (B, M, S, L) Room Impulse Response 텐서
        return_individual_noise: True이면 개별 노이즈 컴포넌트 리스트도 반환

    Returns:
        speech_mc: (B, M, T) 공간화된 음성
        noise_mc_total: (B, M, T) 합산된 노이즈
        noise_components: (optional) list of (B, M, T) 개별 노이즈
    """
    B, M, S, L = rir_tensor.shape
    T = raw_speech.shape[-1]

    # 스피치 공간화 (Source Index 0)
    speech_rir = rir_tensor[:, :, 0, :]  # (B, M, L)
    speech_mc = F_audio.fftconvolve(raw_speech.unsqueeze(1), speech_rir, mode="full")
    speech_mc = speech_mc[:, :, :T]

    # 노이즈 공간화 및 누적 (Source Index 1..S-1)
    noise_mc_total = torch.zeros_like(speech_mc)
    noise_components = [] if return_individual_noise else None

    for k in range(1, S):
        noise_rir = rir_tensor[:, :, k, :]
        noise_wav = raw_noises[:, k - 1, :]
        noise_spatialized = F_audio.fftconvolve(noise_wav.unsqueeze(1), noise_rir, mode="full")
        noise_spatialized = noise_spatialized[:, :, :T]
        noise_mc_total += noise_spatialized
        if return_individual_noise:
            noise_components.append(noise_spatialized)

    if return_individual_noise:
        return speech_mc, noise_mc_total, noise_components
    return speech_mc, noise_mc_total


def apply_bcm_modeling(
    speech_mc: torch.Tensor,
    noise_mc_total: torch.Tensor,
    bcm_kernel: torch.Tensor,
    bcm_noise_attenuation_db: float = 20.0,
    noise_components: Optional[list] = None,
) -> tuple:
    """
    마지막 마이크 채널(BCM)에 대해 로우패스 필터링 및 노이즈 감쇄를 적용합니다.

    Args:
        speech_mc: (B, M, T)
        noise_mc_total: (B, M, T)
        bcm_kernel: (1, 1, num_taps) 사전 생성된 LPF 커널
        bcm_noise_attenuation_db: BCM 노이즈 감쇄량 (dB)
        noise_components: 개별 노이즈 리스트 (있으면 함께 처리)

    Returns:
        speech_mc, noise_mc_total, noise_components (수정된 텐서들)
    """
    T = speech_mc.shape[-1]
    num_taps = bcm_kernel.shape[-1]
    pad = num_taps // 2

    # 스피치 BCM 채널 필터링
    bcm_speech = speech_mc[:, -1:]
    bcm_speech_padded = F.pad(bcm_speech, (pad, pad), mode='reflect')
    speech_mc = speech_mc.clone()
    speech_mc[:, -1:] = F.conv1d(bcm_speech_padded, bcm_kernel)[:, :, :T]

    # 노이즈 BCM 채널 필터링
    bcm_noise = noise_mc_total[:, -1:]
    bcm_noise_padded = F.pad(bcm_noise, (pad, pad), mode='reflect')
    noise_mc_total = noise_mc_total.clone()
    noise_mc_total[:, -1:] = F.conv1d(bcm_noise_padded, bcm_kernel)[:, :, :T]

    # 노이즈 감쇄
    atten_factor = 10 ** (-bcm_noise_attenuation_db / 20.0)
    noise_mc_total[:, -1] *= atten_factor

    # 개별 노이즈 컴포넌트도 처리
    if noise_components is not None:
        for nc in noise_components:
            nc_bcm_padded = F.pad(nc[:, -1:], (pad, pad), mode='reflect')
            nc[:, -1:] = F.conv1d(nc_bcm_padded, bcm_kernel)[:, :, :T]
            nc[:, -1] *= atten_factor

    return speech_mc, noise_mc_total, noise_components


def scale_noise_to_snr(
    speech_mc: torch.Tensor,
    noise_mc_total: torch.Tensor,
    snr: torch.Tensor,
    ref_channel: int = 0,
    eps: float = 1e-8,
    noise_components: Optional[list] = None,
) -> tuple:
    """
    참조 채널 기준 RMS로 노이즈를 스케일링하여 목표 SNR을 달성합니다.

    Args:
        speech_mc: (B, M, T)
        noise_mc_total: (B, M, T)
        snr: (B,) 목표 SNR (dB)
        ref_channel: RMS 기준 채널 인덱스
        noise_components: 개별 노이즈 리스트 (있으면 함께 스케일링)

    Returns:
        noise_mc_total, noise_components (스케일링된 텐서들)
    """
    B = speech_mc.shape[0]
    # snr이 collate에서 float64로 들어올 수 있으므로 float32로 통일
    # (float64가 autocast 범위 밖이라 16-mixed precision에서 dtype 충돌 발생)
    snr = snr.float()
    clean_rms = torch.sqrt(torch.mean(speech_mc[:, ref_channel, :] ** 2, dim=-1) + eps)
    noise_rms = torch.sqrt(torch.mean(noise_mc_total[:, ref_channel, :] ** 2, dim=-1) + eps)

    target_factor = (clean_rms / (10 ** (snr / 20))) / (noise_rms + eps)
    noise_mc_total = noise_mc_total * target_factor.view(B, 1, 1)

    if noise_components is not None:
        for nc in noise_components:
            nc *= target_factor.view(B, 1, 1)

    return noise_mc_total, noise_components


def generate_aligned_dry(
    raw_speech: torch.Tensor,
    rir_tensor: torch.Tensor,
    ref_mic: int = 0,
    ref_source: int = 0,
) -> torch.Tensor:
    """
    RIR peak 기반으로 시간 정렬된 드라이 음성을 생성합니다 (벡터화).

    Args:
        raw_speech: (B, T)
        rir_tensor: (B, M, S, L)

    Returns:
        aligned_dry: (B, 1, T)
    """
    B, T = raw_speech.shape
    device = raw_speech.device

    ref_rir = rir_tensor[:, ref_mic, ref_source, :]  # (B, L)
    peak_indices = torch.argmax(torch.abs(ref_rir), dim=-1)  # (B,)
    peak_values = ref_rir[torch.arange(B, device=device), peak_indices]  # (B,) direct path gain

    # 벡터화: time_idx - peak로 소스 인덱스 계산
    time_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # (B, T)
    src_idx = time_idx - peak_indices.unsqueeze(1)  # (B, T)

    valid = (src_idx >= 0) & (src_idx < T)
    src_idx_clamped = src_idx.clamp(0, T - 1)

    aligned = torch.gather(raw_speech, 1, src_idx_clamped)  # (B, T)
    aligned = aligned * valid.float()
    aligned = aligned * peak_values.view(B, 1)  # direct path amplitude 반영

    return aligned.unsqueeze(1)  # (B, 1, T)


def apply_spatial_synthesis(
    batch: Dict,
    bcm_kernel: Optional[torch.Tensor] = None,
    sample_rate: int = 16000,
    return_individual_noise: bool = False,
) -> Dict:
    """
    GPU/CPU 통합 공간 합성 파이프라인.
    SEModule._apply_gpu_synthesis와 scripts/generate_samples.py의 apply_synthesis를 통합합니다.

    Args:
        batch: raw_speech(B,T), raw_noises(B,S-1,T), rir_tensor(B,M,S,L), snr(B,), mic_config
        bcm_kernel: 사전 생성된 BCM LPF 커널 (None이면 BCM 미적용)
        sample_rate: 샘플 레이트
        return_individual_noise: True이면 개별 노이즈 컴포넌트도 반환

    Returns:
        batch에 noisy, clean, aligned_dry 키 추가 (+ noise_components if requested)
    """
    raw_speech = batch['raw_speech']
    raw_noises = batch['raw_noises']
    rir_tensor = batch['rir_tensor']
    snr = batch['snr']

    # 1. 공간화
    result = spatialize_sources(raw_speech, raw_noises, rir_tensor,
                                return_individual_noise=return_individual_noise)
    if return_individual_noise:
        speech_mc, noise_mc_total, noise_components = result
    else:
        speech_mc, noise_mc_total = result
        noise_components = None

    # 2. BCM 모델링
    use_bcm = batch['mic_config']['use_bcm']
    # collated batch에서는 리스트/텐서일 수 있음
    if isinstance(use_bcm, (list, tuple)):
        use_bcm = use_bcm[0]
    elif isinstance(use_bcm, torch.Tensor):
        use_bcm = use_bcm.item() if use_bcm.numel() == 1 else use_bcm[0].item()

    if use_bcm and bcm_kernel is not None:
        atten_db = batch['mic_config']['bcm_noise_attenuation_db']
        if isinstance(atten_db, (list, tuple)):
            atten_db = atten_db[0]
        elif isinstance(atten_db, torch.Tensor):
            atten_db = atten_db.item() if atten_db.numel() == 1 else atten_db[0].item()

        speech_mc, noise_mc_total, noise_components = apply_bcm_modeling(
            speech_mc, noise_mc_total, bcm_kernel, atten_db, noise_components
        )

    # 3. SNR 스케일링
    if isinstance(snr, (int, float)):
        snr = torch.tensor([snr], device=raw_speech.device)
    noise_mc_total, noise_components = scale_noise_to_snr(
        speech_mc, noise_mc_total, snr, noise_components=noise_components
    )

    # 4. 최종 혼합
    noisy_mc = speech_mc + noise_mc_total

    # 5. Aligned Dry 생성
    aligned_dry = generate_aligned_dry(raw_speech, rir_tensor)

    batch['noisy'] = noisy_mc
    batch['clean'] = speech_mc
    batch['aligned_dry'] = aligned_dry

    if return_individual_noise:
        batch['noise_only'] = noise_mc_total
        batch['noise_components'] = noise_components

    return batch
