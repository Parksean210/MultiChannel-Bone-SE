import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class BaseSEModel(nn.Module):
    """
    모든 음향 향상(Speech Enhancement) 모델의 부모 클래스입니다.
    
    이 클래스는 공통적으로 사용되는 STFT(주파수 변환) 및 iSTFT(파형 복원) 기능을 제공하여,
    자식 모델들이 입출력 규격(Waveform -> Waveform)을 일관되게 유지하도록 돕습니다.
    """
    def __init__(self, 
                 fs: int = 16000, 
                 n_fft: int = 512, 
                 hop_length: int = 256, 
                 win_length: Optional[int] = 512,
                 window_type: str = "hann"): # "hann", "hamming", "rect" (no window)
        super().__init__()
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length else n_fft
        
        # Window 함수 설정
        if window_type == "hann":
            window = torch.hann_window(self.win_length)
        elif window_type == "hamming":
            window = torch.hamming_window(self.win_length)
        elif window_type == "rect":
            window = torch.ones(self.win_length)
        else:
            raise ValueError(f"Unsupported window_type: {window_type}")
            
        self.register_buffer('window', window)

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        """
        Waveform (B, C, T) -> Spectrogram (B, C, F, T) 변환
        """
        B, C, T = x.shape
        x_flat = x.view(B * C, T)
        
        spec = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode='reflect'
        )
        # spec: (B*C, F, T) -> (B, C, F, T)
        return spec.view(B, C, spec.shape[1], spec.shape[2])

    def istft(self, x_spec: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """
        Spectrogram (B, C, F, T) -> Waveform (B, C, T) 복원
        """
        B, C, F, T = x_spec.shape
        x_flat = x_spec.view(B * C, F, T)
        
        wav = torch.istft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=length,
            center=True
        )
        # wav: (B*C, T) -> (B, C, T)
        return wav.view(B, C, -1)

    def to_frames(self, x: torch.Tensor, center: bool = True) -> torch.Tensor:
        """
        Waveform (B, C, T) -> Frames (B, C, NumFrames, WinLength)
        """
        B, C, T = x.shape
        
        if center:
            # stft와 동일하게 양쪽에 win_length // 2 만큼 패딩
            pad_amount = self.win_length // 2
            x = F.pad(x, (pad_amount, pad_amount), mode='reflect')
            T = x.shape[-1]

        # (B, C, T) -> (B*C, 1, T)
        x_flat = x.view(B * C, 1, T)
        
        # Unfold (T -> NumFrames, WinLength)
        frames = F.unfold(
            x_flat.unsqueeze(-2), # (B*C, 1, 1, T)
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length)
        )
        
        # (B*C, WinLength, NumFrames) -> (B, C, NumFrames, WinLength)
        NumFrames = frames.shape[-1]
        frames = frames.view(B, C, self.win_length, NumFrames).transpose(-1, -2)
        
        # 윈도우 적용
        frames = frames * self.window
        
        return frames

    def from_frames(self, frames: torch.Tensor, length: Optional[int] = None, center: bool = True) -> torch.Tensor:
        """
        Frames (B, C, NumFrames, WinLength) -> Waveform (B, C, T)
        """
        B, C, NumFrames, WinLength = frames.shape
        
        # Analysis-Synthesis windowing
        frames = frames * self.window
        
        # (B, C, NumFrames, WinLength) -> (B*C, WinLength, NumFrames)
        frames_flat = frames.transpose(-1, -2).reshape(B * C, WinLength, NumFrames)
        
        # Fold를 이용한 Overlap-Add
        pad_amount = WinLength // 2 if center else 0
        current_len = (NumFrames - 1) * self.hop_length + WinLength
        
        output_size = (1, current_len)
        combined = F.fold(
            frames_flat,
            output_size=output_size,
            kernel_size=(1, WinLength),
            stride=(1, self.hop_length)
        )
        
        # 윈도우 제곱합(NOLA)으로 나누어 진폭 보정
        window_sq = (self.window ** 2).view(1, WinLength, 1).expand(B * C, WinLength, NumFrames)
        norm = F.fold(
            window_sq,
            output_size=output_size,
            kernel_size=(1, WinLength),
            stride=(1, self.hop_length)
        )
        
        norm = torch.where(norm > 1e-10, norm, torch.ones_like(norm))
        combined = combined / norm
        
        # Crop to original length
        res = combined.view(B, C, -1)
        if center:
            res = res[:, :, pad_amount : pad_amount + (length if length else T)]
        elif length:
            res = res[:, :, :length]
            
        return res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        자식 클래스에서 이 메서드를 오버라이드하여 실제 모델 로직을 구현해야 합니다.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")
