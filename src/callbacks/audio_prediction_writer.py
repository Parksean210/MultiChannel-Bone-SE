import torch
import os
from lightning.pytorch.callbacks import BasePredictionWriter
from typing import Any, Optional, Sequence

from src.utils.audio_io import (
    prepare_audio_for_saving,
    save_audio_file,
    build_metadata_filename,
)


class AudioPredictionWriter(BasePredictionWriter):
    """
    추론(Predict) 결과로 나온 오디오 텐서를 WAV 파일로 저장하는 콜백.
    """
    def __init__(self, output_dir: Optional[str] = None, write_interval: str = "batch"):
        super().__init__(write_interval)
        self.output_dir = output_dir or "results/predictions"
        os.makedirs(self.output_dir, exist_ok=True)

    def _extract_meta(self, prediction, index: int, key: str, batch_size: int):
        """prediction 딕셔너리에서 i번째 샘플의 메타데이터를 추출합니다."""
        if key not in prediction or prediction[key] is None:
            return "NA"
        val = prediction[key]

        if isinstance(val, torch.Tensor):
            v = val[index].detach().cpu()
            return v.item() if v.numel() == 1 else v.tolist()

        if isinstance(val, (list, tuple)):
            if len(val) == 0:
                return []
            # collate로 인해 list of batch-tensors가 될 수 있음
            if isinstance(val[0], torch.Tensor) and val[0].shape[0] == batch_size:
                return [
                    src[index].detach().cpu().item() if src[index].numel() == 1
                    else src[index].detach().cpu().tolist()
                    for src in val
                ]
            v = val[index]
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu()
                return v.item() if v.numel() == 1 else v.tolist()
            return v

        return val

    def write_on_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        """각 배치 추론이 끝날 때마다 호출되어 오디오를 저장합니다."""
        noisy = prediction['noisy']       # [B, M, T]
        enhanced = prediction['enhanced'] # [B, M, T]
        target = prediction['target']     # [B, M, T]

        batch_size = noisy.shape[0]
        sr = pl_module.sample_rate

        # 체크포인트 이름 추출 (폴더명으로 사용)
        ckpt_path = trainer.ckpt_path
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0] if ckpt_path else "default_model"

        model_dir = os.path.join(self.output_dir, ckpt_name)
        os.makedirs(model_dir, exist_ok=True)

        for i in range(batch_size):
            # 표준 파일명 접두사 생성
            prefix = build_metadata_filename(
                speech_id=self._extract_meta(prediction, i, 'speech_id', batch_size),
                noise_ids=self._extract_meta(prediction, i, 'noise_ids', batch_size),
                rir_id=self._extract_meta(prediction, i, 'rir_id', batch_size),
                snr=self._extract_meta(prediction, i, 'snr', batch_size),
                suffix="_",
            )

            # 0번 채널(Ref) 기준으로 저장
            for name, tensor in [("noisy", noisy), ("enhanced", enhanced), ("target", target)]:
                wav = prepare_audio_for_saving(tensor[i], channel=0)
                save_audio_file(os.path.join(model_dir, f"{prefix}{name}.wav"), wav, sr)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        pass
