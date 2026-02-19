import torch
import os
import soundfile as sf
from lightning.pytorch.callbacks import BasePredictionWriter
from typing import Any, Optional, Sequence

class AudioPredictionWriter(BasePredictionWriter):
    """
    추론(Predict) 결과로 나온 오디오 텐서를 WAV 파일로 저장하는 콜백.
    """
    def __init__(self, output_dir: Optional[str] = None, write_interval: str = "batch"):
        super().__init__(write_interval)
        self.output_dir = output_dir or "results/predictions"
        os.makedirs(self.output_dir, exist_ok=True)

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
        """
        각 배치 추론이 끝날 때마다 호출되어 오디오를 저장합니다.
        """
        # prediction is the dictionary returned by predict_step
        noisy = prediction['noisy']     # [B, M, T]
        enhanced = prediction['enhanced'] # [B, M, T]
        target = prediction['target']   # [B, M, T]
        
        batch_size = noisy.shape[0]
        sr = pl_module.sample_rate

        # 체크포인트 이름 추출 (폴더명으로 사용)
        ckpt_path = trainer.ckpt_path
        if ckpt_path:
            ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        else:
            ckpt_name = "default_model"
            
        # 모델별 전용 폴더 생성
        model_dir = os.path.join(self.output_dir, ckpt_name)
        os.makedirs(model_dir, exist_ok=True)

        for i in range(batch_size):
            # 메타데이터를 순수 파이썬 타입으로 변환하는 헬퍼
            def to_py(v):
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu()
                    return v.item() if v.numel() == 1 else v.tolist()
                if isinstance(v, (list, tuple)):
                    return [to_py(x) for x in v]
                return v

            def get_meta(key):
                if key not in prediction or prediction[key] is None: return "NA"
                val = prediction[key]
                
                # Case 1: Tensor [Batch, ...]
                if isinstance(val, torch.Tensor):
                    return to_py(val[i])
                
                # Case 2: List (might be Batch of lists OR List of batch-tensors due to collate)
                if isinstance(val, (list, tuple)):
                    if len(val) == 0: return []
                    # Check if it looks like a list of batched items (e.g. noise_ids)
                    if isinstance(val[0], torch.Tensor) and val[0].shape[0] == batch_size:
                        # gathering i-th element from each source
                        return [to_py(src_tens[i]) for src_tens in val]
                    else:
                        # standard list, take i-th item
                        return to_py(val[i])
                
                return to_py(val)

            sid = get_meta('speech_id')
            rid = get_meta('rir_id')
            snr = get_meta('snr')
            nids = get_meta('noise_ids')
            
            # 파일명용 ID 문자열 생성 (Padding -1 제외)
            if isinstance(nids, (list, tuple)):
                actual_nids = [n for n in nids if n != -1]
                nids_str = "_".join(map(str, actual_nids))
            else:
                nids_str = str(nids) if nids != -1 else ""
                
            snr_str = f"{snr:.1f}" if isinstance(snr, (int, float)) else str(snr)
            
            # 파일명 접두사 생성: sid_3_nids_8_16_rid_9_snr_5.0dB_
            prefix = f"sid_{sid}_nids_{nids_str}_rid_{rid}_snr_{snr_str}dB_"
            
            # 0번 채널(Ref) 기준으로 저장
            def prepare_audio(tensor):
                wav = tensor[i, 0].detach().cpu().float()
                wav = torch.nan_to_num(wav, nan=0.0)
                wav = torch.clamp(wav, -1.0, 1.0)
                return wav.numpy()

            n_0 = prepare_audio(noisy)
            e_0 = prepare_audio(enhanced)
            t_0 = prepare_audio(target)
            
            sf.write(os.path.join(model_dir, f"{prefix}noisy.wav"), n_0, sr)
            sf.write(os.path.join(model_dir, f"{prefix}enhanced.wav"), e_0, sr)
            sf.write(os.path.join(model_dir, f"{prefix}target.wav"), t_0, sr)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        pass
