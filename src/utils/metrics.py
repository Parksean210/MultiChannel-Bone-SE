import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
from typing import Optional, Dict, List, Any
from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio as SI_SDR,
    SignalDistortionRatio as SDR,
)
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility as STOI
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ


def create_metric_suite(sample_rate: int = 16000) -> Dict[str, nn.Module]:
    """
    음성 향상 평가용 torchmetrics 객체 세트를 생성합니다.
    DDP 환경에서 올바른 분산 집계를 위해 persistent metric 객체로 사용됩니다.

    Returns:
        {"si_sdr": SI_SDR, "sdr": SDR, "stoi": STOI, "pesq": PESQ}
    """
    return {
        "si_sdr": SI_SDR(),
        "sdr": SDR(),
        "stoi": STOI(fs=sample_rate),
        "pesq": PESQ(fs=sample_rate, mode='wb'),
    }


def compute_and_log_metrics(
    module,
    metric_suite: Dict[str, nn.Module],
    estimated: torch.Tensor,
    target: torch.Tensor,
    prefix: str = "val",
    batch_size: int = 1,
    sync_dist: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    모든 메트릭을 계산하고 Lightning 모듈의 로거에 기록합니다.

    Args:
        module: LightningModule 인스턴스
        metric_suite: create_metric_suite()로 생성된 메트릭 딕셔너리
        estimated: (B, 1, T) 모델 출력
        target: (B, 1, T) 정답
        prefix: 로그 키 접두어 ("val", "test" 등)
        batch_size: 로깅용 배치 크기
        sync_dist: DDP 동기화 여부

    Returns:
        {metric_name: value} 딕셔너리
    """
    # STOI/PESQ는 numpy 기반이라 CPU 텐서만 허용
    _CPU_ONLY = {"stoi", "pesq"}

    results = {}
    for name, metric_fn in metric_suite.items():
        try:
            if name in _CPU_ONLY:
                val = metric_fn.cpu()(estimated.cpu(), target.cpu())
                val = val.to(estimated.device)
            else:
                val = metric_fn.to(estimated.device)(estimated, target)
        except Exception:
            val = torch.tensor(0.0, device=estimated.device)
        results[name] = val
        module.log(f'{prefix}_{name}', val, on_step=False, on_epoch=True,
                   prog_bar=True, sync_dist=sync_dist, batch_size=batch_size)
    return results


def compute_metrics(
    estimated: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int = 16000,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    학습 루프 밖에서 독립적으로 메트릭을 계산합니다.
    스크립트나 노트북에서 사용하기 위한 함수입니다.

    Args:
        estimated: (B, 1, T), (1, T), 또는 (T,)
        target: estimated과 동일한 shape
        sample_rate: 샘플 레이트
        metrics: 계산할 메트릭 이름 리스트. None이면 전체 계산

    Returns:
        {metric_name: float_value}
    """
    if metrics is None:
        metrics = ["si_sdr", "sdr", "stoi", "pesq"]

    # shape 정규화: (B, 1, T)
    for t in [estimated, target]:
        pass  # shape 체크용
    if estimated.dim() == 1:
        estimated = estimated.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)
    elif estimated.dim() == 2:
        estimated = estimated.unsqueeze(0)
        target = target.unsqueeze(0)

    suite = create_metric_suite(sample_rate)
    results = {}

    for name in metrics:
        if name not in suite:
            continue
        if name == "pesq":
            try:
                results[name] = suite[name](estimated, target).item()
            except Exception:
                results[name] = 1.0
        else:
            results[name] = suite[name](estimated, target).item()

    return results


def load_model_from_checkpoint(ckpt_path: str, device: str = "cpu"):
    """
    체크포인트에서 SEModule을 로드합니다.
    LightningCLI가 hyper_parameters에 모델 구조를 저장하므로 별도 감지 불필요.

    Args:
        ckpt_path: 체크포인트 파일 경로
        device: 로드할 디바이스

    Returns:
        SEModule 인스턴스 (eval 모드)
    """
    from src.modules.se_module import SEModule

    return SEModule.load_from_checkpoint(
        ckpt_path, map_location=device
    ).eval().to(device)


def transfer_to_device(batch: Any, device: str) -> Any:
    """배치 내 모든 텐서를 지정된 장치로 재귀적으로 이동합니다."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: transfer_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [transfer_to_device(v, device) for v in batch]
    return batch


@torch.no_grad()
def compare_models(
    checkpoint_paths: List[str],
    dataset,
    num_samples: int = 20,
    snrs: List[float] = [-5, 0, 5, 10, 15, 20],
    device: str = "cuda",
    output_dir: Optional[str] = None,
    sample_rate: int = 16000,
) -> List[Dict]:
    """
    다중 체크포인트를 로드하여 동일한 테스트 샘플에 대해 추론 및 메트릭 비교를 수행합니다.

    Args:
        checkpoint_paths: 체크포인트 파일 경로 리스트
        dataset: SpatialMixingDataset 인스턴스 (test split)
        num_samples: 비교할 샘플 수
        snrs: 테스트할 SNR 값 리스트
        device: 추론 디바이스
        output_dir: 오디오 저장 경로 (None이면 저장 안 함)
        sample_rate: 샘플 레이트

    Returns:
        리스트 of dict: [{model, snr, sample_idx, si_sdr, sdr, stoi, pesq}, ...]
    """
    import random
    from torch.utils.data import DataLoader
    from src.data.dataset import SpatialMixingDataset
    from src.utils.synthesis import apply_spatial_synthesis, create_bcm_kernel
    from src.utils.audio_io import save_audio_file, build_metadata_filename

    # 테스트 샘플 고정
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    fixed_meta = [dataset[i] for i in indices]

    results = []

    for ckpt_path in checkpoint_paths:
        model_name = Path(ckpt_path).stem
        print(f"\n[Model: {model_name}]")

        try:
            pl_module = load_model_from_checkpoint(ckpt_path, device)
        except Exception as e:
            print(f"  Error loading {ckpt_path}: {e}")
            continue

        for snr in snrs:
            for i, meta in enumerate(fixed_meta):
                # 고정 SNR로 데이터셋 재생성
                ds = SpatialMixingDataset(
                    db_path=dataset.engine.url.database,
                    split="test", fixed_snr=snr,
                    speech_id=meta["speech_id"],
                    noise_ids=meta["noise_ids"].tolist(),
                    rir_id=meta["rir_id"]
                )
                loader = DataLoader(ds, batch_size=1)
                batch = next(iter(loader))
                batch = transfer_to_device(batch, device)

                # 합성 + 추론
                bcm_kernel = pl_module.bcm_kernel if hasattr(pl_module, 'bcm_kernel') else None
                batch = apply_spatial_synthesis(batch, bcm_kernel=bcm_kernel, sample_rate=sample_rate)
                est = pl_module(batch["noisy"])

                # 메트릭 계산 (STOI/PESQ는 CPU 연산이므로 CPU로 이동)
                target = batch['aligned_dry'][:, 0:1, :]
                est_ch0 = est[:, 0:1, :]
                metric_vals = compute_metrics(est_ch0.cpu(), target.cpu(), sample_rate)
                metric_vals.update({
                    "model": model_name,
                    "snr": snr,
                    "sample_idx": i,
                })
                results.append(metric_vals)

                # 오디오 저장
                if output_dir:
                    save_dir = os.path.join(output_dir, f"sample_{i}", f"snr_{int(snr)}dB")
                    os.makedirs(save_dir, exist_ok=True)

                    nids = "_".join(map(str, [n for n in meta["noise_ids"].tolist() if n != -1]))
                    tag = f"sid_{meta['speech_id']}_nids_{nids}_rid_{meta['rir_id']}"

                    if ckpt_path == checkpoint_paths[0]:
                        save_audio_file(
                            os.path.join(save_dir, f"input_noisy_{tag}.wav"),
                            batch["noisy"][0, 0].cpu(), sample_rate
                        )
                        save_audio_file(
                            os.path.join(save_dir, f"target_clean_{tag}.wav"),
                            target[0, 0].cpu(), sample_rate
                        )
                    save_audio_file(
                        os.path.join(save_dir, f"output_{model_name}_{tag}.wav"),
                        est_ch0[0, 0].cpu(), sample_rate
                    )

    # 결과 요약 출력
    if results:
        print("\n=== Comparison Results ===")
        print(f"{'Model':<30} {'SNR':>5} {'SI-SDR':>8} {'SDR':>8} {'STOI':>6} {'PESQ':>6}")
        print("-" * 70)

        # 모델별 SNR별 평균
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in results:
            grouped[(r['model'], r['snr'])].append(r)

        for (model, snr_val), items in sorted(grouped.items()):
            avg_si = sum(x.get('si_sdr', 0) for x in items) / len(items)
            avg_sdr = sum(x.get('sdr', 0) for x in items) / len(items)
            avg_stoi = sum(x.get('stoi', 0) for x in items) / len(items)
            avg_pesq = sum(x.get('pesq', 0) for x in items) / len(items)
            print(f"{model:<30} {snr_val:>5.0f} {avg_si:>8.2f} {avg_sdr:>8.2f} {avg_stoi:>6.3f} {avg_pesq:>6.2f}")

    return results
