"""
YAML config 기반 최대 배치 사이즈 탐색 스크립트.
실제 학습 조건(16-mixed AMP, gradient checkpointing)을 그대로 재현.

사용법:
    uv run python scripts/find_max_batch.py --config configs/ic_mamba.yaml
    uv run python scripts/find_max_batch.py --config configs/ic_mamba.yaml --batch_step 2
    uv run python scripts/find_max_batch.py --config configs/ic_mamba.yaml --batch_step 2 --max_batch 32
"""
import argparse
import gc
import importlib
import sys
import yaml
import torch


def load_model_from_config(config_path: str):
    """YAML config에서 모델 클래스와 in_channels를 동적으로 로드."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]["init_args"]["model"]
    class_path = model_cfg["class_path"]   # e.g. "src.models.ICMamba"
    init_args  = model_cfg.get("init_args", {})

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    ModelClass = getattr(module, class_name)

    return ModelClass(**init_args), init_args.get("in_channels", 1)


def try_batch(model, batch_size: int, in_channels: int, T: int = 48000) -> tuple[bool, float]:
    """forward + backward 1회 실행. (성공 여부, 사용 메모리 GB) 반환"""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    try:
        x      = torch.randn(batch_size, in_channels, T, device="cuda", dtype=torch.float16)
        target = torch.randn(batch_size, 1,           T, device="cuda", dtype=torch.float16)

        with torch.autocast("cuda", dtype=torch.float16):
            out  = model(x)
            loss = (out[:, :1] - target).abs().mean()

        loss.backward()
        model.zero_grad(set_to_none=True)

        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        return True, peak_mem

    except torch.cuda.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return False, 0.0


def main():
    sys.path.insert(0, ".")

    parser = argparse.ArgumentParser(description="YAML config 기반 최대 배치 사이즈 탐색")
    parser.add_argument("--config",      required=True,      help="YAML config 경로 (예: configs/ic_mamba.yaml)")
    parser.add_argument("--batch_step",  type=int, default=4, help="배치 탐색 step 크기 (기본: 4)")
    parser.add_argument("--max_batch",   type=int, default=64, help="탐색 상한 배치 사이즈 (기본: 64)")
    parser.add_argument("--start_batch", type=int, default=None, help="탐색 시작 배치 사이즈 (기본: batch_step)")
    args = parser.parse_args()

    model, in_channels = load_model_from_config(args.config)
    model = model.cuda()

    TOTAL_VRAM = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {TOTAL_VRAM:.1f} GB")
    print(f"Model  : {model.__class__.__name__}  (in_channels={in_channels})")
    print(f"Config : {args.config}\n")

    start      = args.start_batch or args.batch_step
    candidates = list(range(start, args.max_batch + 1, args.batch_step))

    print(f"{'Batch':>6}  {'Peak Mem':>10}  {'Free':>10}  {'Status'}")
    print("-" * 44)

    max_ok = 0
    for bs in candidates:
        ok, peak = try_batch(model, bs, in_channels)
        free   = TOTAL_VRAM - peak
        status = "OK" if ok else "OOM"
        if ok:
            print(f"{bs:>6}  {peak:>8.2f}GB  {free:>8.2f}GB  {status}")
            max_ok = bs
        else:
            print(f"{bs:>6}  {'':>10}  {'':>10}  {status}")
            break

    print(f"\n최대 배치 사이즈 (안전)  : {max_ok}")
    print(f"권장 배치 사이즈 (여유분): {max(args.batch_step, max_ok - args.batch_step)}")


if __name__ == "__main__":
    main()
