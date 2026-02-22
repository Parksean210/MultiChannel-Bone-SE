"""
IC-Mamba 최대 배치 사이즈 탐색 스크립트.
실제 학습 조건(16-mixed AMP, gradient checkpointing)을 그대로 재현.

사용법:
    uv run python scripts/find_max_batch.py
"""
import torch
import gc

def try_batch(model, batch_size: int, T: int = 48000, in_channels: int = 5) -> tuple[bool, float]:
    """forward + backward 1회 실행. (성공 여부, 사용 메모리 GB) 반환"""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    try:
        x = torch.randn(batch_size, in_channels, T, device="cuda", dtype=torch.float16)
        target = torch.randn(batch_size, 1, T, device="cuda", dtype=torch.float16)

        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)          # (B, M, T)
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
    import sys
    sys.path.insert(0, ".")
    from src.models import ICMamba

    TOTAL_VRAM = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {TOTAL_VRAM:.1f} GB\n")

    # ic_mamba.yaml과 동일한 설정
    model = ICMamba(
        in_channels=5,
        out_channels=64,
        enc_kernel=256,
        enc_num_feats=512,
        bot_num_feats=128,
        d_state=16,
        d_conv=4,
        expand=2,
        num_layers=8,
        use_checkpoint=True,   # 학습 조건과 동일
    ).cuda()

    print(f"{'Batch':>6}  {'Peak Mem':>10}  {'Free':>10}  {'Status'}")
    print("-" * 44)

    max_ok = 0
    # 2의 배수로 탐색: 4, 8, 12, 16, 20, 24, 28, 32 ...
    candidates = list(range(4, 65, 4))

    for bs in candidates:
        ok, peak = try_batch(model, bs)
        free = TOTAL_VRAM - peak
        status = "OK" if ok else "OOM"
        if ok:
            print(f"{bs:>6}  {peak:>8.2f}GB  {free:>8.2f}GB  {status}")
            max_ok = bs
        else:
            print(f"{bs:>6}  {'':>10}  {'':>10}  {status}")
            break

    print(f"\n최대 배치 사이즈 (안전): {max_ok}")
    print(f"권장 배치 사이즈 (OOM 여유): {max(4, max_ok - 4)}")


if __name__ == "__main__":
    main()
