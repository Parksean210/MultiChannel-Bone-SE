"""
다중 체크포인트 비교 스크립트.
src.utils.metrics.compare_models를 사용하여 동일한 테스트 샘플에 대해
추론, 메트릭 비교, 오디오 저장을 일괄 수행합니다.
"""
import argparse
import os
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.dataset import SpatialMixingDataset
from src.utils.metrics import compare_models


def get_args():
    parser = argparse.ArgumentParser(description="Professional Model Comparison Script")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="체크포인트 파일 경로 목록")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="비교할 테스트 샘플 수")
    parser.add_argument("--snrs", type=float, nargs="+", default=[-5, 0, 5, 10, 15, 20],
                        help="테스트할 SNR 값 목록 (dB)")
    parser.add_argument("--output_dir", type=str, default="results/comparison",
                        help="오디오 및 결과 저장 디렉토리")
    parser.add_argument("--db_path", type=str, default="data/metadata.db",
                        help="메타데이터 DB 경로")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="추론 디바이스")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="오디오 샘플 레이트")
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = SpatialMixingDataset(db_path=args.db_path, split="test")

    results = compare_models(
        checkpoint_paths=args.checkpoints,
        dataset=dataset,
        num_samples=args.num_samples,
        snrs=args.snrs,
        device=args.device,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
    )

    print(f"\nTotal: {len(results)} entries")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
