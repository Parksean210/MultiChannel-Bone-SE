import argparse
import sys
import os
import json

# Add project root to sys.path
sys.path.insert(0, os.getcwd())

from src.db.engine import create_db_engine
from src.db.manager import DatabaseManager

def main():
    parser = argparse.ArgumentParser(description="음향 향상 프로젝트 통합 데이터베이스 관리 도구")
    parser.add_argument("--db_path", type=str, default="data/metadata.db", help="SQLite 데이터베이스 파일 경로")
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 통계 확인 명령어
    subparsers.add_parser("stats", help="데이터베이스 구축 현황 및 통계 출력")

    # 경로 동기화 명령어
    subparsers.add_parser("sync", help="WAV 파일의 NPY 전환에 따른 DB 경로 자동 동기화")

    # 분할 재할당 명령어
    split_parser = subparsers.add_parser("realloc", help="데이터를 Train:Val:Test (8:1:1) 비율로 재지정")
    split_parser.add_argument("--type", choices=["speech", "noise", "rir"], required=True)
    split_parser.add_argument("--ratio", type=float, nargs=3, default=[0.8, 0.1, 0.1], help="Train Val Test 비율 (예: 0.8 0.1 0.1)")

    # 음성 데이터 인덱싱
    speech_parser = subparsers.add_parser("speech", help="음성(Speech) 데이터 인덱싱")
    speech_parser.add_argument("--path", type=str, required=True, help="데이터 최상위 디렉토리")
    speech_parser.add_argument("--dataset", type=str, required=True, help="데이터셋 식별자")
    speech_parser.add_argument("--speaker", type=str, default=None, help="화자 식별자 (생략 시 폴더명 사용)")
    speech_parser.add_argument("--language", type=str, default="ko", help="언어 코드 (ko, en 등)")
    speech_parser.add_argument("--split", choices=["train", "val", "test"], default="train", help="데이터 분할 영역")
    speech_parser.add_argument("--sr", type=int, default=16000, help="NPY 변환 시 사용된 샘플 레이트")

    # 잡음 데이터 인덱싱
    noise_parser = subparsers.add_parser("noise", help="잡음(Noise) 데이터 인덱싱")
    noise_parser.add_argument("--path", type=str, required=True, help="데이터 최상위 디렉토리")
    noise_parser.add_argument("--dataset", type=str, required=True, help="데이터셋 식별 명칭")
    noise_parser.add_argument("--category", type=str, default=None, help="잡음 대분류 (강제 지정)")
    noise_parser.add_argument("--sub", type=str, default=None, help="소분류 강제 지정")
    noise_parser.add_argument("--sub_depth", type=int, default=1, help="디렉토리 구조상 소분류 추출 깊이 (자동 파싱용)")
    noise_parser.add_argument("--split", choices=["train", "val", "test"], default="train", help="데이터 분할 영역")
    noise_parser.add_argument("--sr", type=int, default=16000, help="샘플 레이트")

    # RIR 데이터 인덱싱
    rir_parser = subparsers.add_parser("rir", help="방 임펄스 응답(RIR) 데이터 인덱싱 (.pkl, .wav)")
    rir_parser.add_argument("--path", type=str, required=True, help="RIR 데이터 디렉토리")
    rir_parser.add_argument("--dataset", type=str, default="unknown", help="데이터셋 명칭")
    rir_parser.add_argument("--split", choices=["train", "val", "test"], default="train", help="데이터 분할 영역")
    rir_parser.add_argument("--sr", type=int, default=16000, help="샘플 레이트")

    args = parser.parse_args()

    # Initialize Engine and Manager
    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)
    engine = create_db_engine(args.db_path)
    manager = DatabaseManager(engine)

    if args.command == "stats":
        stats = manager.get_stats()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    elif args.command == "sync":
        manager.sync_paths()

    elif args.command == "realloc":
        manager.reallocate_splits(args.type, tuple(args.ratio))

    elif args.command == "speech":
        manager.index_speech(args.path, args.dataset, speaker=args.speaker, language=args.language, split=args.split, sample_rate=args.sr)
        
    elif args.command == "noise":
        manager.index_noise(args.path, args.dataset, category=args.category, sub_category=args.sub, sub_depth=args.sub_depth, sample_rate=args.sr, split=args.split)
        
    elif args.command == "rir":
        manager.index_rirs(args.path, dataset_name=args.dataset, split=args.split, sample_rate=args.sr)

if __name__ == "__main__":
    main()
