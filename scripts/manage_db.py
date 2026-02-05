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

    # 음성 데이터 인덱싱
    speech_parser = subparsers.add_parser("speech", help="음성(Speech) 데이터 인덱싱")
    speech_parser.add_argument("--path", type=str, required=True, help="데이터 최상위 디렉토리")
    speech_parser.add_argument("--dataset", type=str, required=True, help="데이터셋 식별자")
    speech_parser.add_argument("--eval", action="store_true", help="평가용(Eval) 데이터로 분류")
    speech_parser.add_argument("--sr", type=int, default=16000, help="NPY 변환 시 사용된 샘플 레이트")

    # 잡음 데이터 인덱싱
    noise_parser = subparsers.add_parser("noise", help="잡음(Noise) 데이터 인덱싱")
    noise_parser.add_argument("--path", type=str, required=True, help="데이터 최상위 디렉토리")
    noise_parser.add_argument("--category", type=str, required=True, help="잡음 대분류")
    noise_parser.add_argument("--sub", type=str, default=None, help="소분류 강제 지정")
    noise_parser.add_argument("--sub_depth", type=int, default=1, help="디렉토리 구조상 소분류 추출 깊이")
    noise_parser.add_argument("--sr", type=int, default=16000, help="샘플 레이트")

    # RIR 데이터 인덱싱
    rir_parser = subparsers.add_parser("rir", help="방 임펄스 응답(RIR) 데이터 인덱싱 (.pkl, .wav)")
    rir_parser.add_argument("--path", type=str, required=True, help="RIR 데이터 디렉토리")

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

    elif args.command == "speech":
        manager.index_speech(args.path, args.dataset, args.eval, sample_rate=args.sr)
        
    elif args.command == "noise":
        manager.index_noise(args.path, args.category, args.sub, sub_depth=args.sub_depth, sample_rate=args.sr)
        
    elif args.command == "rir":
        manager.index_rirs(args.path)

if __name__ == "__main__":
    main()
