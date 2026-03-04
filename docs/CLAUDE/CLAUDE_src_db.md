# src/db/ - Database Management

## Overview
SQLite 기반 메타데이터 데이터베이스의 엔진 생성과 CRUD 관리를 담당한다.

## Files
| File | Class/Function | Description |
|---|---|---|
| `engine.py` | `create_db_engine()` | SQLite 엔진 생성 + 테이블 자동 생성 |
| `manager.py` | `DatabaseManager` | 인덱싱, 분할 재배치, 경로 동기화, 통계 |

## DatabaseManager API
| Method | Description |
|---|---|
| `index_speech(root_dir, dataset_name, ...)` | 음성 파일(.wav/.npy) 스캔 -> DB 인덱싱 |
| `index_noise(root_dir, dataset_name, ...)` | 노이즈 파일 스캔 -> DB 인덱싱. 폴더명에서 카테고리 자동 파싱 |
| `index_rirs(root_dir, ...)` | RIR 파일(.pkl/.wav) 스캔 -> DB 인덱싱 |
| `reallocate_splits(table_type, ratios)` | Train:Val:Test 비율 재배치 (기존 val/test 보존) |
| `sync_paths()` | .wav -> .npy 경로 자동 동기화 |
| `get_stats()` | 전체 데이터 현황 JSON 반환 |

## Key Design Points
- 배치 커밋: 1000건 단위로 트랜잭션 커밋하여 DB 부하 최소화
- 분할 재배치 시 기존 val/test 데이터는 유지 (데이터 오염 방지)
- 노이즈 카테고리는 폴더명 파싱 (TS_/VS_ 접두어 패턴 지원)
- 중복 방지: path unique 제약 + 삽입 전 존재 확인
