# 📊 MLflow 활용 가이드 (MLOps 표준)

| 문서 버전 | 작성일 | 작성자 | 비고 |
| :--- | :--- | :--- | :--- |
| v1.0 | 2026-02-05 | AI Research Team | 초기 작성 |

---

## 1. 개요: MLflow 도입 배경 및 목적

### 1.1. MLflow란 무엇인가?
**MLflow**는 머신러닝의 전체 수명 주기(Lifecycle)를 관리하기 위한 오픈소스 플랫폼입니다. 본 프로젝트에서는 실험 추적(Experiment Tracking) 및 모델 버전 관리(Model Registry)의 핵심 프로세스로 MLflow를 채택하여 사용합니다.

### 1.2. 도입 필요성 (Pain Points & Solutions)
기존의 수동 관리 방식이 가진 한계를 극복하고, 연구 개발 생산성을 극대화하기 위해 다음과 같은 기능을 제공합니다.

| 구분 | 기존 방식의 문제점 | MLflow 도입 효과 |
| :--- | :--- | :--- |
| **실험 기록** | 엑셀/메모장에 수동 기록하므로 누락 및 오기 발생 | 모든 하이퍼파라미터, 메트릭, 소스 코드 버전을 **자동 로깅** |
| **성능 시각화** | 로그 파일을 파싱하여 매번 그래프를 새로 그려야 함 | **실시간 웹 대시보드**를 통해 손실(Loss) 및 성능 추이를 즉시 확인 |
| **재현성 확보** | "지난달에 성능 좋았던 모델 설정이 뭐였지?" 파악 불가 | 실험 당시의 코드 커밋 해시(Git Hash)와 환경 설정을 완벽하게 보존 |
| **비교 분석** | 여러 실험 결과를 엑셀로 취합하여 비교해야 함 | 다중 실험에 대한 **Parallel Coordinates Plot** 및 성능 지표 오버레이 지원 |

### 1.3. 본 프로젝트에서의 역할
본 음성 향상(Speech Enhancement) 프로젝트에서 MLflow는 다음과 같은 역할을 수행합니다.
*   **Metric Tracking:** `train_loss`, `val_loss`, `SI-SDR` 등 핵심 성능 지표의 실시간 모니터링
*   **Artifact Storage:** 생성된 오디오 샘플(`.wav`), 모델 체크포인트(`.ckpt`), 설정 파일(`.yaml`)의 중앙 집중식 저장
*   **Model Comparison:** 다양한 모델 아키텍처(DPRNN vs Wave-U-Net) 및 데이터 처리 기법 간의 정량적 성능 비교

---

## 2. 서버 실행 및 환경 구성 (Server Deployment)

MLflow Tracking Server를 구동하여 웹 대시보드에 접근하기 위한 절차입니다. 운영 목적에 따라 두 가지 실행 방식을 지원합니다.

### 2.1. 실행 모드 (Execution Modes)

#### A. 포그라운드 실행 (Interactive Mode)
단기적인 로그 확인이나 디버깅 목적으로 터미널 세션이 유지되는 동안만 서버를 구동합니다.
```bash
uv run mlflow ui
```
*   **특징:** 터미널 종료 시 서버 프로세스가 즉시 종료됩니다. (`Ctrl + C`로 중단)

#### B. 백그라운드 서비스 실행 (Persistent Service Mode) ✨ 권장
터미널 세션 종료 후에도 상시 접근이 가능하도록 데몬(Daemon) 형태로 실행합니다.
```bash
nohup uv run mlflow ui --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &
```
*   **`nohup`**: 세션 연결이 끊겨도 프로세스 실행을 유지합니다.
*   **`--host 0.0.0.0`**: 외부(Remote) 접속을 허용합니다. (팀원 공유 시 필수)
*   **`> mlflow.log 2>&1`**: 표준 출력 및 에러 로그를 파일로 리다이렉션하여 이력을 보관합니다.
*   **`&`**: 프로세스를 백그라운드에서 실행하여 터미널 제어권을 즉시 반환합니다.

---

## 3. 웹 대시보드 인터페이스 가이드 (Dashboard Interface)

### 3.1. 대시보드 접근 (Access)
브라우저 주소창에 Tracking Server URI를 입력하여 접속합니다.
*   **Local URL:** `http://127.0.0.1:5000` (로컬 접속 시)
*   **Remote URL:** `http://[Server_IP]:5000` (원격 접속 시)

### 3.2. UI 구성 요소 (Components)

#### A. Experiments (사이드바) - 프로젝트 관리
화면 좌측 사이드바는 **실험(Project)** 단위의 그룹을 관리합니다.
*   **역할:** 프로젝트 또는 모델 단위로 실험 기록을 폴더링하여 격리합니다.
*   **사용법:** 본 프로젝트의 실험명인 `baseline_verification`을 선택하여 해당 실험의 Run 목록을 로드합니다. (`Default`는 미지정 시 기본 저장소)

#### B. Runs Table (메인 뷰) - 실행 이력 관리
중앙의 테이블은 개별 **학습 실행(Run)** 이력을 리스트 형태로 제공합니다.

| 컬럼명 | 설명 | 비고 |
| :--- | :--- | :--- |
| **Start Time** | 학습 시작 시간 (Timezone 기준) | 최신순 정렬 권장 |
| **Run Name** | 실행 식별자 (예: `trusting-flea-521`) | 클릭 시 **상세 분석 페이지**로 진입 |
| **Source** | 실험을 실행한 진입점 (Entry Point) | `main.py` 또는 `Git Commit Hash` 표시 |
| **Models** | 등록된 모델 버전 (Model Registry 연동 시) | 배포 단계 확인 가능 |
| **Metrics** | 주요 성과 지표 (`Loss`, `Accuracy` 등) | 목록에서 즉시 비교 가능 |

#### C. Run Detail (상세 라우트) - 심층 분석
Run Name을 클릭하면 진입하는 상세 페이지로, 단일 실행에 대한 심층 정보를 제공합니다.
*   **Parameters:** 학습에 사용된 하이퍼파라미터 (`batch_size`, `lr`, `n_fft` 등) 전체 목록. 재현성을 위한 핵심 정보입니다.
*   **Metrics:** 학습 과정에서 로깅된 시계열 데이터. (섹션 4에서 상세 설명)
*   **Artifacts:** 학습 결과물 저장소. 전처리된 오디오 샘플, 로그 파일, 체크포인트 바이너리가 저장됩니다.

---

## 4. 메트릭 분석 및 시각화 (Metric Analysis & Visualization)

학습 과정에서 수집된 정량적 지표(Metrics)를 분석하여 모델의 수렴 여부와 성능을 판단하는 가이드입니다.

### 4.1. 상세 뷰 진입
1.  **Runs Table**에서 분석 대상 실행의 `Run Name`을 클릭합니다.
2.  상세 페이지 중단 **Metrics** 섹션으로 이동합니다.
3.  `train_loss_epoch` 또는 `val_loss_epoch`와 같은 지표를 클릭하여 **Chart View**를 활성화합니다.

### 4.2. 학습 곡선 해석 가이드 (Learning Curve Interpretation)

| 패턴 유형 | 형상 (Shape) | 진단 (Diagnosis) | 조치 방안 (Action Item) |
| :--- | :--- | :--- | :--- |
| **정상 수렴** (Convergence) | **우하향 곡선** | 모델이 데이터 패턴을 정상적으로 학습 중입니다. | 학습을 지속하거나 `Early Stopping` 시점을 모니터링합니다. |
| **학습 불가** (No Learning) | **평행선** (Flat) | 초기 가중치 설정이나 데이터 로더에 문제가 있습니다. | `Learning Rate`를 높이거나, 입력 데이터의 정규화(Normalization) 상태를 점검합니다. |
| **발산** (Divergence) | **우상향 곡선** | 손실 함수가 최소점을 찾지 못하고 이탈하고 있습니다. | `Learning Rate`를 대폭 낮추거나(1/10 수준), `Gradient Clipping`을 적용합니다. |
| **과적합** (Overfitting) | **Train ↓ / Val ↑** | 학습 데이터에만 편향되어 일반화 성능이 저하되었습니다. | `Dropout`, `Weight Decay`를 적용하거나 데이터 증강(Augmentation)을 강화합니다. |

### 4.3. 시각화 도구 활용 (Visualization Tools)
*   **Smoothing (평활화):** 배치 단위의 손실 값은 노이즈(Jitter)가 심할 수 있습니다. 슬라이더를 조정하여(`0.6` 이상 권장) 전체적인 **추세(Trend)**를 파악하십시오.
*   **Log Scale:** 초기 손실 값이 너무 커서 후반부의 미세한 변화가 보이지 않을 경우, Y축을 로그 스케일로 변환하여 확인합니다.

---

## 5. 심층 시각화 및 실험 비교 (Advanced Visualization & Comparison)

단일 실험 분석을 넘어, 다수 실험 간의 상관관계를 분석하고 최적의 하이퍼파라미터를 도출하기 위한 고급 시각화 도구 사용법입니다.

### 5.1. 다중 실험 비교 모드 (Compare Mode)
1.  **Experiments** 목록에서 비교 대상이 될 실험(Run)들의 체크박스(Checkbox)를 선택합니다.
2.  상단의 **Compare** 버튼을 클릭하여 비교 대시보드로 진입합니다.

### 5.2. 고급 시각화 차트 (Advanced Charts)

| 차트 유형 | 용도 및 분석 포인트 | 활용 예시 |
| :--- | :--- | :--- |
| **Parallel Coordinates** | **하이퍼파라미터 튜닝 분석.** 다차원 파라미터(`lr`, `batch_size`)와 목표 성능(`val_loss`) 간의 인과관계를 선으로 연결하여 시각화합니다. | "Loss가 가장 낮은 선들이 공통적으로 `lr=1e-4`를 지나가는지 확인" |
| **Scatter Plot** | **변수 간 상관관계 분석.** 두 변수(X축: `epoch`, Y축: `loss`) 간의 분포를 산점도로 표현하여 이상치(Outlier)나 경향성을 파악합니다. | "Epoch가 늘어날 때 Loss가 선형적으로 줄어드는지, 특정 시점에서 멈추는지 확인" |
| **Contour Plot** | **최적화 지형 탐색.** 3차원 데이터(X, Y, Z)를 등고선 형태로 표현하여 최적의 성능 구간(Global Minimum)을 시각적으로 탐색합니다. | "Learning Rate와 Batch Size 조합 중 가장 Loss가 낮은 '계곡(Valley)' 지점 찾기" |

### 5.3. 차트 커스터마이징 (Customization)
*   **X축 변경:** 기본 `Step` 외에 `Time (Relative)`로 변경하면, 학습 소요 시간 대비 성능 향상 속도를 비교할 수 있습니다.
*   **Line Smoothness:** 그래프의 노이즈를 제거하여 전체적인 경향성을 뚜렷하게 봅니다.
*   **Export:** 생성된 차트는 우상단 메뉴를 통해 `PNG` 이미지나 `CSV` 데이터로 즉시 다운로드하여 보고서에 활용 가능합니다.

---

## 6. 서비스 및 리소스 관리 (Service & Resource Management)

백그라운드로 구동 중인 MLflow Tracking Server를 안전하게 종료하고 리소스를 관리하는 절차입니다.

### 6.1. 프로세스 식별 및 종료 (Process Termination)
`Persistent Service Mode`(백그라운드)로 실행된 서버는 별도의 종료 명령이 없으므로, PID(Process ID)를 식별하여 종료해야 합니다.

1.  **프로세스 탐색 (Process Discovery):**
    ```bash
    ps -ef | grep mlflow
    ```
    *   **출력 예시:**
        ```text
        user   12345  1  0 10:00 ?  00:00:01 /usr/bin/python3 ... mlflow ui ...
        ```
    *   여기서 `12345`가 해당 프로세스의 **PID**입니다.

2.  **서비스 종료 (Kill Signal):**
    ```bash
    kill -TERM [PID]   # Graceful Shutdown (권장)
    # 또는
    kill -9 [PID]      # Force Kill (응답 없을 시)
    ```

### 6.2. 로그 모니터링 및 트러블슈팅 (Logging & Troubleshooting)

*   **실시간 로그 확인:** 서버 구동 중 발생하는 에러나 접속 기록을 실시간으로 확인합니다.
    ```bash
    tail -f mlflow.log
    ```

*   **포트 충돌 해결 (Port Conflict):**
    `Address already in use` 에러 발생 시, 5000번 포트를 다른 프로세스가 점유 중인 것입니다.
    1.  점유 프로세스 확인: `lsof -i :5000`
    2.  해당 프로세스 종료 후 재실행
    3.  또는 다른 포트로 실행: `mlflow ui --port 5001`
