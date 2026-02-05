# 📘 사외(External) Git 저장소 동기화 및 협업 가이드

이 문서는 보안/네트워크 문제로 분리된 사외 Git 저장소의 코드를 사내 Git으로 안전하게 가져오고, 우리만의 기능을 확장하기 위한 워크플로우를 설명합니다.

---

## 1. 핵심 전략 (Vendor Branch 패턴)

우리는 사내 저장소에서 두 개의 핵심 브랜치를 운영합니다.

| 브랜치 이름 | 역할 및 설명 | 권한 규칙 |
| :--- | :--- | :--- |
| `external-base` | 사외 Git의 원본 미러링. 외부 코드가 업데이트되면 이 브랜치만 최신화됩니다. | ⛔ 직접 수정 금지 (Read-Only) |
| `main` | 사내 협업용 메인 브랜치. `external-base`를 기반으로 우리만의 커스텀 기능이 추가됩니다. | ✅ 개발 및 배포 진행 |

---

## 2. [관리자용] 초기 세팅 (최초 1회)
**담당자:** 사외망과 사내망 모두에 접근 가능한 PC를 가진 관리자

### 사내 저장소 클론 및 리모트 설정
```bash
# 1. 사내 저장소(origin) 클론
git clone [사내_Git_주소] my-project
cd my-project

# 2. 사외 저장소(upstream) 추가
git remote add upstream [사외_Git_주소]
```

### `external-base` 브랜치 생성 및 푸시
```bash
# 사외 저장소 내용 가져오기
git fetch upstream

# 사외 메인 코드를 기반으로 브랜치 생성
git checkout -b external-base upstream/main

# 사내 저장소에 공유
git push origin external-base
```

---

## 3. [관리자용] 사외 코드 업데이트 절차 (정기 수행)
**목표:** 사외 Git에 업데이트가 생겼을 때, `external-base`를 갱신하고 `main`에 병합합니다.

### Step 1. 원본 동기화 (`external-base` 갱신)
가장 먼저 사외 코드를 그대로 가져와 사내 서버에 업데이트합니다. 이 과정에서는 충돌이 발생하지 않습니다.
```bash
# 1. 최신 내용 가져오기
git fetch upstream

# 2. external-base 브랜치로 이동
git checkout external-base

# 3. 사외 코드로 덮어쓰기 (Fast-forward)
git merge upstream/main

# 4. 사내 저장소에 원본 최신화 반영
git push origin external-base
```

### Step 2. 우리 코드에 합치기 (`main` 갱신)
이제 업데이트된 원본을 우리가 개발 중인 코드에 합칩니다.
```bash
# 1. 개발용 브랜치로 이동
git checkout main

# 2. 업데이트된 external-base를 병합
git merge external-base
```

> [!WARNING]
> 이 단계에서 **Merge Conflict(충돌)**가 발생할 수 있습니다.
> *   **충돌 발생 시:** 우리가 수정한 파일과 외부 업데이트 파일이 겹친 것입니다.
> *   **해결 방법:** 사내 개발 담당자와 상의하여 충돌을 해결(`add` & `commit`)한 후 푸시합니다.

```bash
# 3. 최종 결과 사내 공유
git push origin main
```

---

## 4. [개발자용] 동료들을 위한 활용 팁

일반 개발자는 복잡한 upstream 설정 없이 사내 저장소(`origin`)만 사용하면 됩니다.

**Q1. 순수 원본 코드와 우리가 수정한 코드를 비교하고 싶다면?**
`external-base`는 항상 순수한 외부 코드 상태입니다. 아래 명령어로 우리의 작업 내역만 발라내어 볼 수 있습니다.
```bash
# external-base(원본)와 main(우리꺼)의 차이점 비교
git diff external-base main
```

**Q2. 개발은 어떻게 하나요?**
평소처럼 `main` 브랜치에서 Feature 브랜치를 따서 작업하시면 됩니다. `external-base` 브랜치는 참고용으로만 확인하세요.
```bash
git checkout main
git checkout -b feature/my-new-function
```

---

## 5. ⚠️ 절대 지켜야 할 규칙 (Safety Rules)

1.  **`external-base` 브랜치에서는 절대 작업하지 마세요.**
    이 브랜치는 사외 Git과 100% 동일하게 유지되어야 합니다. 여기에 사내 코드가 섞이면 나중에 업데이트를 받을 때 족보가 꼬이게 됩니다.
2.  **`main` 브랜치 업데이트는 신중하게.**
    관리자가 `external-base`를 `main`에 병합할 때는, 기존 기능이 깨지지 않는지 반드시 테스트가 필요합니다.
