# Git 동기화 가이드 (사외 ↔ 사내망)

> 보안 정책으로 분리된 사외 Git과 사내 Git 사이에서 코드를 안전하게 동기화하는 방법을 설명합니다.
> 사내 개발 코드와 사외 원본 코드를 분리된 브랜치로 관리하여 충돌 없이 업데이트를 받을 수 있습니다.

---

## 브랜치 구조

```
사내 Git (origin)
├── main              ← 개발 브랜치. 우리 커스텀 코드가 여기에 쌓입니다.
└── external-base     ← 사외 원본 미러. 직접 수정 금지 (Read-Only).

사외 Git (upstream)
└── main              ← 오픈소스/외부 원본
```

| 브랜치 | 역할 | 수정 가능 여부 |
|---|---|---|
| `main` | 사내 개발용. `external-base` 기반으로 우리 기능을 추가 | ✅ 개발 진행 |
| `external-base` | 사외 Git 원본 미러. 업데이트 시에만 갱신 | ❌ 직접 수정 금지 |

---

## [최초 1회] 관리자 초기 설정

사외망과 사내망 모두 접근 가능한 PC에서 수행합니다.

```bash
# 1. 사내 저장소 클론
git clone <사내_Git_주소> my-project
cd my-project

# 2. 사외 저장소를 upstream으로 등록
git remote add upstream <사외_Git_주소>

# 3. 사외 코드를 가져와 external-base 브랜치 생성
git fetch upstream
git checkout -b external-base upstream/main

# 4. 사내 서버에 external-base 공유
git push origin external-base
```

---

## [정기 수행] 사외 코드 업데이트 반영

사외 Git에 업데이트가 생겼을 때 관리자가 수행합니다.

### Step 1. `external-base` 갱신 (원본 동기화)

```bash
# 사외 최신 코드 가져오기
git fetch upstream

# external-base로 이동 후 갱신 (충돌 없음)
git checkout external-base
git merge upstream/main

# 사내 서버에 반영
git push origin external-base
```

### Step 2. `main`에 병합 (우리 코드에 적용)

```bash
git checkout main
git merge external-base
```

> **Merge Conflict 발생 시**
> 우리가 수정한 파일과 외부 업데이트가 겹친 경우입니다.
> 충돌 파일을 열어 `<<<<<<< HEAD` / `>>>>>>> external-base` 마커를 보고 수동으로 해결한 뒤:
>
> ```bash
> git add <충돌_해결한_파일>
> git commit -m "Merge external-base: resolve conflicts"
> ```

### Step 3. 사내 공유

```bash
git push origin main
```

---

## [개발자] 일반 개발 워크플로우

일반 팀원은 `external-base`에 신경 쓸 필요 없습니다. `origin`(사내 저장소)만 사용합니다.

```bash
# 최신 코드 받기
git pull origin main

# 기능 브랜치 생성
git checkout -b dev/my-feature

# 작업 후 커밋
git add src/models/my_model.py
git commit -m "Add MyModel architecture"

# 사내 저장소에 올리기
git push origin dev/my-feature
```

---

## 현재 브랜치 구조 확인

```bash
# 로컬 + 리모트 브랜치 전체 확인
git branch -a

# 원본(external-base)과 우리 코드(main)의 차이 확인
git diff external-base main

# 우리가 추가한 커밋만 보기
git log external-base..main --oneline
```

---

## 준수 사항

1. **`external-base` 브랜치에서 직접 작업하지 않습니다.**
   이 브랜치가 오염되면 이후 업데이트를 받을 때 족보가 꼬입니다.

2. **`main` 병합 전 테스트를 반드시 수행합니다.**
   외부 업데이트가 우리 커스텀 코드와 충돌 없이 동작하는지 확인합니다.
   ```bash
   uv run python main.py fit --config configs/baseline.yaml \
       --trainer.limit_train_batches=5 \
       --trainer.limit_val_batches=2 \
       --trainer.max_epochs=1
   ```

3. **`main` 직접 push는 관리자만** 수행합니다. 개발자는 feature 브랜치에서 PR을 통해 병합합니다.
