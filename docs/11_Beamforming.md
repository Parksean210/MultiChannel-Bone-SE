# 빔포밍 모듈 가이드

> `src/utils/beamforming.py` — AR 글래스 4채널 배열 고정 빔포머
>
> 공간 필터 가중치를 오프라인(SOCP/LCMV)으로 계산하고,
> GPU에서 STFT 도메인 행렬곱으로 실시간 적용한다.

---

## 1. 좌표계 및 배열 구성

```
          Front (+y)
              ↑
  Left (+x) ←┼→ Right (−x)
              ↓
          Back  (−y)       z: Up (+)
```

**방위각 (Azimuth)**: 정면(+y)에서 반시계 방향 양수
- 0° = 정면, 90° = 좌, −90° = 우, ±180° = 후방

**고도각 (Elevation)**: 수평에서 위가 양수
- 0° = 수평, −45° = 아래 45° (자기 발화 방향)

### 마이크 위치 (MIC_POSITIONS, 단위: m)

| Mic | 위치 | 좌표 (x, y, z) |
|-----|------|----------------|
| 0 | 좌전 | (−0.07, +0.04, 0) |
| 1 | 우전 | (+0.07, +0.04, 0) |
| 2 | 좌후 | (−0.07, −0.04, 0) |
| 3 | 우후 | (+0.07, −0.04, 0) |

배열 크기: 140 mm (좌우) × 80 mm (전후), 평면 배열 (z=0).

### 5개 빔 방향

| 빔 이름 | az (°) | el (°) | 용도 |
|---------|--------|--------|------|
| front   | 0      | 0      | 정면 화자 |
| left    | 90     | 0      | 좌측 화자 |
| right   | −90    | 0      | 우측 화자 |
| back    | 180    | 0      | 후방 화자 |
| mouth   | 0      | −45    | 자기 발화 (골전도 보조) |

---

## 2. 조향 벡터 (Steering Vector)

자유장(free-field) 평면파 가정:

$$
\mathbf{a}(f, \theta) = e^{-j 2\pi f \, (\mathbf{p}^\top \mathbf{u}(\theta)) / c}
\in \mathbb{C}^M
$$

- $\mathbf{p} \in \mathbb{R}^{3 \times M}$: 마이크 위치 행렬
- $\mathbf{u}(\theta) = [-\sin(\text{az})\cos(\text{el}),\; \cos(\text{az})\cos(\text{el}),\; \sin(\text{el})]^\top$: 방향 단위벡터
- $c = 343$ m/s

### 확산 소음장 공분산 (Diffuse Noise Covariance)

등방성 확산 소음장(isotropic diffuse field) 모델:

$$
\Gamma_{ij}(f) = \text{sinc}\!\left(\frac{2 f d_{ij}}{c}\right), \quad d_{ij} = \|\mathbf{p}_i - \mathbf{p}_j\|
$$

$\text{sinc}(x) = \sin(\pi x)/(\pi x)$ 형태. `diffuse_noise_covariance(freqs, mic_pos)` → `(F, M, M)`.

---

## 3. LCMV 빔포머 (빠른 폐쇄형)

`method='lcmv'` (기본값)

### 수식

최소 분산 무왜곡 응답 (MVDR, 단일 구속):

$$
\mathbf{W}_{\text{MVDR}} = \frac{\boldsymbol{\Gamma}^{-1}\mathbf{a}_t}{\mathbf{a}_t^H \boldsymbol{\Gamma}^{-1} \mathbf{a}_t}
$$

널 방향 추가 시 LCMV (2개 구속):

$$
\mathbf{W}_{\text{LCMV}} = \boldsymbol{\Gamma}^{-1}\mathbf{C}(\mathbf{C}^H\boldsymbol{\Gamma}^{-1}\mathbf{C})^{-1}\mathbf{f}
$$

- $\mathbf{C} = [\mathbf{a}_t,\; \mathbf{a}_n] \in \mathbb{C}^{M \times 2}$
- $\mathbf{f} = [1, 0]^\top$: 타겟 게인=1, 널 방향 게인=0

**주의**: 고주파(>1 kHz)에서 공간 앨리어싱으로 인해 비타겟 방향 게인이 1 이상이 될 수 있다.

---

## 4. SOCP 빔포머 (정확한 게인 제한)

`method='socp'` — `clarabel` 직접 API 사용

### 배경: 공간 앨리어싱 한계

이 배열의 물리적 앨리어싱 주파수:
- 좌우 방향 (d=140 mm): $f_a = c / (2d) \approx 1{,}225$ Hz
- 전후 방향 (d=80 mm):  $f_a = c / (2d) \approx 2{,}144$ Hz

**1,225 Hz 이상에서 단엽(cardioid) 패턴은 물리적으로 불가능하다.** 앨리어싱 방향에서 게인이 1을 초과하는 그레이팅 로브가 필연적으로 발생한다. SOCP는 이를 수학적으로 1.0 이하로 억제할 수 있지만, 메인 로브가 넓어지거나 패턴이 달라지는 트레이드오프가 있다.

### 최적화 문제 (주파수 bin별 독립 풀이)

$$
\begin{aligned}
\min_{\mathbf{w} \in \mathbb{R}^{2M}} \quad & \mathbf{w}^\top \mathbf{Q} \mathbf{w} + \mathbf{q}^\top \mathbf{w} \\
\text{s.t.} \quad
& \text{Re}(\mathbf{W}^H \mathbf{a}_t) = 1 \quad \text{(distortionless)} \\
& \text{Im}(\mathbf{W}^H \mathbf{a}_t) = 0 \\
& \text{Re}(\mathbf{W}^H \mathbf{a}_n) = 0 \quad \text{(hard null, 선택적)} \\
& \text{Im}(\mathbf{W}^H \mathbf{a}_n) = 0 \\
& |\mathbf{W}^H \mathbf{a}(\theta_i)| \leq g_c, \quad i = 1,\ldots,N_s
\end{aligned}
$$

**실수 재공식화**: $\mathbf{w} = [\mathbf{w}_r;\, \mathbf{w}_i] \in \mathbb{R}^{2M}$으로 복소 벡터를 실수 블록으로 변환.

$$
\mathbf{Q}_{\text{real}} = \begin{bmatrix} \boldsymbol{\Gamma}_r & -\boldsymbol{\Gamma}_i \\ \boldsymbol{\Gamma}_i & \boldsymbol{\Gamma}_r \end{bmatrix} \in \mathbb{R}^{2M \times 2M}
$$

### 게인 구속: SOC (Second-Order Cone)

$|\mathbf{W}^H \mathbf{a}_i| \leq g_c$는 아래 SOC(3) 조건으로 변환:

$$
\mathbf{s} = \begin{bmatrix} g_c \\ -\text{Re}(\mathbf{W}^H\mathbf{a}_i) \\ -\text{Im}(\mathbf{W}^H\mathbf{a}_i) \end{bmatrix} \in \mathcal{K}_\text{SOC}^3
\iff g_c \geq \|\mathbf{W}^H\mathbf{a}_i\|
$$

CLARABEL 표준형: `minimize (1/2) x^T P x + q^T x, s.t. Ax + s = b, s ∈ K`

- **P**: 위 $2\mathbf{Q}_{\text{real}}$ (factor-of-2 주의)
- **ZeroConeT(n_eq)**: 등호 구속 (distortionless + null)
- **SecondOrderConeT(3) × N_s**: 방향별 SOC 구속

### 섹터 인식 목적함수 (Sector-aware Objective)

`socp_sector_width=30.0` 설정 시 기본 $\mathbf{W}^H\boldsymbol{\Gamma}\mathbf{W}$ 목적함수 대신:

$$
\min \; \alpha \sum_{|\Delta\text{az}| > \delta} |\mathbf{W}^H\mathbf{a}_i|^2
      + \beta \sum_{|\Delta\text{az}| \leq \delta} |\mathbf{W}^H\mathbf{a}_i - 1|^2
$$

- **외부 섹터** ($|\Delta\text{az}| > \delta$): 억압 (minimize power)
- **내부 섹터** ($|\Delta\text{az}| \leq \delta$): 1에 가깝게 추적 (타겟 주변 메인 로브 유지)
- $\delta$ = `socp_sector_width` (기본 30°)

실수 블록으로 정리하면:

$$
\mathbf{P}_{\text{mixed}} = 2(\alpha\,\mathbf{C}_{\text{out}} + \beta\,\mathbf{C}_{\text{in}}), \qquad
\mathbf{q} = -2\beta \sum_{i \in \text{in}} \begin{bmatrix}\mathbf{a}_{i,r}\\\mathbf{a}_{i,i}\end{bmatrix}
$$

여기서 $\mathbf{C}(S) = \sum_{i\in S} \begin{bmatrix}\mathbf{S}_r^\top\mathbf{S}_r + \mathbf{S}_i^\top\mathbf{S}_i & \mathbf{S}_r^\top\mathbf{S}_i - \mathbf{S}_i^\top\mathbf{S}_r \\ \cdots & \cdots\end{bmatrix}$ (실수 공분산 블록).

### 폴백 체인

각 주파수 bin마다 순서대로 시도:

1. **SOCP (with null)**: 널 구속 포함 풀이
2. **SOCP (without null)**: 널 구속 제거 후 재시도 (드물게 infeasible)
3. **SOCP (relaxed gain_ceil=1.0)**: 게인 상한 완화 후 재시도
4. **MVDR 폴백**: $\boldsymbol{\Gamma}^{-1}\mathbf{a}_t / (\mathbf{a}_t^H\boldsymbol{\Gamma}^{-1}\mathbf{a}_t)$

---

## 5. API 레퍼런스

### `compute_lcmv_weights`

```python
weights, freqs = compute_lcmv_weights(
    n_fft=512,
    sample_rate=16000,
    method='socp',          # 'lcmv' 또는 'socp'
    socp_n_scan=180,        # 방위각 스캔 해상도 (2° 간격)
    socp_sector_width=30.0, # 섹터 반폭 (°), None이면 기본 Γ 목적함수
    socp_sector_alpha=1.0,  # 외부 섹터 억압 가중치
    socp_sector_beta=1.0,   # 내부 섹터 추적 가중치
)
# weights: {'front': (F,M), 'left': (F,M), ..., 'mouth': (F,M)} complex
# freqs:   (F,) Hz,  F = n_fft//2 + 1
```

소요 시간: LCMV ≈ 0.1초, SOCP (n_scan=180, 5빔) ≈ 15–30초.

### `lcmv_weights_to_tensor` + `apply_beamforming`

```python
# 가중치 → GPU 텐서
W = lcmv_weights_to_tensor(weights)   # (K=5, F, M) complex64

# 모델에서 register_buffer로 등록 (1회)
self.register_buffer('beam_weights', W)

# GPU 빔포밍 (STFT 도메인)
x_stft    # (B, M, F, T) complex
beams = apply_beamforming(x_stft, self.beam_weights)  # (B, K, F, T) complex
```

내부적으로 `einsum('kfm,bmft->bkft', W.conj(), x_stft)` — 단순 행렬곱.

### 빔패턴 계산 (시각화)

```python
pattern = beam_pattern_horizontal(
    W_f,            # (M,) complex, 단일 주파수 가중치
    freq_hz,        # Hz
    scan_azimuths,  # (N,) deg
    elevation_deg=0.0,  # 스캔 평면 고도 (mouth 빔은 -45.0)
)  # → (N,) magnitude

pattern_v = beam_pattern_vertical(
    W_f, freq_hz, scan_elevations,
    azimuth_deg=0.0,
    mic_pos=MIC_POSITIONS,
)  # → (N,) magnitude
```

---

## 6. 시각화

```bash
uv run python scripts/visualize_beampattern.py
```

출력:
- `results/beam_patterns_horizontal.png` — 주파수 × 빔 극좌표 그리드 (수평 스캔)
- `results/beam_patterns_mouth_vertical.png` — Mouth 빔 수직 단면 (az=0 고정, el 스캔)

**동적 범위**: 0 dB (외부링 = 타겟 방향) / −6 dB (중심) 고정.
LCMV 목적함수 특성상 타겟 방향은 항상 $|\mathbf{W}^H\mathbf{a}_t|=1.0$ (0 dB) 보장.

---

## 7. 설계 결정 사항

| 항목 | 선택 | 이유 |
|------|------|------|
| 빔포머 종류 | SOCP (기본), LCMV (폴백) | SOCP: 게인 상한 엄격 보장; LCMV: 고속 폐쇄형 |
| 소음 모델 | 등방성 확산 소음장 | 다방향 소음이 섞이는 극한 환경에 적합 |
| 널 방향 | 타겟 반대편 고정 | 후방/측면 지향성 소음 억압; mouth 빔은 후방 널 |
| 섹터 목적함수 | ±30° 내부 추적 + 외부 억압 | 타겟 ±30° 주변 메인 로브 폭 유지하면서 사이드로브 억압 |
| CLARABEL 직접 API | CVXPY 없이 직접 사용 | 257 bin × 180 스캔 × 5빔 = 약 230K SOCP → CVXPY 오버헤드 제거 |
| GPU 적용 | `einsum` 행렬곱 | 가중치는 고정이므로 추론마다 재계산 불필요 |
| 평면 배열 한계 | el=0 배열 → 고도각 구분 불가 | Mouth 빔은 az=0, el=−45 타겟이지만 수평 스캔 패턴은 front 빔과 유사; 수직 패턴으로 차이 확인 가능 |
