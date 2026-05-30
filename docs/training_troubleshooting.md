# GPU 재학습 시행착오 기록
> 작성일: 2026-05-28 | 환경: WSL2 Ubuntu-24.04, RTX 3070 Laptop GPU, TF 2.21.0

---

## 1. 개요

새로 증강된 데이터셋(총 21,425개 → clean 13,885개)으로 YAMNet 파인튜닝 재학습을 시도하는 과정에서 발생한 문제와 해결책을 기록한다.

**데이터 구성 (학습 전)**
| 클래스 | 원본 | 증강/합성 후 |
|--------|------|------------|
| background | 2,000 | 12,000 (10,000 합성 추가) |
| car_horn | 429 | 956 (527 augmentation 추가) |
| siren | 929 | 929 (변동 없음) |
| **합계** | **3,358** | **13,885** |

---

## 2. 문제 1: BOM(Byte Order Mark) 깨짐

### 발생 상황
PowerShell `Out-File`로 run_finetune.sh 작성 시 파일 첫 줄에 UTF-8 BOM(`\xEF\xBB\xBF`)이 삽입됨.

```
/bin/bash: #!/bin/bash: No such file or directory
```

### 원인
PowerShell 5.1의 `Out-File` 기본 인코딩이 UTF-16 LE(BOM 포함)이며, `-Encoding utf8`도 UTF-8 with BOM을 출력함.

### 영향
쉐뱅(`#!/bin/bash`) 파싱 실패. 단, bash는 첫 줄 오류 무시 후 계속 실행하므로 실제 학습에는 영향 없었음.

### 해결
스크립트 실행 자체는 정상 동작했으므로 무시. 근본 해결책은 `-Encoding ascii` 또는 WSL 내에서 직접 `echo`로 작성.

---

## 3. 문제 2: OOM(Out of Memory) — 핵심 문제

### 발생 상황
첫 번째 학습 시도(14:44 시작) 직후, waveform 로딩 완료 후 약 18초 만에 프로세스가 무음 종료.

```
=== ADAS Finetune Start: Thu May 28 14:44:42 KST 2026 ===
...
Loaded 11161 waveforms — shape (11161, 64000)
...
Loaded 1364 waveforms — shape (1364, 64000)
Created device GPU:0 with 5595 MB memory  ← GPU 감지됨
Allocation of 2857216000 exceeds 10% of free system memory.  ← 경고
=== ADAS Finetune End: Thu May 28 14:49:37 KST 2026 ===  ← 즉시 종료!
```

### 원인 분석

`_dedup_to_clean()` 함수가 SNR 파일을 제외했지만, clean 파티션에 synthesize_background.py가 생성한 합성 background 10,000개가 포함돼 있었음.

**메모리 계산**:
```
X_train (11,161개) × 64,000 samples × 4 bytes (float32) = 2,857,216,000 bytes = 2.86 GB
tf.data.Dataset.from_tensor_slices()가 numpy 배열의 복사본 생성: +2.86 GB
X_val: 1,360 × 64,000 × 4 = 348 MB
X_test: 1,364 × 64,000 × 4 = 349 MB
───────────────────────────────────────────────────────
합계: ≈ 6.5 GB  +  Python/TF 오버헤드  →  7.5 GB WSL2 RAM 초과
```

WSL2 할당 RAM: 7.5 GB → 가용 메모리 초과로 OS OOM-killer 발동.

### 해결책 — background 샘플 수 상한(cap) 적용

**config.yaml 수정**:
```yaml
finetune:
  background_cap: 2000  # 추가: RAM OOM 방지
```

**finetune.py 수정** (`run_finetune()` 내 `fold_split()` 직후에 삽입):
```python
# ── 1b. background 샘플 수 상한 적용 (RAM OOM 방지) ──
bg_cap: int = ft_cfg.get("background_cap", 2000)
path_col = train_df["path"].str.replace("\\", "/", regex=False)
is_snr   = path_col.str.contains("processed/snr_", regex=False)
clean_df = train_df[~is_snr]
bg_mask  = clean_df["class"] == "background"
n_bg     = bg_mask.sum()
if n_bg > bg_cap:
    bg_idx    = clean_df[bg_mask].sample(n=bg_cap, random_state=seed).index
    other_idx = clean_df[~bg_mask].index
    train_df  = train_df.loc[bg_idx.union(other_idx)]
    logger.info(
        "background 상한 적용: %d → %d  (cap=%d, train 전체: %d)",
        n_bg, bg_cap, bg_cap, len(train_df),
    )
```

**결과**:
```
background 상한 적용: 9,606 → 2,000  (cap=2000, train 전체: 3,555)
X_train 메모리: 3,555 × 64,000 × 4 = 912 MB  (OOM 해결)
```

---

## 4. 문제 3: 내장 GPU(라데온) 사용 의심

### 발생 상황
Windows 작업 관리자에서 WSL2 Python 프로세스가 내장 Radeon GPU 항목 아래 표시됨.

### 원인
Windows 작업 관리자는 WSL2 프로세스의 **화면 렌더링**을 내장 GPU로 분류하는 버그/특성이 있음. 실제 CUDA 연산은 nvidia-smi로만 정확히 확인 가능.

### 확인 방법
```bash
wsl -d Ubuntu-24.04 nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

**정상 출력 (학습 중)**:
```
45 %, 3200 MiB
```

### 결론
`gpu_device.cc: Created device GPU:0 … NVIDIA GeForce RTX 3070 Laptop GPU, compute capability: 8.6` 로그 확인 → **RTX 3070 정상 사용 중**.

---

## 5. 최종 학습 구성 (2차 시도 — 성공)

| 항목 | 값 |
|------|-----|
| 시작 시각 | 2026-05-28 14:51:50 KST |
| 학습 프레임워크 | TensorFlow 2.21.0 + CUDA 12.3 + cuDNN 9.2.2 |
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU (8 GB VRAM, 5,595 MB 할당) |
| 데이터 | train 3,555 / val 1,360 / test 1,364 |
| YAMNet 변수 | 56개 중 33개(60%) 훈련, 23개 동결 |
| Epochs | 최대 100 (Early Stopping patience=10) |
| Batch size | 16 |
| yamnet_lr | 5e-6 |
| clf_lr | 2e-5 |

**초기 학습 결과**:
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | 비고 |
|-------|-----------|-----------|----------|---------|------|
| 1 | 3.3920 | 30.7% | 1.0379 | 53.1% | best |
| 2 | 1.1028 | 63.4% | 0.4183 | 85.6% | best |
| 3 | 0.7963 | 76.8% | 0.2259 | 96.2% | best |
| 4 | 0.5495 | 85.0% | 0.1827 | 96.5% | best |
| 5 | 0.4393 | 89.1% | 0.1766 | 97.0% | best |
| 6 | 0.3611 | 90.3% | 0.1571 | 97.0% | best |
| 7 | 0.3118 | 92.4% | 0.1795 | 95.9% | — |
| 8 | 0.2817 | 93.3% | 0.1545 | 96.2% | best |
| 9 | 0.2147 | 94.5% | 0.1547 | 96.5% | — |
| 10 | 0.1898 | 95.4% | 0.1426 | 96.3% | best |
| 11 | 0.1959 | 96.3% | **0.1358** | 96.7% | best |
| 12 | 0.1704 | 96.2% | 0.1676 | 97.4% | — |
| 13 | 0.1857 | 96.1% | 0.1374 | 97.3% | — |
| 14 | 0.1629 | 96.6% | **0.1076** | 97.1% | best |
| 15 | 0.1924 | 96.1% | 0.1724 | 96.9% | — |
| 16 | 0.1421 | 97.1% | 0.1605 | 97.5% | — |
| 17 | 0.1219 | 97.5% | 0.1672 | 97.4% | — |
| 18 | 0.1155 | 97.4% | 0.1509 | 97.5% | — |
| 19 | 0.1067 | 97.8% | 0.1762 | 97.4% | — (216s, 시스템 부하) |
| 20 | 0.1349 | 97.6% | 0.1189 | 97.7% | patience 6/10 |
| 21 | 0.1126 | 97.5% | 0.0821 | 98.0% | ✅ NEW BEST |
| 22 | 0.1182 | 97.6% | **0.0788** | 98.2% | ✅ NEW BEST |
| 23 | 0.1223 | 97.5% | 0.0817 | 98.5% | patience 1/10 |
| 24 | 0.0890 | **98.3%** | 0.1029 | 97.8% | patience 2/10 |
| 25 | 0.1106 | 97.8% | 0.0812 | 98.0% | patience 3/10 |
| 26 | 0.0667 | **98.6%** | **0.0765** | 98.4% | ✅ NEW BEST |
| 27 | 0.0941 | 98.4% | 0.1182 | 98.0% | patience 1/10 |
| 28 | 0.0858 | 98.0% | 0.1165 | 97.9% | patience 2/10 |
| 29 | 0.0875 | 98.1% | 0.1051 | 98.1% | patience 3/10 |
| 30 | 0.0887 | 98.5% | 0.0925 | 97.8% | patience 4/10 |
| 31 | 0.0780 | 98.1% | 0.1102 | 98.1% | patience 5/10 |
| 32 | 0.0761 | **98.6%** | 0.0855 | 98.4% | patience 6/10 |

---

## 6. 2차 학습 결과 (새 데스크탑, 2026-05-30)

### 환경 차이 및 시행착오
- 새 데스크탑에서 WSL2 venv 없음 → `python3.12-venv` 설치 후 새로 구성
- `pkg_resources` 미포함 오류 → `setuptools==69.5.1` 다운그레이드로 해결
- tmux 세션 불안정 → Ubuntu 터미널에서 직접 실행으로 해결

### 주요 epoch 기록

| Epoch | Val Loss | Val Acc | 비고 |
|-------|----------|---------|------|
| 9 | 0.1349 | 96.4% | best |
| 19 | 0.1298 | 97.5% | best |
| 27 | 0.1213 | 97.7% | best |
| 33 | 0.0942 | 98.2% | best |
| 43 | 0.0800 | 98.2% | best |
| 51 | 0.0539 | 98.5% | best |
| **60** | **0.0572** | **98.5%** | |
| 61 | 0.1573 | 97.7% | Early Stopping 발동 |

### 최종 결과

| 항목 | 값 |
|------|-----|
| Early stopping epoch | **61** |
| 최종 best val_loss | **0.0539** (epoch 51) |
| **Test accuracy** | **96.11%** |
| 학습 소요 시간 | 약 **1시간 45분** (16:39 ~ 18:21) |
| TFLite INT8 | `models/adas_detector.tflite` **(0.67 MB)** ✅ |
| Classifier | `models/custom_classifier_finetuned.h5` (2.7 MB) ✅ |

## 7. 다음 단계

- [ ] Android 앱 모델 교체: `SoundDetectorApp/app/src/main/assets/model.tflite`
- [ ] 실기기 테스트
- [ ] 시행착오 PDF 최종 작성
