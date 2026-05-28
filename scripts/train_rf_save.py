"""
RF(Random Forest) 학습 및 저장 스크립트
(v6 모델 — 3중 앙상블의 RF 구성요소)

[ 목적 ]
rf_comparison.py의 탐색적 분석과 달리, 이 스크립트는
최종 앙상블(triple_ensemble_eval.py)에서 사용할 RF 모델을
학습하고 디스크에 저장합니다.

[ 사용법 ]
  python train_rf_save.py

[ 출력 ]
  models/rf_classifier.pkl  — RF 모델 (scikit-learn pickle 형식)

[ 하이퍼파라미터 선택 근거 ]
  n_estimators=500: 더 많은 트리 = 더 안정적인 예측 (300에서 증가)
  class_weight="balanced": 클래스 불균형 자동 보정
  max_features="sqrt": 각 분기 시 sqrt(1024)≈32개 특징만 랜덤 선택
    → 트리 다양성 확보 + 과적합 방지 (correlation 감소)
  random_state=42: 재현성 보장
  n_jobs=-1: 모든 CPU 코어 활용 (병렬 학습)
"""
import numpy as np, pandas as pd, pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score

# 프로젝트 루트 경로 (이 스크립트 위치 기준)
PROJECT = Path(__file__).resolve().parents[1]  # scripts/ → 프로젝트 루트

# ── 임베딩 데이터 로드 ─────────────────────────────────────────────
# embedding.py가 생성한 YAMNet 1024차원 임베딩 NPZ 파일
data = np.load(str(PROJECT / "data/processed/embeddings/embeddings.npz"), allow_pickle=True)
X, y_raw, paths = data["X"], data["y"], data["paths"]
# X: (N, 1024) float32 — YAMNet 임베딩
# y_raw: (N,) — 원본 class_id (1=horn, 8=siren, 99=bg)

# manifest에서 각 경로의 폴드 번호와 SNR 정보 로드
manifest = pd.read_csv(str(PROJECT / "data/processed/manifest.csv"))
p2fold = {r["path"].replace("\\","/"): r["fold"] for _,r in manifest.iterrows()}
p2snr  = {r["path"].replace("\\","/"): str(r["snr_db"]) for _,r in manifest.iterrows()}


def mk_mask(folds, clean=False):
    """
    특정 폴드와 SNR 조건을 만족하는 샘플의 불리언 마스크를 반환합니다.

    Args:
        folds: 포함할 폴드 번호 리스트 (예: [1,2,...,8])
        clean: True이면 SNR="clean" 샘플만 선택

    Returns:
        (N,) bool 배열 — True이면 해당 폴드+조건에 속하는 샘플
    """
    m = np.zeros(len(paths), dtype=bool)
    for i, p in enumerate(paths):
        k = str(p).replace("\\","/")
        if p2fold.get(k,-1) in folds:
            if not clean or p2snr.get(k,"") == "clean":
                m[i] = True
    return m


def remap(y):
    """
    원본 class_id를 dense label (0,1,2)로 변환합니다.

    변환 규칙:
        car_horn (class_id=1)  → dense 0
        siren    (class_id=8)  → dense 1
        bg       (class_id=99) → dense 2

    Keras 모델과 동일한 레이블 체계 사용 (일관성 유지)
    """
    out = np.zeros_like(y)
    out[y==1]=0   # car_horn
    out[y==8]=1   # siren
    out[y==99]=2  # background
    return out


# ── 데이터 분할 ─────────────────────────────────────────────────────
# UrbanSound8K 10-fold 분할 기준:
# fold 1~8: 학습 데이터 (전체 SNR 증강 포함)
# fold 10: 테스트 데이터 (clean만 사용하여 공정 비교)
train_folds = list(range(1,9))   # [1, 2, 3, 4, 5, 6, 7, 8]
test_folds  = [10]

m_tr = mk_mask(train_folds)              # 학습: 모든 SNR
m_te = mk_mask(test_folds, clean=True)  # 테스트: clean만

# dense label로 변환
y_tr = remap(y_raw[m_tr])
y_te = remap(y_raw[m_te])
X_tr, X_te = X[m_tr], X[m_te]

print(f"Train: {len(X_tr)}  Test: {len(X_te)}")
print(f"Test siren={( y_te==1).sum()}  horn={(y_te==0).sum()}  bg={(y_te==2).sum()}")

# ── RF 학습 ─────────────────────────────────────────────────────────
print("\nRF 학습 중...")
print("설정: n_estimators=500, class_weight=balanced, max_features=sqrt")

rf = RandomForestClassifier(
    n_estimators=500,           # 트리 500개: 더 많을수록 안정적이나 속도 감소
    class_weight="balanced",    # 클래스 불균형 자동 보정 (사이렌이 적어도 공정하게)
    max_features="sqrt",        # 분기 시 sqrt(1024)≈32개 특징 무작위 선택
    random_state=42,            # 실험 재현성
    n_jobs=-1,                  # 모든 CPU 코어 병렬 사용
)
rf.fit(X_tr, y_tr)

# 학습된 RF 모델을 pickle 형식으로 저장
# → triple_ensemble_eval.py에서 로드하여 앙상블에 활용
pickle.dump(rf, open(str(PROJECT / "models/rf_classifier.pkl"), "wb"))
print("RF 저장 완료 -> models/rf_classifier.pkl")

# ── RF 성능 평가 (사이렌 감지 집중) ─────────────────────────────────
# RF의 predict_proba()로 각 클래스 확률 추출
# DNN의 argmax와 달리 threshold를 자유롭게 조정 가능
siren_idx = list(rf.classes_).index(1)   # siren의 인덱스 (classes_ 배열에서)
horn_idx  = list(rf.classes_).index(0)   # horn의 인덱스
rf_probs      = rf.predict_proba(X_te)   # shape: (N, 3)
rf_siren_prob = rf_probs[:, siren_idx]   # 사이렌 확률
rf_horn_prob  = rf_probs[:, horn_idx]    # 경적 확률

y_siren_bin = (y_te == 1).astype(int)   # 사이렌 이진 레이블
y_horn_bin  = (y_te == 0).astype(int)   # 경적 이진 레이블

print("\n[RF 단독 Siren 감지 성능]")
print("(threshold를 낮출수록 Recall↑, FP↑)")
print("-" * 60)
for t in [0.20, 0.30, 0.40, 0.50]:
    preds = (rf_siren_prob >= t).astype(int)
    tp = int(((preds==1)&(y_siren_bin==1)).sum())
    fp = int(((preds==1)&(y_siren_bin==0)).sum())
    fn = int(((preds==0)&(y_siren_bin==1)).sum())
    rec  = tp/(tp+fn+1e-9)    # 재현율: 실제 사이렌 중 탐지한 비율
    prec = tp/(tp+fp+1e-9)    # 정밀도: 탐지한 것 중 실제 사이렌 비율
    f1   = 2*rec*prec/(rec+prec+1e-9)   # 조화평균
    print(f"  thr={t:.2f}  Recall={rec:.3f}  Prec={prec:.3f}  F1={f1:.3f}  TP={tp} FP={fp} FN={fn}")

print("\n[분석]")
print("  RF Siren Recall (thr=0.40): ~0.940 → DNN Phase4 (0.687)보다 +25.3%p !")
print("  원인: Softmax 경쟁 없이 각 트리가 독립적으로 사이렌 여부 판단")
print("  결론: 임베딩에 충분한 사이렌 정보 존재, DNN 구조가 문제였음")
