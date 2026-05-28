"""
Ensemble Evaluation — 3-class DNN + Siren Specialist 앙상블 평가
(v7 앙상블의 2중 버전: DNN + Specialist)

[ 앙상블이란? ]
여러 모델의 예측을 결합하여 단일 모델보다 나은 성능을 얻는 기법.
각 모델이 다른 방식으로 틀리므로, 결합 시 오류가 상쇄되는 효과.

[ 2중 앙상블 전략 비교 ]
  1. OR rule  : DNN≥threshold1 OR Specialist≥threshold2
                → 둘 중 하나라도 사이렌이면 경보 (최고 Recall, FP 많음)
  2. AVG rule : (p_DNN + p_Specialist) / 2 로 평균 확률 계산
                → Recall과 Precision의 균형
  3. MAX rule : max(p_DNN, p_Specialist) 로 더 확신하는 값 사용
                → AVG와 비슷하지만 더 공격적

[ 사용법 ]
  python3 -m src.ensemble_eval --config config.yaml
"""
from __future__ import annotations
import logging, sys, warnings, pickle
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def classification_block(y_true_bin, y_prob, threshold, label):
    """
    단일 threshold에서 이진 분류 성능 지표를 계산하고 출력합니다.

    [ 반환 지표 설명 ]
    - TP (True Positive): 실제 사이렌을 사이렌으로 정확히 탐지
    - FP (False Positive): 배경/경적을 사이렌으로 오탐 (오경보)
    - FN (False Negative): 실제 사이렌을 놓침 (가장 위험)
    - TN (True Negative): 비사이렌을 올바르게 비사이렌으로 분류
    - Recall: TP/(TP+FN) — 실제 사이렌 중 탐지 비율 (핵심 지표!)
    - Precision: TP/(TP+FP) — 탐지한 것 중 실제 사이렌 비율
    - F1: 2×Recall×Precision/(Recall+Precision) — 조화평균

    Args:
        y_true_bin: 실제 이진 레이블 (siren=1, non-siren=0)
        y_prob: 예측 사이렌 확률 (0~1)
        threshold: 이 값 이상이면 사이렌으로 판단
        label: 출력에 표시할 설명 문자열

    Returns:
        성능 지표 딕셔너리
    """
    preds = (y_prob >= threshold).astype(int)
    tp = int(((preds==1)&(y_true_bin==1)).sum())
    fp = int(((preds==1)&(y_true_bin==0)).sum())
    fn = int(((preds==0)&(y_true_bin==1)).sum())
    tn = int(((preds==0)&(y_true_bin==0)).sum())
    rec  = tp/(tp+fn+1e-9)
    prec = tp/(tp+fp+1e-9)
    f1   = 2*rec*prec/(rec+prec+1e-9)
    log.info("  %-40s Recall=%.3f  Prec=%.3f  F1=%.3f  TP=%d FP=%d FN=%d",
             label, rec, prec, f1, tp, fp, fn)
    return {"label": label, "threshold": threshold,
            "recall": rec, "precision": prec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def evaluate(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    DNN + Siren Specialist 2중 앙상블 평가를 실행합니다.

    [ 평가 흐름 ]
    1. embeddings.npz에서 test 세트 임베딩 추출
    2. 3-class DNN으로 사이렌 확률 추출 (probs[:,1])
    3. Siren Specialist로 사이렌 확률 추출 (sigmoid 출력)
    4. 여러 앙상블 전략 비교:
       - DNN 단독
       - Specialist 단독
       - AVG(DNN + Specialist)
       - MAX(DNN, Specialist)
       - OR 규칙
    5. 최고 Recall 설정 강조 출력

    Args:
        config_path: config.yaml 경로

    Returns:
        각 전략/threshold 조합 성능 DataFrame
    """
    import tensorflow as tf
    from src.utils.config import load_config

    cfg = load_config(config_path)

    # ── 임베딩 로드 ──────────────────────────────────────────────
    emb_path = PROJECT / cfg["yamnet"]["cache_dir"] / "embeddings.npz"
    data = np.load(str(emb_path), allow_pickle=True)
    X, y_raw, paths = data["X"], data["y"], data["paths"]

    # 경로 기반으로 폴드/SNR 매핑
    manifest = pd.read_csv(str(PROJECT / cfg["dataset"]["processed_dir"] / "manifest.csv"))
    p2fold = {r["path"].replace("\\","/"): r["fold"] for _, r in manifest.iterrows()}
    p2snr  = {r["path"].replace("\\","/"): str(r["snr_db"]) for _, r in manifest.iterrows()}

    # test 세트 마스크: fold 10 + clean 조건
    m_te = np.zeros(len(paths), dtype=bool)
    for i, p in enumerate(paths):
        k = str(p).replace("\\","/")
        if p2fold.get(k,-1) in cfg["dataset"]["test_folds"]:
            if p2snr.get(k,"") == "clean":
                m_te[i] = True

    X_te = X[m_te]
    y_te = y_raw[m_te]
    y_siren_bin = (y_te == 8).astype(np.float32)   # 사이렌 이진 레이블
    y_3class = np.where(y_te==1, 0, np.where(y_te==8, 1, 2))  # 3-class 레이블

    log.info("Test: %d samples (siren=%d, car_horn=%d, bg=%d)",
             len(y_te), (y_te==8).sum(), (y_te==1).sum(), (y_te==99).sum())

    # ── 3-class DNN 로드 및 예측 ─────────────────────────────────
    clf3_path = PROJECT / "models" / "custom_classifier_finetuned.h5"
    log.info("3-class 분류기 로드: %s", clf3_path.name)
    clf3 = tf.keras.models.load_model(str(clf3_path))
    probs3 = clf3.predict(X_te, verbose=0)  # shape: (N, 3) — car/siren/bg
    prob_siren_3class = probs3[:, 1]        # 사이렌(인덱스 1) 확률만 추출

    # ── Siren Specialist 로드 및 예측 ────────────────────────────
    spec_path   = PROJECT / "models" / "siren_specialist.h5"
    scaler_path = PROJECT / "models" / "siren_specialist_scaler.pkl"
    thresh_path = PROJECT / "models" / "siren_threshold.txt"

    log.info("Siren Specialist 로드: %s", spec_path.name)
    specialist = tf.keras.models.load_model(str(spec_path), compile=False)
    with open(str(scaler_path), "rb") as f:
        scaler = pickle.load(f)
    best_thresh = float(thresh_path.read_text().strip())  # val 기준 최적 threshold
    log.info("최적 threshold: %.2f", best_thresh)

    # Specialist 입력은 StandardScaler로 정규화 필요
    X_te_sc = scaler.transform(X_te)
    prob_siren_spec = specialist.predict(X_te_sc, verbose=0).flatten()

    # ── 앙상블 확률 계산 ─────────────────────────────────────────
    # AVG: 두 모델 확률의 단순 평균
    prob_avg = 0.5 * prob_siren_3class + 0.5 * prob_siren_spec
    # MAX: 두 모델 중 더 확신하는 쪽 선택
    prob_max = np.maximum(prob_siren_3class, prob_siren_spec)

    # ── 각 전략별 성능 평가 ─────────────────────────────────────
    results = []
    log.info("="*70)
    log.info("SIREN RECALL 비교")
    log.info("="*70)

    # [기준선] 3-class DNN 단독 성능
    log.info("\n[기준선 — 3-class DNN 단독]")
    for t in [0.50, 0.20]:
        results.append(classification_block(
            y_siren_bin, prob_siren_3class, t,
            f"3-class DNN (threshold={t:.2f})"))

    # [비교] Siren Specialist 단독 성능
    log.info("\n[Siren Specialist 단독]")
    for t in [0.50, best_thresh, 0.15, 0.10]:
        results.append(classification_block(
            y_siren_bin, prob_siren_spec, t,
            f"Specialist (threshold={t:.2f})"))

    # [앙상블] AVG 전략
    log.info("\n[앙상블: AVG (0.5×DNN + 0.5×Specialist)]")
    for t in [0.30, 0.20, 0.15]:
        results.append(classification_block(
            y_siren_bin, prob_avg, t,
            f"Ensemble AVG (threshold={t:.2f})"))

    # [앙상블] MAX 전략
    log.info("\n[앙상블: MAX (둘 중 높은 값)]")
    for t in [0.30, 0.20, 0.15]:
        results.append(classification_block(
            y_siren_bin, prob_max, t,
            f"Ensemble MAX (threshold={t:.2f})"))

    # [앙상블] OR 규칙
    # DNN이 조금이라도 사이렌 같다거나(≥0.20) Specialist가 확신하면(≥0.15) 탐지
    # → 가장 공격적인 전략, FP 가장 많지만 FN 가장 적음
    log.info("\n[앙상블: OR rule — DNN≥0.20 OR Specialist≥0.15]")
    preds_or = ((prob_siren_3class >= 0.20) | (prob_siren_spec >= 0.15)).astype(int)
    tp = int(((preds_or==1)&(y_siren_bin==1)).sum())
    fp = int(((preds_or==1)&(y_siren_bin==0)).sum())
    fn = int(((preds_or==0)&(y_siren_bin==1)).sum())
    rec  = tp/(tp+fn+1e-9); prec = tp/(tp+fp+1e-9)
    f1   = 2*rec*prec/(rec+prec+1e-9)
    log.info("  %-40s Recall=%.3f  Prec=%.3f  F1=%.3f  TP=%d FP=%d FN=%d",
             "OR rule (DNN≥0.20 | Spec≥0.15)", rec, prec, f1, tp, fp, fn)
    results.append({"label":"OR rule","threshold":"0.20|0.15",
                    "recall":rec,"precision":prec,"f1":f1,
                    "tp":tp,"fp":fp,"fn":fn,"tn":0})

    # ── 앙상블 적용 후 전체 3-class 정확도 ──────────────────────
    # AVG 앙상블 확률이 threshold 이상이면 사이렌으로 오버라이드
    log.info("\n[전체 정확도 — 앙상블 AVG threshold=0.20]")
    preds_3 = np.argmax(probs3, axis=1)    # DNN argmax 예측
    preds_ens = preds_3.copy()
    preds_ens[prob_avg >= 0.20] = 1         # 앙상블이 사이렌이라면 → 사이렌으로 덮어쓰기
    from sklearn.metrics import classification_report
    rpt = classification_report(y_3class, preds_ens,
                                 target_names=["car_horn","siren","background"],
                                 zero_division=0)
    log.info("\n%s", rpt)

    # ── 최종 요약 출력 ───────────────────────────────────────────
    df = pd.DataFrame(results)
    print("\n" + "="*75)
    print("  앙상블 최종 요약")
    print("="*75)
    print(df[["label","threshold","recall","precision","f1","tp","fp","fn"]]
            .to_string(index=False))

    # 최고 Recall 설정 강조 표시
    best_row = df.loc[df["recall"].idxmax()]
    print(f"\n★ 최고 Recall 설정: {best_row['label']}")
    print(f"  Recall={best_row['recall']:.3f}  Precision={best_row['precision']:.3f}")

    return df


if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    os.chdir(PROJECT)
    evaluate(args.config)
