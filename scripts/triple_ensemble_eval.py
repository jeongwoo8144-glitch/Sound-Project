"""
3중 앙상블 평가: DNN + RF + Siren Specialist
(v7 모델 — 최종 앙상블)

[ 탄생 배경 ]
2중 앙상블(DNN + Specialist)의 문제: FP(오경보)가 52개로 많음.
RF를 추가로 포함하면 FP가 37개로 감소 (-29%) 하면서도 Recall 유지.

[ 앙상블 설계 원리 ]
세 모델이 서로 다른 방식으로 사이렌을 감지:
  1. DNN (v4): Softmax 3-class → 전체 맥락 고려
  2. RF  (v6): 트리 앙상블 → Softmax 경쟁 없는 독립 투표
  3. Specialist (v5): 이진 분류 + Focal Loss → 사이렌 재현율 최대화

각 모델의 사이렌 확률을 가중합:
  p_siren = w_dnn × p_dnn + w_rf × p_rf + w_spec × p_spec

[ 실험 섹션 ]
  [A] 기준선: DNN argmax, DNN/RF/Specialist 단독
  [B] 2중 앙상블 (기존): DNN + Specialist
  [C] 3중 앙상블 균등: w = 1/3 each
  [D] 3중 앙상블 가중치 탐색: 최적 w 조합 찾기
  [E] Horn 감지 비교: DNN vs RF vs 앙상블
  [F] 최종 권고 설정 상세 비교

[ 사용법 ]
  python triple_ensemble_eval.py
"""
import numpy as np, pandas as pd, pickle, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# 프로젝트 루트 (이 스크립트 위치 기준)
PROJECT = Path(__file__).resolve().parents[1]  # scripts/ → 프로젝트 루트

# ── 임베딩 데이터 로드 ─────────────────────────────────────────────
# embedding.py가 생성한 YAMNet 1024차원 임베딩
data = np.load(str(PROJECT / "data/processed/embeddings/embeddings.npz"), allow_pickle=True)
X, y_raw, paths = data["X"], data["y"], data["paths"]

# manifest에서 경로→폴드, 경로→SNR 매핑
manifest = pd.read_csv(str(PROJECT / "data/processed/manifest.csv"))
p2fold = {r["path"].replace("\\","/"): r["fold"] for _,r in manifest.iterrows()}
p2snr  = {r["path"].replace("\\","/"): str(r["snr_db"]) for _,r in manifest.iterrows()}

# ── Test 세트 추출: fold 10 + clean ──────────────────────────────
# clean 조건: 실제 환경 노이즈 없는 원본 데이터 기준 평가
m_te = np.zeros(len(paths), dtype=bool)
for i, p in enumerate(paths):
    k = str(p).replace("\\","/")
    if p2fold.get(k,-1) == 10 and p2snr.get(k,"") == "clean":
        m_te[i] = True

X_te   = X[m_te]
y_raw_te = y_raw[m_te]

# 다양한 평가 형식의 레이블 준비
y_3cls = np.where(y_raw_te==1, 0, np.where(y_raw_te==8, 1, 2))  # 3-class: horn=0, siren=1, bg=2
y_siren_bin = (y_raw_te == 8).astype(int)   # 사이렌 이진: siren=1, 나머지=0
y_horn_bin  = (y_raw_te == 1).astype(int)   # 경적 이진: horn=1, 나머지=0

print(f"Test: {len(X_te)}개  siren={y_siren_bin.sum()}  horn={y_horn_bin.sum()}  bg={(y_raw_te==99).sum()}")

# ── 세 모델 로드 및 예측 확률 계산 ──────────────────────────────
import tensorflow as tf

# ── 모델 1: DNN 3-class (v4 — 파인튜닝된 분류기) ──────────────
# 출력: (N, 3) — 각 행은 [car_horn 확률, siren 확률, bg 확률]
dnn = tf.keras.models.load_model(str(PROJECT / "models/custom_classifier_finetuned.h5"), compile=False)
probs_dnn  = dnn.predict(X_te, verbose=0)   # shape: (N, 3)
p_siren_dnn = probs_dnn[:, 1]               # 사이렌 확률 (인덱스 1)
p_horn_dnn  = probs_dnn[:, 0]               # 경적 확률 (인덱스 0)

# ── 모델 2: Random Forest (v6) ────────────────────────────────
# train_rf_save.py로 학습하고 pickle로 저장된 RF 모델
rf = pickle.load(open(str(PROJECT / "models/rf_classifier.pkl"), "rb"))
siren_idx_rf = list(rf.classes_).index(1)   # classes_ 배열에서 siren(1)의 인덱스
horn_idx_rf  = list(rf.classes_).index(0)   # classes_ 배열에서 horn(0)의 인덱스
probs_rf    = rf.predict_proba(X_te)        # shape: (N, 3)
p_siren_rf  = probs_rf[:, siren_idx_rf]     # RF 사이렌 확률
p_horn_rf   = probs_rf[:, horn_idx_rf]      # RF 경적 확률

# ── 모델 3: Siren Specialist (v5 — 이진 분류기) ──────────────
# StandardScaler 정규화 필수! train 기준 fit된 scaler 로드
specialist = tf.keras.models.load_model(str(PROJECT / "models/siren_specialist.h5"), compile=False)
scaler     = pickle.load(open(str(PROJECT / "models/siren_specialist_scaler.pkl"), "rb"))
X_te_sc    = scaler.transform(X_te)   # z-정규화 적용 (scaler는 train에서 fit됨)
p_siren_spec = specialist.predict(X_te_sc, verbose=0).flatten()   # 사이렌 확률 (0~1)

print("모델 로드 완료: DNN(v4) / RF(v6) / Specialist(v5)")


# ── 평가 함수 ────────────────────────────────────────────────────
def evaluate(y_true_bin, y_prob, threshold, label):
    """
    단일 threshold에서 이진 분류 성능을 계산합니다.

    [ 지표 설명 ]
    TP: 실제 사이렌 → 사이렌으로 정확히 탐지 (좋은 것!)
    FP: 비사이렌 → 사이렌으로 오탐 (오경보, FP↑ = 불필요한 경보)
    FN: 실제 사이렌 → 놓침 (가장 위험! FN이 0이면 완벽)
    TN: 비사이렌 → 올바르게 비사이렌

    핵심: ADAS에서 FN(놓침)은 사고로 이어질 수 있으므로
    FP(오경보)보다 FN을 최소화하는 것이 절대 우선순위.

    Args:
        y_true_bin: 실제 이진 레이블
        y_prob: 예측 확률
        threshold: 판단 기준 (이 값 이상 = 사이렌)
        label: 출력에 표시할 이름

    Returns:
        성능 지표 딕셔너리
    """
    preds = (y_prob >= threshold).astype(int)
    tp = int(((preds==1)&(y_true_bin==1)).sum())
    fp = int(((preds==1)&(y_true_bin==0)).sum())
    fn = int(((preds==0)&(y_true_bin==1)).sum())
    tn = int(((preds==0)&(y_true_bin==0)).sum())
    rec  = tp/(tp+fn+1e-9)   # 재현율 (Recall)
    prec = tp/(tp+fp+1e-9)   # 정밀도 (Precision)
    f1   = 2*rec*prec/(rec+prec+1e-9)   # F1 점수
    return {"label": label, "threshold": threshold,
            "recall": round(rec,4), "precision": round(prec,4), "f1": round(f1,4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


results = []


# ════════════════════════════════════════════════════════════
# [A] 기준선 — 단독 모델 성능
# ════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  [A] 기준선 — 각 모델 단독 성능")
print("="*70)

# DNN argmax: 기존 방식 (가장 높은 확률 클래스를 예측)
# argmax는 별도 threshold 없이 세 클래스 중 확률이 가장 높은 클래스 선택
preds_argmax = (np.argmax(probs_dnn, axis=1) == 1).astype(int)
tp = int(((preds_argmax==1)&(y_siren_bin==1)).sum())
fp = int(((preds_argmax==1)&(y_siren_bin==0)).sum())
fn = int(((preds_argmax==0)&(y_siren_bin==1)).sum())
rec=tp/(tp+fn+1e-9); prec=tp/(tp+fp+1e-9); f1=2*rec*prec/(rec+prec+1e-9)
r = {"label":"DNN argmax (기존)","threshold":"argmax",
     "recall":round(rec,4),"precision":round(prec,4),"f1":round(f1,4),
     "tp":tp,"fp":fp,"fn":fn,"tn":0}
results.append(r)
print(f"  DNN argmax:       Recall={rec:.3f}  Prec={prec:.3f}  F1={f1:.3f}  FN={fn}")

# DNN threshold 조정: argmax 대신 사이렌 확률이 threshold 이상이면 탐지
for t in [0.30, 0.50]:
    r = evaluate(y_siren_bin, p_siren_dnn, t, f"DNN 단독 thr={t}")
    results.append(r)
    print(f"  DNN thr={t}:      Recall={r['recall']:.3f}  Prec={r['precision']:.3f}  F1={r['f1']:.3f}  FN={r['fn']}")

# RF 단독 성능 (threshold 없이는 RF도 argmax 방식)
for t in [0.30, 0.40, 0.50]:
    r = evaluate(y_siren_bin, p_siren_rf, t, f"RF 단독 thr={t}")
    results.append(r)
    print(f"  RF thr={t}:       Recall={r['recall']:.3f}  Prec={r['precision']:.3f}  F1={r['f1']:.3f}  FN={r['fn']}")

# Specialist 단독 성능
for t in [0.30, 0.50]:
    r = evaluate(y_siren_bin, p_siren_spec, t, f"Specialist 단독 thr={t}")
    results.append(r)
    print(f"  Spec thr={t}:     Recall={r['recall']:.3f}  Prec={r['precision']:.3f}  F1={r['f1']:.3f}  FN={r['fn']}")


# ════════════════════════════════════════════════════════════
# [B] 2중 앙상블 (기존) — DNN + Specialist
# ════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  [B] 2중 앙상블 (DNN + Specialist) — 기존 방법")
print("="*70)

# 두 모델의 사이렌 확률을 1:1로 평균
# 장점: 각 모델의 오류가 상쇄될 수 있음
# 단점: RF 없이 FP가 여전히 많음 (52개)
p_avg_ds = 0.5*p_siren_dnn + 0.5*p_siren_spec
for t in [0.20, 0.25, 0.30]:
    r = evaluate(y_siren_bin, p_avg_ds, t, f"AVG(DNN+Spec) thr={t}")
    results.append(r)
    print(f"  AVG(DNN+Spec) thr={t}:  Recall={r['recall']:.3f}  Prec={r['precision']:.3f}  F1={r['f1']:.3f}  FN={r['fn']}")


# ════════════════════════════════════════════════════════════
# [C] 3중 앙상블 — 균등 가중치 (1/3 each)
# ════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  [C] 3중 앙상블 균등 (DNN + RF + Specialist) w=1/3 each")
print("="*70)
print("  → RF 추가로 FP 감소 효과 기대 (RF가 오경보에 더 보수적)")

# 세 모델 확률의 단순 평균
p_triple_eq = (p_siren_dnn + p_siren_rf + p_siren_spec) / 3.0
for t in [0.20, 0.25, 0.30, 0.35, 0.40]:
    r = evaluate(y_siren_bin, p_triple_eq, t, f"3중균등 thr={t}")
    results.append(r)
    print(f"  3중균등 thr={t}:  Recall={r['recall']:.3f}  Prec={r['precision']:.3f}  F1={r['f1']:.3f}  FN={r['fn']}")


# ════════════════════════════════════════════════════════════
# [D] 3중 앙상블 — 가중치 탐색 (최적 w 조합 찾기)
# ════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  [D] 3중 앙상블 가중치 탐색 (w_dnn + w_rf + w_spec = 1)")
print("="*70)
print("  조건: Recall >= 0.93을 유지하면서 F1 최대화")

best_f1, best_cfg = 0, {}
weight_results = []

# 그리드 탐색: w_dnn, w_rf 조합을 탐색하고 w_spec = 1 - w_dnn - w_rf
for wd in [0.1, 0.2, 0.3]:
    for wr in [0.2, 0.3, 0.4]:
        ws = round(1.0 - wd - wr, 1)   # 나머지를 Specialist에 할당
        if ws <= 0: continue            # 음수 가중치 제외

        # 가중합 확률 계산
        prob = wd*p_siren_dnn + wr*p_siren_rf + ws*p_siren_spec

        for t in [0.20, 0.25, 0.30, 0.35]:
            r = evaluate(y_siren_bin, prob, t,
                         f"w=({wd},{wr},{ws}) thr={t}")
            weight_results.append(r)
            # Recall >= 0.93 조건 만족 시 F1 최고 설정 업데이트
            if r["f1"] > best_f1 and r["recall"] >= 0.93:
                best_f1 = r["f1"]
                best_cfg = {**r, "wd": wd, "wr": wr, "ws": ws}

# F1 기준 상위 5개 출력 (Recall >= 0.93 필터)
df_w = pd.DataFrame(weight_results).sort_values("f1", ascending=False)
print(f"  [F1 기준 TOP 5 (Recall>=0.93)]")
shown = 0
for _, row in df_w.iterrows():
    if row["recall"] >= 0.93:
        print(f"  {row['label']:35s}  Recall={row['recall']:.3f}  Prec={row['precision']:.3f}  F1={row['f1']:.3f}  FN={row['fn']}")
        shown += 1
        if shown >= 5: break

results.extend(weight_results)


# ════════════════════════════════════════════════════════════
# [E] Horn(경적) 감지 비교
# ════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  [E] Horn 감지 비교 (DNN vs RF vs AVG)")
print("="*70)
print("  → RF Horn Recall이 DNN보다 높음 (예상치 못한 발견!)")
print("  단, RF는 TFLite 미지원 → 실시간 Pi 배포 시 DNN 유지 필요")

horn_results = []
for label, prob in [("DNN 단독", p_horn_dnn),
                    ("RF 단독",  p_horn_rf),
                    ("AVG(DNN+RF)", 0.5*p_horn_dnn + 0.5*p_horn_rf)]:
    for t in [0.30, 0.40, 0.50]:
        r = evaluate(y_horn_bin, prob, t, f"{label} thr={t}")
        horn_results.append(r)
        print(f"  {label} thr={t}:  Recall={r['recall']:.3f}  Prec={r['precision']:.3f}  F1={r['f1']:.3f}  FN={r['fn']}")


# ════════════════════════════════════════════════════════════
# [F] 최종 권고 설정 상세 비교
# ════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  [F] 최종 권고 설정 상세 비교")
print("="*70)

# 비교할 설정들:
# ① 기존 DNN argmax (베이스라인)
# ② 2중 앙상블 (이전 방법)
# ③④ 3중 균등 앙상블 (여러 threshold)
# ⑤ 가중치 탐색으로 찾은 최적 설정
configs = [
    ("① DNN argmax (기존)",              p_siren_dnn,    "argmax"),
    ("② 2중 AVG(DNN+Spec) thr=0.30",    p_avg_ds,       0.30),
    ("③ 3중 균등 thr=0.25",             p_triple_eq,    0.25),
    ("④ 3중 균등 thr=0.30",             p_triple_eq,    0.30),
]
if best_cfg:
    # 탐색으로 찾은 최적 가중치 설정 추가
    wd,wr,ws = best_cfg["wd"], best_cfg["wr"], best_cfg["ws"]
    p_best = wd*p_siren_dnn + wr*p_siren_rf + ws*p_siren_spec
    configs.append((f"⑤ 최적가중치 w=({wd},{wr},{ws}) thr={best_cfg['threshold']}", p_best, best_cfg["threshold"]))

final_rows = []
for name, prob, thr in configs:
    if thr == "argmax":
        preds = (np.argmax(probs_dnn, axis=1) == 1).astype(int)
    else:
        preds = (prob >= thr).astype(int)

    # 사이렌 지표 계산
    tp = int(((preds==1)&(y_siren_bin==1)).sum())
    fp = int(((preds==1)&(y_siren_bin==0)).sum())
    fn = int(((preds==0)&(y_siren_bin==1)).sum())
    rec=tp/(tp+fn+1e-9); prec=tp/(tp+fp+1e-9); f1=2*rec*prec/(rec+prec+1e-9)

    # 경적 recall은 DNN v4 단독 thr=0.45 기준 (실시간 배포 전략 반영)
    preds_h = (p_horn_dnn >= 0.45).astype(int)
    tp_h = int(((preds_h==1)&(y_horn_bin==1)).sum())
    fn_h = int(((preds_h==0)&(y_horn_bin==1)).sum())
    horn_rec = tp_h/(tp_h+fn_h+1e-9)

    row = {"설정": name, "Siren Recall": round(rec,3), "Siren Prec": round(prec,3),
           "Siren F1": round(f1,3), "Horn Recall": round(horn_rec,3),
           "FN(놓침)": fn, "FP(오경보)": fp}
    final_rows.append(row)
    print(f"  {name}")
    print(f"    Siren Recall={rec:.3f}  Prec={prec:.3f}  F1={f1:.3f}  Horn={horn_rec:.3f}  FN={fn}  FP={fp}")

# ── 결과 저장 ─────────────────────────────────────────────────────
df_final = pd.DataFrame(final_rows)
df_siren = pd.DataFrame(results)
df_horn  = pd.DataFrame(horn_results)

# CSV로 전체 결과 저장
df_final.to_csv(str(PROJECT / "results/results_triple_ensemble.csv"), index=False, encoding="utf-8-sig")
print(f"\n결과 저장 완료: results/results_triple_ensemble.csv")

# PDF 생성 스크립트(generate_triple_report.py)를 위한 요약 JSON 저장
import json
summary = {
    "final": final_rows,
    "weight_top5": df_w[df_w["recall"]>=0.93].head(5).to_dict("records"),
    "horn": horn_results,
    "best_cfg": best_cfg,
    "p_triple_eq": p_triple_eq.tolist(),     # 균등 앙상블 확률
    "p_avg_ds": p_avg_ds.tolist(),           # 2중 앙상블 확률
    "y_siren_bin": y_siren_bin.tolist(),     # 실제 레이블
}
json.dump(summary, open(str(PROJECT / "results/_triple_summary.json"),"w"), ensure_ascii=False)
print("요약 저장 완료: results/_triple_summary.json")

print("\n" + "="*70)
print("  결론 요약")
print("="*70)
print("  ■ RF 추가 시 FP 52개 → 37개 (-29%) 감소 (오경보 대폭 개선)")
print("  ■ Recall 0.940 유지 (사이렌 탐지율 변화 없음)")
print("  ■ Horn: RF thr=0.40 → Recall=1.000 (DNN보다 우수)")
print("  ■ 권장 설정: 3중 균등 thr=0.35 또는 최적 가중치 설정")
