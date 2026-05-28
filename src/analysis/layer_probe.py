"""
Layer Probing Analysis — 레이어별 임베딩 판별력 분석

[ 교수 피드백 ]
  "레이어별로 임베딩값 뽑아서 사이렌/경적을 구분하는 레이어 찾기"
  "정확도가 확 차이나는 레이어 찾아서 부분을 해제"

[ 분석 목적 ]
YAMNet의 14개 MobileNet 블록 중 어느 블록이 사이렌/경적 판별에
가장 중요한지 파악하여, 파인튜닝 시 어느 레이어부터 해제할지 결정.

[ 방법론: Linear Probe (선형 프로브) ]
YAMNet의 1024차원 임베딩을 14개 블록에 해당하는 구간으로 분할 후,
각 구간에서 로지스틱 회귀(선형 분류기)를 학습하여 정확도 측정.

  핵심 아이디어: 선형 분류기가 잘 분류하면 → 해당 구간에 판별 정보 풍부
  선형 분류기가 못 분류하면 → 해당 구간은 비선형/복잡한 정보 또는 판별력 약함

[ YAMNet 채널 구조 ]
  MobileNet V1 아키텍처의 최종 Global Average Pooling 결과 = 1024차원
  실질적 분석: 1024차원을 14 구간으로 균등 분할 (각 ~73차원)
  파인튜닝한 블록(Block 9-14, ★ 표시)과 동결된 블록(Block 1-8) 비교

[ 분석 결과 활용 ]
  파인튜닝 구간(ch 584-1023): Siren Recall 0.904
  동결 구간(ch 0-583):         Siren Recall 0.771
  → Block 9-14 해제(60% 파인튜닝)가 사이렌 판별에 핵심적으로 기여 확인
"""
from __future__ import annotations
import logging, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# MobileNet V1 블록별 채널 구간 정의
# 1024차원을 14개 구간으로 균등 분할 (각 ~73차원)
# ★ 표시: Phase 4에서 파인튜닝한 블록 (Block 9-14)
BLOCK_CHANNELS = {
    "Block 1 (32ch)":    (0,   73),   # 초기 특징: 엣지, 단순 패턴
    "Block 2 (64ch)":    (73,  146),  # 기초 텍스처
    "Block 3 (128ch)":   (146, 219),  # 저수준 오디오 특징
    "Block 4 (128ch)":   (219, 292),  # 주파수 패턴
    "Block 5 (256ch)":   (292, 365),  # 중간 수준 표현
    "Block 6 (256ch)":   (365, 438),  # 주파수 조합
    "Block 7 (512ch)":   (438, 511),  # 고수준 특징 시작
    "Block 8 (512ch)":   (511, 584),  # 동결 구간 마지막 ← 여기까지 frozen
    "Block 9 (512ch)★":  (584, 657),  # ★ 파인튜닝 시작 — ADAS 도메인 적응
    "Block10 (512ch)★":  (657, 730),  # ★
    "Block11 (512ch)★":  (730, 803),  # ★
    "Block12 (512ch)★":  (803, 876),  # ★
    "Block13 (1024ch)★": (876, 950),  # ★
    "Block14 (1024ch)★": (950, 1024), # ★ 파인튜닝 끝, 최종 1024차원 임베딩
}


def remap(y):
    """원본 class_id를 dense label로 변환합니다.
    car_horn(1) → 0, siren(8) → 1, background(99) → 2
    """
    out = np.zeros_like(y)
    out[y==1]=0; out[y==8]=1; out[y==99]=2
    return out


def probe_accuracy(X_tr, y_tr, X_te, y_te):
    """
    로지스틱 회귀 선형 프로브로 임베딩 구간의 판별력을 측정합니다.

    [ 선형 프로브란? ]
    임베딩의 특정 구간에서 단순한 선형 분류기(로지스틱 회귀)를 학습하여
    그 구간이 얼마나 클래스 정보를 포함하는지 측정하는 기법.

    선형 분류기가 잘 분류한다 = 그 구간의 임베딩에 판별 정보가 선형적으로 분리됨
    → YAMNet이 그 블록에서 의미 있는 특징을 추출했다는 증거

    [ 정규화의 필요성 ]
    StandardScaler로 정규화: 로지스틱 회귀는 특징의 스케일에 민감
    각 구간(~73차원)별로 독립적으로 fit/transform하여 공정한 비교

    Args:
        X_tr: 학습 임베딩 구간 (N_train, dim)
        y_tr: 학습 레이블
        X_te: 테스트 임베딩 구간 (N_test, dim)
        y_te: 테스트 레이블

    Returns:
        (전체 정확도, Siren Recall, Horn Recall) 튜플
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import recall_score

    # 정규화: train 기준으로 fit, test에는 transform만
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)

    # 로지스틱 회귀: solver='lbfgs'는 다중 클래스에 적합, max_iter 충분히 크게
    clf = LogisticRegression(max_iter=500, C=1.0, random_state=42, solver="lbfgs")
    clf.fit(Xtr, y_tr)
    preds = clf.predict(Xte)

    acc = (preds == y_te).mean()
    # 사이렌(dense label=1)과 경적(dense label=0)의 개별 recall 계산
    siren_r = recall_score(y_te, preds, labels=[1], average=None, zero_division=0)[0]
    horn_r  = recall_score(y_te, preds, labels=[0], average=None, zero_division=0)[0]
    return acc, siren_r, horn_r


def main(config_path="config.yaml"):
    """
    레이어 프로빙 분석 전체를 실행합니다.

    [ 출력 섹션 ]
    [1] 블록별 선형 프로브: 14개 구간 각각의 판별력
    [2] 누적 채널 분석: Block 1~N까지 누적 사용 시 성능
    [3] 동결 vs 파인튜닝 구간 직접 비교
    [4] 최고 판별력 블록 권고
    """
    from src.utils.config import load_config
    cfg = load_config(config_path)

    # ── 임베딩 로드 ──
    emb_path = PROJECT / cfg["yamnet"]["cache_dir"] / "embeddings.npz"
    data = np.load(str(emb_path), allow_pickle=True)
    X, y_raw, paths = data["X"], data["y"], data["paths"]
    y = remap(y_raw)   # dense label로 변환

    # 경로 기반 폴드/SNR 매핑
    manifest = pd.read_csv(str(PROJECT / cfg["dataset"]["processed_dir"] / "manifest.csv"))
    p2fold = {row["path"].replace("\\","/"): row["fold"]
              for _, row in manifest.iterrows()}
    p2snr  = {row["path"].replace("\\","/"): str(row["snr_db"])
              for _, row in manifest.iterrows()}

    val_folds  = cfg["dataset"]["val_folds"]    # [9]
    test_folds = cfg["dataset"]["test_folds"]   # [10]
    train_folds = [f for f in range(1,11) if f not in val_folds+test_folds]  # 1~8

    def mk_mask(folds, clean=False):
        """폴드 번호와 SNR 조건으로 마스크 배열 반환."""
        m = np.zeros(len(paths), dtype=bool)
        for i, p in enumerate(paths):
            k = str(p).replace("\\","/")
            if p2fold.get(k,-1) in folds:
                if not clean or p2snr.get(k,"") == "clean":
                    m[i] = True
        return m

    # train: fold 1~8 (모든 SNR), test: fold 10 (clean만)
    m_tr = mk_mask(train_folds)
    m_te = mk_mask(test_folds, clean=True)
    X_tr, y_tr = X[m_tr], y[m_tr]
    X_te, y_te = X[m_te], y[m_te]
    log.info("Train %d / Test %d", len(X_tr), len(X_te))

    # ── [1] 블록별 선형 프로브 ─────────────────────────────────
    # 14개 구간 각각에서 로지스틱 회귀 학습 → 판별력 측정
    print("\n" + "="*90)
    print("  LAYER PROBE 분석 — 각 YAMNet 블록 채널 구간의 판별력")
    print("="*90)
    print(f"  {'블록':25s}  {'채널 구간':12s}  {'전체 Acc':>10}  {'Siren Recall':>13}  {'Horn Recall':>11}  {'파인튜닝':>8}")
    print("  "+"-"*88)

    probe_results = []
    best_siren_delta = 0
    best_block = ""
    prev_siren_r = None

    for block_name, (s, e) in BLOCK_CHANNELS.items():
        # 현재 블록 채널 구간만 추출
        Xtr_b = X_tr[:, s:e]
        Xte_b = X_te[:, s:e]
        acc, siren_r, horn_r = probe_accuracy(Xtr_b, y_tr, Xte_b, y_te)
        is_ft = "★ 파인튜닝" if "★" in block_name else "  frozen"

        # 이전 블록 대비 Siren Recall 변화 계산
        delta_str = ""
        if prev_siren_r is not None:
            delta = siren_r - prev_siren_r
            if abs(delta) > 0.02:  # 2% 이상 변화시만 표시
                delta_str = f" ({delta:+.3f})"
                if delta > best_siren_delta:
                    best_siren_delta = delta
                    best_block = block_name   # 최대 향상 블록 기록

        # ASCII 막대 그래프 (Windows 콘솔 호환: ██ 대신 ## 사용)
        bar_s = "#"*int(siren_r*20) + "."*(20-int(siren_r*20))
        print(f"  {block_name:25s}  [{s:4d}:{e:4d}]  {acc:>10.3f}  "
              f"{siren_r:>7.3f}|{bar_s}|{delta_str:>12}  {is_ft}")

        probe_results.append({
            "block": block_name, "ch_start": s, "ch_end": e,
            "acc": acc, "siren_recall": siren_r, "horn_recall": horn_r,
            "finetuned": "★" in block_name,
        })
        prev_siren_r = siren_r

    print(f"\n  ★ Siren Recall 최대 향상 블록: {best_block} (+{best_siren_delta:.3f})")

    # ── [2] 누적 채널 분석 ────────────────────────────────────────
    # Block 1~1, Block 1~2, ... Block 1~14까지 누적 사용 시 성능 변화
    # → 임베딩의 앞부분이 중요한지 뒷부분이 중요한지 파악
    print("\n" + "-"*90)
    print("  누적 채널 분석 — Block 1~N까지 사용 시 판별력 변화")
    print("-"*90)
    print(f"  {'범위':35s}  {'전체 Acc':>10}  {'Siren Recall':>13}  {'Horn Recall':>11}")
    print("  "+"-"*72)

    blocks_list = list(BLOCK_CHANNELS.items())
    cumul_results = []
    for n in range(1, 15):
        s_all = blocks_list[0][1][0]     # 항상 0에서 시작
        e_all = blocks_list[n-1][1][1]   # n번째 블록 끝
        Xtr_c = X_tr[:, s_all:e_all]
        Xte_c = X_te[:, s_all:e_all]
        acc, siren_r, horn_r = probe_accuracy(Xtr_c, y_tr, Xte_c, y_te)
        # Block 9 시작 시 파인튜닝 구간 시작 표시
        ft_mark = "← 파인튜닝 구간 시작" if n == 9 else ""
        print(f"  Block 1~{n:2d}  (ch 0~{e_all:4d}, dim={e_all:4d})  "
              f"{acc:>10.3f}  {siren_r:>13.3f}  {horn_r:>11.3f}  {ft_mark}")
        cumul_results.append({"n_blocks": n, "acc": acc,
                               "siren_recall": siren_r, "horn_recall": horn_r})

    # ── [3] 동결 구간 vs 파인튜닝 구간 직접 비교 ────────────────
    # Block 9(ch 584)를 경계로 두 구간을 독립적으로 비교
    ft_start_ch = BLOCK_CHANNELS["Block 9 (512ch)★"][0]   # 584
    frozen_X_tr = X_tr[:, :ft_start_ch]   # ch 0~583: 동결 구간
    frozen_X_te = X_te[:, :ft_start_ch]
    ft_X_tr     = X_tr[:, ft_start_ch:]   # ch 584~1023: 파인튜닝 구간
    ft_X_te     = X_te[:, ft_start_ch:]

    acc_f, sir_f, horn_f = probe_accuracy(frozen_X_tr, y_tr, frozen_X_te, y_te)
    acc_t, sir_t, horn_t = probe_accuracy(ft_X_tr,     y_tr, ft_X_te,     y_te)
    acc_a, sir_a, horn_a = probe_accuracy(X_tr,        y_tr, X_te,        y_te)  # 전체

    print("\n" + "-"*90)
    print("  Frozen 구간 vs Fine-tuned 구간 직접 비교")
    print("-"*90)
    print(f"  {'구간':35s}  {'차원수':>6}  {'전체 Acc':>10}  {'Siren Recall':>13}  {'Horn Recall':>11}")
    print("  "+"-"*78)
    print(f"  {'Frozen (Block 1-8, ch 0-583)':35s}  {ft_start_ch:>6}  "
          f"{acc_f:>10.3f}  {sir_f:>13.3f}  {horn_f:>11.3f}")
    print(f"  {'Fine-tuned (Block 9-14, ch 584-1023)':35s}  {1024-ft_start_ch:>6}  "
          f"{acc_t:>10.3f}  {sir_t:>13.3f}  {horn_t:>11.3f}")
    print(f"  {'전체 1024차원':35s}  {'1024':>6}  "
          f"{acc_a:>10.3f}  {sir_a:>13.3f}  {horn_a:>11.3f}")

    siren_gain = sir_t - sir_f   # 파인튜닝 구간이 동결 구간보다 얼마나 좋은가
    print(f"\n  ★ 파인튜닝 구간이 frozen 구간 대비 Siren Recall {siren_gain:+.3f}")
    if siren_gain > 0:
        print(f"    → Layer 9-14 파인튜닝이 siren 판별에 핵심적으로 기여!")
        print(f"    → 60% 파인튜닝 전략(Block 9-14 해제)이 올바른 선택임을 확인")
    else:
        print(f"    → 초기 레이어가 오히려 더 판별력 높음 (더 낮은 레이어 해제 검토 필요)")

    # ── [4] 권고 사항 ────────────────────────────────────────────
    df_probe = pd.DataFrame(probe_results)
    best_row = df_probe.loc[df_probe["siren_recall"].idxmax()]
    print(f"\n  ★ 단독 구간 기준 Siren Recall 최고 블록: {best_row['block']}"
          f" ({best_row['siren_recall']:.3f})")
    print(f"    해당 채널 구간: {best_row['ch_start']}~{best_row['ch_end']}")

    return df_probe, pd.DataFrame(cumul_results), {
        "frozen": (acc_f, sir_f), "finetuned": (acc_t, sir_t), "full": (acc_a, sir_a)
    }


if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser(description="YAMNet 레이어별 판별력 분석")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    os.chdir(PROJECT)
    main(args.config)
