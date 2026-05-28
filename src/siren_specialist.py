"""
Siren Specialist — 사이렌 전용 이진 분류기
(v5 모델 — 3-class DNN의 Softmax 경쟁 구조 한계를 극복하기 위해 설계)

[ 탄생 배경 ]
3-class DNN (classifier.py, finetune.py)의 Softmax 출력에서
사이렌(siren) 클래스의 Recall이 68.7%에 머물렀습니다.
원인 분석: Softmax 경쟁 구조 + 클래스 불균형(배경 65%)
  → 사이렌이 배경으로 "밀려나는" 현상

해결책: 사이렌만 따로 이진 분류하는 전문가 모델 훈련
  → Softmax 경쟁 없음 + 사이렌 recall 극대화 손실함수 사용

YAMNet 임베딩(이미 추출됨)을 재사용하여
siren vs non-siren 이진 분류기를 별도 학습합니다.
→ YAMNet 재실행 없이 빠른 학습 가능

사용 기법:
  1. Asymmetric Focal Loss (γ=2) — FN(놓침) 패널티를 FP(오경보)보다 5배 크게
  2. MixUp Augmentation — siren+background 임베딩 보간으로 경계 샘플 생성
  3. StandardScaler — 임베딩 정규화 (z=(x-μ)/σ)
  4. Val 기준 최적 threshold 자동 탐색 (Recall≥90% 조건)

출력:
  models/siren_specialist.h5        — 이진 분류기 가중치
  models/siren_threshold.txt        — 최적 threshold 값
  models/siren_specialist_scaler.pkl — StandardScaler (추론 시 동일 정규화)
"""
from __future__ import annotations
import logging, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 손실 함수: Asymmetric Focal Loss (비대칭 초점 손실)
# ─────────────────────────────────────────────────────────────
def asymmetric_focal_loss(gamma_pos: float = 0.0,
                           gamma_neg: float = 2.0,
                           fn_weight: float = 5.0):
    """
    Asymmetric Focal Loss — 사이렌 탐지에 최적화된 손실 함수

    [ 설계 철학 ]
    ADAS 시스템에서 사이렌을 놓치는 것(FN: False Negative)은
    오경보(FP: False Positive)보다 훨씬 더 위험합니다.
    → FN 패널티를 FP의 5배로 설정하여 재현율(Recall) 극대화

    [ Focal Loss의 역할 ]
    표준 Binary Cross-Entropy를 사용하면 쉬운 샘플(명백한 배경)이
    손실을 지배하여 경계 샘플(애매한 사이렌)을 제대로 학습하지 못합니다.
    gamma_neg=2.0: (1-p_neg)^2 항이 쉬운 negative 샘플의 loss를 크게 줄임
    → 모델이 어려운 경계 샘플에 집중하도록 유도

    [ 수식 ]
    Positive (siren=1):  loss = -fn_weight × y × log(p)
    Negative (non-siren=0): loss = -(1-y) × (1-p_neg)^γ × log(p_neg)

    Args:
        gamma_pos: positive 샘플의 focal 파라미터 (보통 0.0, 즉 focal 없음)
        gamma_neg: negative 샘플의 focal 파라미터 (2.0 = 표준 focal)
        fn_weight: FN(사이렌 놓침) 패널티 배수 (기본 5.0)

    Returns:
        Keras loss_fn(y_true, y_pred) 함수
    """
    import tensorflow as tf

    def loss_fn(y_true, y_pred):
        # 수치 안정성: 예측값을 [1e-7, 1-1e-7] 범위로 클리핑
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # ── Positive (siren=1) 항 ──
        # fn_weight 배로 패널티 → 놓치면 안 되는 사이렌에 집중
        p_t_pos = y_pred
        loss_pos = -fn_weight * y_true * (
            (1 - p_t_pos) ** gamma_pos * tf.math.log(p_t_pos)
        )

        # ── Negative (non-siren=0) 항 ──
        # gamma_neg: 쉬운 negative(확실한 배경)의 loss를 크게 줄임
        p_t_neg = 1.0 - y_pred
        loss_neg = -(1.0 - y_true) * (
            (1 - p_t_neg) ** gamma_neg * tf.math.log(p_t_neg)
        )

        return tf.reduce_mean(loss_pos + loss_neg)

    return loss_fn


# ─────────────────────────────────────────────────────────────
# 데이터 증강: MixUp (임베딩 레벨)
# ─────────────────────────────────────────────────────────────
def mixup_embeddings(X: np.ndarray, y: np.ndarray,
                     alpha: float = 0.4,
                     n_extra: int | None = None,
                     rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Siren + Non-siren 임베딩 보간으로 경계(hard boundary) 샘플 생성.

    [ MixUp이란? ]
    두 샘플을 λ:(1-λ) 비율로 선형 보간하여 새로운 샘플을 만드는 기법.
    원본 데이터 분포의 경계 근방 샘플을 인위적으로 생성하여
    모델이 결정 경계를 더 정확하게 학습하도록 합니다.

    [ 왜 임베딩 레벨에서 하는가? ]
    파형 수준의 MixUp은 실제로 들리지 않는 조합이 생길 수 있습니다.
    YAMNet이 이미 의미 있는 표현(1024차원 임베딩)을 추출했으므로
    그 공간에서 보간하는 것이 더 의미 있는 중간 샘플을 만듭니다.

    [ λ 클리핑 [0.3, 0.7] ]
    너무 한쪽으로 치우친 샘플(λ→0 또는 λ→1)은 거의 원본과 같아서
    경계 샘플로서의 가치가 없습니다.
    0.3~0.7 범위에서만 보간하여 siren과 non-siren이 충분히 혼합된 샘플 생성.

    Args:
        X: 전체 임베딩 배열 (N, 1024)
        y: 이진 레이블 배열 (N,) — siren=1, non-siren=0
        alpha: Beta 분포 파라미터 (alpha=alpha → 대칭 분포)
        n_extra: 생성할 추가 샘플 수 (None이면 siren 수 × 2)
        rng: numpy 난수 생성기 (None이면 seed=42로 초기화)

    Returns:
        (mixed_X, mixed_y) — 보간된 임베딩과 soft label (0.3~0.7 사이 실수)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    siren_idx = np.where(y == 1)[0]    # 사이렌 샘플 인덱스
    non_idx   = np.where(y == 0)[0]    # 비사이렌 샘플 인덱스
    if n_extra is None:
        n_extra = len(siren_idx) * 2   # siren 수의 2배 생성

    mixed_X, mixed_y = [], []
    for _ in range(n_extra):
        # λ ~ Beta(alpha, alpha): 대칭 분포, alpha=0.4이면 양 끝에 집중
        lam = rng.beta(alpha, alpha)
        # [0.3, 0.7]로 클리핑: 경계에 가까운 샘플만 생성
        lam = np.clip(lam, 0.3, 0.7)

        i = rng.choice(siren_idx)   # 사이렌 샘플 무작위 선택
        j = rng.choice(non_idx)     # 비사이렌 샘플 무작위 선택

        # 선형 보간: λ×siren + (1-λ)×non_siren
        mixed_X.append(lam * X[i] + (1 - lam) * X[j])
        mixed_y.append(float(lam))  # soft label: 사이렌 비율이 레이블

    return np.array(mixed_X, dtype=np.float32), np.array(mixed_y, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# 모델 아키텍처
# ─────────────────────────────────────────────────────────────
def build_specialist(input_dim: int = 1024,
                     hidden: list[int] = [512, 128, 32],
                     dropout: float = 0.35) -> "tf.keras.Model":
    """
    사이렌 전용 이진 분류기 DNN을 구성합니다.

    [ 아키텍처 선택 이유 ]
    Input(1024) → Dense(512) → Dense(128) → Dense(32) → Dense(1, sigmoid)

    - 3개 은닉층: 임베딩 공간의 비선형 경계를 충분히 표현
    - use_bias=False + BatchNormalization: 편향 항을 BN이 흡수
      (BN이 자체적으로 평균 이동을 학습하므로 별도 bias 불필요)
    - Dropout(0.35): 과적합 방지, 앙상블 효과
    - 출력: sigmoid → 0~1 사이 사이렌 확률값
      (softmax 대신 sigmoid → 배경과의 직접 경쟁 없음)

    Args:
        input_dim: 입력 임베딩 차원 (YAMNet = 1024)
        hidden: 각 은닉층 유닛 수 리스트
        dropout: Dropout 비율

    Returns:
        컴파일 전 Keras 모델
    """
    import tensorflow as tf
    inp = tf.keras.Input(shape=(input_dim,), name="yamnet_embedding")
    x = inp
    for units in hidden:
        # Dense → BatchNorm → ReLU → Dropout 순서
        x = tf.keras.layers.Dense(units, use_bias=False)(x)    # bias=False (BN이 대신)
        x = tf.keras.layers.BatchNormalization()(x)             # 학습 안정화
        x = tf.keras.layers.Activation("relu")(x)               # 비선형성
        x = tf.keras.layers.Dropout(dropout)(x)                 # 과적합 방지
    # 출력층: sigmoid → 사이렌일 확률 (0~1)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="siren_prob")(x)
    return tf.keras.Model(inp, out, name="siren_specialist")


# ─────────────────────────────────────────────────────────────
# 최적 Threshold 탐색
# ─────────────────────────────────────────────────────────────
def find_best_threshold(y_true: np.ndarray,
                         y_prob: np.ndarray,
                         min_recall: float = 0.90) -> tuple[float, float, float]:
    """
    Recall >= min_recall 조건에서 Precision을 최대화하는 threshold를 찾습니다.

    [ Threshold란? ]
    sigmoid 출력값이 threshold 이상이면 "사이렌"으로 판단.
    - 낮은 threshold: 더 많이 탐지 → Recall↑ 하지만 FP(오경보)↑
    - 높은 threshold: 더 확실할 때만 탐지 → Precision↑ 하지만 FN(놓침)↑

    [ 탐색 전략 ]
    0.05~0.94 사이를 0.01 간격으로 전수 탐색.
    Recall >= min_recall(0.90) 조건을 만족하는 threshold 중
    Precision이 가장 높은 값을 선택 (최소 오경보로 최대 탐지).

    [ Validation 세트 사용 이유 ]
    Test 세트는 최종 평가까지 절대 사용하지 않음.
    Val 세트로 threshold를 결정한 후, Test 세트에서 최종 성능 측정.

    Args:
        y_true: 실제 이진 레이블 (siren=1, non-siren=0)
        y_prob: 모델이 예측한 사이렌 확률 (0~1)
        min_recall: 최소 재현율 제약 조건 (기본 0.90 = 90% 이상)

    Returns:
        (threshold, recall, precision) 튜플
    """
    best_thresh, best_prec = 0.5, 0.0
    best_rec = 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (y_prob >= t).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        rec  = tp / (tp + fn + 1e-9)   # 재현율: 실제 사이렌 중 탐지한 비율
        prec = tp / (tp + fp + 1e-9)   # 정밀도: 탐지한 것 중 실제 사이렌 비율
        if rec >= min_recall and prec > best_prec:
            best_prec = prec
            best_thresh = t
            best_rec = rec

    # Recall 조건 만족 못하면 최대 Recall threshold 반환 (차선책)
    if best_prec == 0.0:
        idx = np.argmax([
            ((y_prob >= t).astype(int) * (y_true == 1)).sum() / (y_true.sum() + 1e-9)
            for t in np.arange(0.05, 0.95, 0.01)
        ])
        best_thresh = 0.05 + idx * 0.01
        preds = (y_prob >= best_thresh).astype(int)
        tp = ((preds==1)&(y_true==1)).sum()
        fn = ((preds==0)&(y_true==1)).sum()
        fp = ((preds==1)&(y_true==0)).sum()
        best_rec  = tp/(tp+fn+1e-9)
        best_prec = tp/(tp+fp+1e-9)
    return float(best_thresh), float(best_rec), float(best_prec)


# ─────────────────────────────────────────────────────────────
# 메인 학습 함수
# ─────────────────────────────────────────────────────────────
def train_specialist(config_path: str = "config.yaml") -> dict:
    """
    사이렌 전용 이진 분류기 학습 전체 파이프라인을 실행합니다.

    [ 전체 흐름 ]
    1. embeddings.npz에서 YAMNet 임베딩 로드
    2. 이진 레이블 생성 (siren=1, others=0)
    3. MixUp 증강으로 경계 샘플 추가
    4. StandardScaler로 정규화
    5. 모델 구성 + Asymmetric Focal Loss 컴파일
    6. ModelCheckpoint + EarlyStopping(val_recall) + ReduceLROnPlateau 콜백
    7. 학습 (최대 200 에폭)
    8. val 세트에서 최적 threshold 탐색
    9. test 세트 최종 평가
    10. 모델/threshold/scaler 저장

    Returns:
        {model_path, threshold, scaler_path} 딕셔너리
    """
    import tensorflow as tf
    from src.utils.config import load_config

    cfg = load_config(config_path)
    val_folds  = cfg["dataset"]["val_folds"]    # [9]
    test_folds = cfg["dataset"]["test_folds"]   # [10]
    # train: fold 1~8 (val, test 제외)
    train_folds = [f for f in range(1, 11) if f not in val_folds + test_folds]

    # ── 임베딩 로드 ──────────────────────────────────────────────
    # embedding.py가 미리 추출해 놓은 1024차원 YAMNet 임베딩 사용
    emb_path = PROJECT / cfg["yamnet"]["cache_dir"] / "embeddings.npz"
    data = np.load(str(emb_path), allow_pickle=True)
    X, y_raw, paths = data["X"], data["y"], data["paths"]

    # manifest에서 경로→폴드, 경로→SNR 매핑 구성
    manifest = pd.read_csv(str(PROJECT / cfg["dataset"]["processed_dir"] / "manifest.csv"))
    p2fold = {r["path"].replace("\\","/"): r["fold"] for _, r in manifest.iterrows()}
    p2snr  = {r["path"].replace("\\","/"): str(r["snr_db"]) for _, r in manifest.iterrows()}

    def mk_mask(folds, clean=False):
        """특정 폴드 + 선택적으로 clean 조건의 인덱스 마스크 반환."""
        m = np.zeros(len(paths), dtype=bool)
        for i, p in enumerate(paths):
            k = str(p).replace("\\","/")
            if p2fold.get(k, -1) in folds:
                if not clean or p2snr.get(k,"") == "clean":
                    m[i] = True
        return m

    # 이진 레이블: siren(class_id=8)=1, 나머지(horn, background)=0
    # Softmax 3-class 모델과 달리 "사이렌이냐 아니냐"만 판단
    y_bin = (y_raw == 8).astype(np.float32)

    # 폴드 마스크 생성
    m_tr = mk_mask(train_folds)              # 학습: fold 1~8, 모든 SNR
    m_va = mk_mask(val_folds)               # 검증: fold 9, 모든 SNR
    m_te = mk_mask(test_folds, clean=True)  # 테스트: fold 10, clean만

    X_tr, y_tr = X[m_tr], y_bin[m_tr]
    X_va, y_va = X[m_va], y_bin[m_va]
    X_te, y_te = X[m_te], y_bin[m_te]

    log.info("Train: %d (siren=%d, non=%d)", len(y_tr), y_tr.sum().astype(int),
             (y_tr==0).sum().astype(int))
    log.info("Val:   %d (siren=%d, non=%d)", len(y_va), y_va.sum().astype(int),
             (y_va==0).sum().astype(int))
    log.info("Test:  %d (siren=%d, non=%d)", len(y_te), y_te.sum().astype(int),
             (y_te==0).sum().astype(int))

    # ── MixUp 증강 ───────────────────────────────────────────────
    # siren 수의 3배만큼 보간 샘플 생성 → 경계 학습 강화
    rng = np.random.default_rng(42)
    X_mix, y_mix = mixup_embeddings(X_tr, y_tr, alpha=0.4,
                                      n_extra=int(y_tr.sum()) * 3,
                                      rng=rng)
    # 원본 + 보간 샘플 결합
    X_tr_aug = np.concatenate([X_tr, X_mix], axis=0)
    y_tr_aug = np.concatenate([y_tr, y_mix], axis=0)
    # 순서 섞기 (원본↔보간 패턴 방지)
    idx = rng.permutation(len(X_tr_aug))
    X_tr_aug, y_tr_aug = X_tr_aug[idx], y_tr_aug[idx]
    log.info("After MixUp — Train: %d (original %d + mixed %d)",
             len(X_tr_aug), len(X_tr), len(X_mix))

    # ── 정규화 (StandardScaler) ───────────────────────────────────
    # z = (x - μ) / σ
    # fit은 train에서만! val/test는 transform만 적용 (데이터 누수 방지)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr_aug = scaler.fit_transform(X_tr_aug)   # 학습 + 변환
    X_va_sc  = scaler.transform(X_va)           # 변환만 (학습 데이터의 μ, σ 사용)
    X_te_sc  = scaler.transform(X_te)           # 변환만

    # 추론 시 동일한 정규화를 위해 scaler 저장
    import pickle
    scaler_path = PROJECT / "models" / "siren_specialist_scaler.pkl"
    with open(str(scaler_path), "wb") as f:
        pickle.dump(scaler, f)

    # ── 모델 구성 ────────────────────────────────────────────────
    model = build_specialist(input_dim=1024, hidden=[512, 128, 32], dropout=0.35)
    model.summary(print_fn=log.info)

    # Asymmetric Focal Loss: FN 패널티 5배, 쉬운 negative focal γ=2
    loss_fn = asymmetric_focal_loss(gamma_pos=0.0, gamma_neg=2.0, fn_weight=5.0)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.Recall(name="recall"),        # 주요 모니터링 지표
            tf.keras.metrics.Precision(name="precision"),
        ],
    )

    # ── 콜백 설정 ─────────────────────────────────────────────────
    model_path = PROJECT / "models" / "siren_specialist.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor="val_recall",           # val_recall 최대 시 저장
            mode="max",                     # 최대화 목표
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_recall", mode="max",
            patience=20,                    # 20 에폭 동안 개선 없으면 종료
            restore_best_weights=True,      # 최고 val_recall 가중치 복원
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_recall", mode="max",
            factor=0.5,                     # 학습률 절반으로 감소
            patience=8,                     # 8 에폭 정체 후 감소
            min_lr=1e-7,                    # 최소 학습률
            verbose=1,
        ),
    ]

    # ── 학습 ─────────────────────────────────────────────────────
    # val_recall 최대화가 목표 (사이렌 놓치면 안 됨!)
    log.info("="*60)
    log.info("Siren Specialist 학습 시작")
    log.info("  Loss  : Asymmetric Focal (FN×5, γ=2)")
    log.info("  MixUp : siren×%d mixed samples", len(X_mix))
    log.info("  Monitor: val_recall (max)")
    log.info("="*60)

    history = model.fit(
        X_tr_aug, y_tr_aug,
        validation_data=(X_va_sc, y_va),
        epochs=200,          # EarlyStopping이 있으므로 충분히 큰 수
        batch_size=256,      # 큰 배치 → 안정적인 기울기 추정
        callbacks=callbacks,
        verbose=1,
    )

    # ── 최적 Threshold 탐색 ──────────────────────────────────────
    # val 세트 예측 확률로 threshold 결정 (test는 절대 사용 안 함)
    log.info("Val set에서 최적 threshold 탐색 중…")
    val_probs = model.predict(X_va_sc, verbose=0).flatten()
    best_t, best_rec, best_prec = find_best_threshold(y_va, val_probs, min_recall=0.90)
    log.info("  최적 threshold=%.2f  recall=%.3f  precision=%.3f",
             best_t, best_rec, best_prec)

    # threshold 파일 저장 (실시간 추론 시 로드)
    thresh_path = PROJECT / "models" / "siren_threshold.txt"
    thresh_path.write_text(str(best_t))

    # ── 테스트 평가 ───────────────────────────────────────────────
    # 처음으로 test 세트 사용 — 여러 threshold에서 성능 확인
    log.info("="*60)
    log.info("TEST SET 평가 (fold 10, clean)")
    test_probs = model.predict(X_te_sc, verbose=0).flatten()

    for t_label, t_val in [
            ("threshold=0.50 (기본)", 0.50),
            ("threshold=0.30",        0.30),
            (f"threshold={best_t:.2f} (최적)", best_t),
            ("threshold=0.15",        0.15),
            ("threshold=0.10",        0.10),
    ]:
        preds = (test_probs >= t_val).astype(int)
        tp = ((preds==1) & (y_te==1)).sum()
        fp = ((preds==1) & (y_te==0)).sum()
        fn = ((preds==0) & (y_te==1)).sum()
        tn = ((preds==0) & (y_te==0)).sum()
        rec  = tp/(tp+fn+1e-9)
        prec = tp/(tp+fp+1e-9)
        f1   = 2*rec*prec/(rec+prec+1e-9)
        log.info("  %s  →  Recall=%.3f  Prec=%.3f  F1=%.3f  (TP=%d FP=%d FN=%d)",
                 t_label, rec, prec, f1, tp, fp, fn)

    log.info("="*60)
    log.info("모델 저장: %s", model_path)
    log.info("Threshold 저장: %s", thresh_path)

    return {
        "model_path": str(model_path),
        "threshold":  best_t,
        "scaler_path": str(scaler_path),
    }


# ─────────────────────────────────────────────────────────────
# CLI 진입점
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser(description="Siren Specialist 학습")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    os.chdir(PROJECT)
    result = train_specialist(args.config)
    print("\n완료:", result)
