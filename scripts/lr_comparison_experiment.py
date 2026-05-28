# -*- coding: utf-8 -*-
"""
yamnet_lr 후보 비교 실험 스크립트
================================
1e-4 / 1e-5 / 5e-6 / 1e-6 네 가지 학습률로 YAMNet 파인튜닝을 실행하고
각각의 Accuracy, Siren Recall, Horn Recall 수치를 비교합니다.

[ 실험 조건 ]
- 파인튜닝 대상: Block 9~14 (unfreeze_frac=0.60, 33/56 변수)
- clf_lr: 고정 2e-5 (yamnet_lr만 변화)
- max_epochs: 20 (early stopping patience=4)
- 학습 데이터: fold 1~8, clean only
- 테스트 데이터: fold 10, clean only
- 시작 가중치: models/custom_classifier.h5 (동일 기준선)

실행: python -X utf8 scripts/lr_comparison_experiment.py
결과: results/lr_comparison_results.csv
"""
import os, sys, time, logging, ctypes
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── WSL2/Linux GPU 라이브러리 선 로드 (TF보다 먼저 해야 인식됨) ──
_NV = "/home/jeongwoo/adas-env/lib/python3.12/site-packages/nvidia"
for _lib in [
    f"{_NV}/cuda_runtime/lib/libcudart.so.12",
    f"{_NV}/cublas/lib/libcublas.so.12",
    f"{_NV}/cublas/lib/libcublasLt.so.12",
    f"{_NV}/cufft/lib/libcufft.so.11",
    f"{_NV}/curand/lib/libcurand.so.10",
    f"{_NV}/cusolver/lib/libcusolver.so.11",
    f"{_NV}/cusparse/lib/libcusparse.so.12",
    f"{_NV}/cudnn/lib/libcudnn.so.9",
    f"{_NV}/nvjitlink/lib/libnvJitLink.so.12",
    f"{_NV}/cuda_nvrtc/lib/libnvrtc.so.12",
]:
    try:
        ctypes.CDLL(_lib)
    except OSError:
        pass   # Windows 환경이면 스킵

from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 설정 ──────────────────────────────────────────────────────────
YAMNET_URL    = "https://tfhub.dev/google/yamnet/1"
CLF_H5        = PROJECT / "models/custom_classifier.h5"
MANIFEST      = PROJECT / "data/processed/manifest.csv"
SR            = 16_000
MAX_DUR_SEC   = 4.0
MAX_SAMPLES   = int(SR * MAX_DUR_SEC)
BATCH_SIZE    = 16
MAX_EPOCHS    = 20
PATIENCE      = 4
UNFREEZE_FRAC = 0.60   # Block 9~14 (60%)
CLF_LR        = 2e-5   # 분류기 헤드 고정

LR_CANDIDATES = [1e-4, 1e-5, 5e-6, 1e-6]   # 비교할 yamnet_lr 후보

# ── 레이블 매핑 ────────────────────────────────────────────────────
# class_id: 1=car_horn, 8=siren, 99=background → dense 0/1/2
def remap(cid):
    if cid == 1:  return 0   # car_horn
    if cid == 8:  return 1   # siren
    return 2                  # background

# ── 데이터 로드 ────────────────────────────────────────────────────
log.info("매니페스트 로드 중...")
mf = pd.read_csv(str(MANIFEST))
mf["label"] = mf["class_id"].apply(remap)

train_df = mf[(mf["fold"].isin(range(1, 9))) & (mf["snr_db"] == "clean")]
test_df  = mf[(mf["fold"] == 10)             & (mf["snr_db"] == "clean")]
log.info("Train: %d  Test: %d", len(train_df), len(test_df))

def load_wav(path):
    """파형을 16kHz 모노로 로드하고 MAX_SAMPLES 길이로 맞춤"""
    path = path.replace("\\", "/")   # WSL2: Windows 백슬래시 → 슬래시 변환
    raw = tf.io.read_file(path)
    wav, sr = tf.audio.decode_wav(raw, desired_channels=1)
    wav = tf.squeeze(wav, -1)
    # 길이 맞추기
    wav = wav[:MAX_SAMPLES]
    pad = MAX_SAMPLES - tf.shape(wav)[0]
    wav = tf.pad(wav, [[0, pad]])
    return wav  # (MAX_SAMPLES,)

log.info("파형 로드 중 (train %d개)...", len(train_df))
t0 = time.time()
X_tr_list, y_tr_list = [], []
for _, row in train_df.iterrows():
    try:
        wav = load_wav(str(PROJECT / row["path"]).replace("\\", "/"))
        X_tr_list.append(wav.numpy())
        y_tr_list.append(row["label"])
    except Exception as e:
        log.warning("스킵: %s (%s)", row["path"], e)

X_tr = np.stack(X_tr_list).astype(np.float32)  # (N, 64000)
y_tr = np.array(y_tr_list, dtype=np.int32)
log.info("Train 파형 로드 완료: %.1f초", time.time() - t0)

log.info("파형 로드 중 (test %d개)...", len(test_df))
X_te_list, y_te_list = [], []
for _, row in test_df.iterrows():
    try:
        wav = load_wav(str(PROJECT / row["path"]).replace("\\", "/"))
        X_te_list.append(wav.numpy())
        y_te_list.append(row["label"])
    except Exception as e:
        log.warning("스킵: %s (%s)", row["path"], e)

X_te = np.stack(X_te_list).astype(np.float32)
y_te = np.array(y_te_list, dtype=np.int32)
log.info("Test  파형 로드 완료")

# 클래스 가중치 계산
from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight("balanced", classes=np.array([0,1,2]), y=y_tr)
cw_tensor = tf.constant(cw, dtype=tf.float32)

# ── YAMNet 로드 (한 번만) ──────────────────────────────────────────
log.info("YAMNet 로드 중...")
yamnet = hub.load(YAMNET_URL)
log.info("YAMNet 로드 완료")

# 변수 발견 (GradientTape 트릭)
dummy = tf.zeros(MAX_SAMPLES, dtype=tf.float32)
with tf.GradientTape() as _tape:
    _, _emb, _ = yamnet(dummy)
    _ = tf.reduce_sum(_emb)
all_vars = sorted(_tape.watched_variables(), key=lambda v: v.name)
log.info("YAMNet 변수: %d개 발견", len(all_vars))

# 파인튜닝 대상 변수 (마지막 60%)
n_total   = len(all_vars)
n_unfreeze = int(n_total * UNFREEZE_FRAC)
yamnet_train_vars = all_vars[n_total - n_unfreeze:]   # Block 9~14
log.info("파인튜닝 대상: %d / %d 변수 (Block 9~14)", len(yamnet_train_vars), n_total)

# 원본 가중치 저장 (각 실험마다 동일 시작점으로 복원)
original_yamnet_weights = [v.numpy().copy() for v in all_vars]

# ── 임베딩 추출 함수 (배치 단위) ──────────────────────────────────
@tf.function
def get_embedding(wav):
    """단일 파형 → 임베딩 (시간 평균)"""
    _, emb, _ = yamnet(wav)
    return tf.reduce_mean(emb, axis=0)   # (1024,)

def extract_embeddings(X):
    """X: (N, MAX_SAMPLES) → embeddings: (N, 1024)"""
    embs = []
    for i in range(0, len(X), 32):
        batch = X[i:i+32]
        batch_embs = [get_embedding(wav).numpy() for wav in batch]
        embs.extend(batch_embs)
    return np.array(embs, dtype=np.float32)

# ── 학습 스텝 ─────────────────────────────────────────────────────
@tf.function
def train_step(wavs, labels, classifier, yamnet_opt, clf_opt):
    with tf.GradientTape() as tape:
        embeddings = tf.map_fn(
            lambda w: tf.reduce_mean(yamnet(w)[1], axis=0),
            wavs, dtype=tf.float32
        )
        logits = classifier(embeddings, training=True)
        weights = tf.cast(tf.gather(cw_tensor, labels), tf.float32)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels, logits, from_logits=False
            ) * weights
        )
    clf_vars = classifier.trainable_variables
    all_train = yamnet_train_vars + clf_vars
    grads = tape.gradient(loss, all_train)
    n_y = len(yamnet_train_vars)
    yamnet_opt.apply_gradients(zip(grads[:n_y], yamnet_train_vars))
    clf_opt.apply_gradients(zip(grads[n_y:], clf_vars))
    return loss

# ── 평가 함수 ─────────────────────────────────────────────────────
def evaluate(X, y_true, classifier):
    embs = extract_embeddings(X)
    probs = classifier.predict(embs, verbose=0)
    preds = np.argmax(probs, axis=1)
    acc   = (preds == y_true).mean()
    # Siren Recall (class 1)
    siren_mask = (y_true == 1)
    s_rec = (preds[siren_mask] == 1).mean() if siren_mask.sum() > 0 else 0.0
    # Horn Recall (class 0)
    horn_mask = (y_true == 0)
    h_rec = (preds[horn_mask] == 0).mean() if horn_mask.sum() > 0 else 0.0
    # BG Recall (class 2)
    bg_mask = (y_true == 2)
    b_rec = (preds[bg_mask] == 2).mean() if bg_mask.sum() > 0 else 0.0
    return acc, s_rec, h_rec, b_rec

# ── val 데이터 (fold 9, clean) ──────────────────────────────────
val_df = mf[(mf["fold"] == 9) & (mf["snr_db"] == "clean")]
X_val_list, y_val_list = [], []
for _, row in val_df.iterrows():
    try:
        wav = load_wav(str(PROJECT / row["path"]).replace("\\", "/"))
        X_val_list.append(wav.numpy())
        y_val_list.append(row["label"])
    except:
        pass
X_val = np.stack(X_val_list).astype(np.float32)
y_val = np.array(y_val_list, dtype=np.int32)
log.info("Val: %d개", len(X_val))

# ── 셔플 인덱스 ────────────────────────────────────────────────────
rng = np.random.default_rng(42)
idx = rng.permutation(len(X_tr))
X_tr_shuf, y_tr_shuf = X_tr[idx], y_tr[idx]

# ════════════════════════════════════════════════════════════════
# 메인 실험 루프
# ════════════════════════════════════════════════════════════════
results = []

for yamnet_lr in LR_CANDIDATES:
    log.info("=" * 60)
    log.info("▶  yamnet_lr = %.0e  시작", yamnet_lr)
    log.info("=" * 60)

    # YAMNet 가중치 원본으로 복원
    for var, orig in zip(all_vars, original_yamnet_weights):
        var.assign(orig)

    # 분류기 새로 로드 (동일 초기 가중치)
    classifier = tf.keras.models.load_model(str(CLF_H5))

    # 옵티마이저
    yamnet_opt = tf.keras.optimizers.Adam(learning_rate=yamnet_lr)
    clf_opt    = tf.keras.optimizers.Adam(learning_rate=CLF_LR)

    best_val_loss = float("inf")
    patience_cnt  = 0
    best_acc = best_siren = best_horn = best_bg = 0.0
    best_epoch = 0

    t_start = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        # ── 학습 ──
        train_losses = []
        for i in range(0, len(X_tr_shuf), BATCH_SIZE):
            wavs_b   = tf.constant(X_tr_shuf[i:i+BATCH_SIZE])
            labels_b = tf.constant(y_tr_shuf[i:i+BATCH_SIZE])
            loss = train_step(wavs_b, labels_b, classifier, yamnet_opt, clf_opt)
            train_losses.append(float(loss))

        train_loss = np.mean(train_losses)

        # ── val loss 계산 ──
        val_embs = extract_embeddings(X_val)
        val_probs = classifier.predict(val_embs, verbose=0)
        val_loss_val = float(tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                y_val, val_probs
            )
        ))

        # ── val 지표 ──
        val_preds = np.argmax(val_probs, axis=1)
        val_acc   = (val_preds == y_val).mean()
        val_siren = (val_preds[y_val==1] == 1).mean() if (y_val==1).sum() > 0 else 0.0

        log.info(
            "  Epoch %2d/%d | train_loss=%.4f  val_loss=%.4f  "
            "val_acc=%.3f  val_siren_rec=%.3f",
            epoch, MAX_EPOCHS, train_loss, val_loss_val, val_acc, val_siren
        )

        # ── Early Stopping ──
        if val_loss_val < best_val_loss - 1e-4:
            best_val_loss = val_loss_val
            patience_cnt  = 0
            # 이 에폭의 test 지표 저장
            ta, ts, th, tb = evaluate(X_te, y_te, classifier)
            best_acc, best_siren, best_horn, best_bg = ta, ts, th, tb
            best_epoch = epoch
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                log.info("  Early stopping at epoch %d", epoch)
                break

    elapsed = time.time() - t_start
    log.info(
        "  ✓ yamnet_lr=%.0e 완료 | best_epoch=%d | "
        "test_acc=%.3f  siren_rec=%.3f  horn_rec=%.3f  bg_rec=%.3f  (%.0fs)",
        yamnet_lr, best_epoch, best_acc, best_siren, best_horn, best_bg, elapsed
    )

    results.append({
        "yamnet_lr":   f"{yamnet_lr:.0e}",
        "best_epoch":  best_epoch,
        "test_acc":    round(best_acc,   4),
        "siren_recall":round(best_siren, 4),
        "horn_recall": round(best_horn,  4),
        "bg_recall":   round(best_bg,    4),
        "elapsed_sec": int(elapsed),
    })

# ── 결과 저장 ──────────────────────────────────────────────────────
out_path = PROJECT / "results/lr_comparison_results.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(results)
df.to_csv(str(out_path), index=False, encoding="utf-8-sig")

print("\n" + "=" * 70)
print("  yamnet_lr 비교 실험 결과")
print("=" * 70)
print(df.to_string(index=False))
print(f"\n결과 저장: {out_path}")
