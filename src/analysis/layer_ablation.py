"""
Layer Ablation Study — 레이어별 파인튜닝 기여도 분석

YAMNet의 각 레이어 그룹이 siren/car_horn 판별에 얼마나 기여하는지 측정합니다.

방법:
  1. TF-Hub 원본 YAMNet (파인튜닝 없음) → 임베딩 추출 → 선형 프로브
  2. Layer 9 파인튜닝 적용 → 임베딩 추출 → 선형 프로브
  3. Layer 9-10 파인튜닝 적용 → ...
  4. Layer 9-14 파인튜닝 적용 (현재 모델) → ...

각 단계에서 siren recall / car_horn recall / 전체 정확도를 비교합니다.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CLASS_MAP = {1: "car_horn", 8: "siren", 99: "background"}
IDX = {1: 0, 8: 1, 99: 2}  # class_id → probe label index


# ─── 레이어 그룹 정의 ──────────────────────────────────────────
LAYER_GROUPS = {
    "layer9":       ["layer9__"],
    "layer9-10":    ["layer9__", "layer10__"],
    "layer9-11":    ["layer9__", "layer10__", "layer11__"],
    "layer9-12":    ["layer9__", "layer10__", "layer11__", "layer12__"],
    "layer9-13":    ["layer9__", "layer10__", "layer11__", "layer12__", "layer13__"],
    "layer9-14":    ["layer9__", "layer10__", "layer11__", "layer12__", "layer13__", "layer14__"],
}


def load_waveform(wav_path: Path, sr: int = 16000) -> np.ndarray | None:
    try:
        import soundfile as sf
        data, file_sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if file_sr != sr:
            import librosa
            data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
        if data.ndim == 2:
            data = data.mean(axis=1)
        return data.astype(np.float32)
    except Exception as e:
        log.warning("Failed to load %s: %s", wav_path, e)
        return None


def extract_embeddings_batch(model, wav_paths: list[Path],
                              sr: int = 16000) -> tuple[np.ndarray, list[int]]:
    """YAMNet 임베딩을 배치로 추출 (mean-pool)."""
    import tensorflow as tf

    embeddings, valid_idx = [], []
    for i, wp in enumerate(wav_paths):
        wf = load_waveform(wp, sr)
        if wf is None:
            continue
        _, emb, _ = model(tf.constant(wf))
        embeddings.append(emb.numpy().mean(axis=0))
        valid_idx.append(i)
    return np.stack(embeddings).astype(np.float32), valid_idx


def linear_probe(X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """로지스틱 회귀 선형 프로브로 클래스 판별력 측정."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, recall_score

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42,
                              multi_class="multinomial")
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)

    acc = (preds == y_test).mean()
    # siren=label 1, car_horn=label 0, background=label 2
    labels_present = np.unique(y_test)
    report = classification_report(y_test, preds,
                                   target_names=["car_horn", "siren", "background"],
                                   output_dict=True, zero_division=0)
    return {
        "accuracy": acc,
        "siren_recall": report.get("siren", {}).get("recall", 0.0),
        "car_horn_recall": report.get("car_horn", {}).get("recall", 0.0),
        "background_recall": report.get("background", {}).get("recall", 0.0),
        "siren_f1": report.get("siren", {}).get("f1-score", 0.0),
    }


def remap_labels(class_ids: np.ndarray) -> np.ndarray:
    """class_id (1,8,99) → probe label (0,1,2)."""
    out = np.zeros_like(class_ids)
    out[class_ids == 1]  = 0  # car_horn
    out[class_ids == 8]  = 1  # siren
    out[class_ids == 99] = 2  # background
    return out


def run_ablation(config_path: str = "config.yaml") -> list[dict]:
    import tensorflow as tf
    import tensorflow_hub as hub
    from src.utils.config import load_config

    cfg = load_config(config_path)
    sr = cfg["audio"]["sample_rate"]
    hub_url = cfg["yamnet"]["tfhub_url"]
    npz_path = PROJECT / cfg["yamnet"]["finetuned_dir"] / "yamnet_vars.npz"
    manifest_path = PROJECT / cfg["dataset"]["processed_dir"] / "manifest.csv"

    # ── 데이터 준비: train(fold 1-8 clean) / test(fold 10 clean) ──
    df = pd.read_csv(str(manifest_path))
    df_clean = df[df["snr_db"] == "clean"].copy()
    val_folds  = cfg["dataset"]["val_folds"]
    test_folds = cfg["dataset"]["test_folds"]
    train_df = df_clean[~df_clean["fold"].isin(val_folds + test_folds)]
    test_df  = df_clean[df_clean["fold"].isin(test_folds)]

    log.info("Train clean: %d  |  Test clean: %d", len(train_df), len(test_df))

    def paths_labels(sub_df):
        paths = [PROJECT / p.replace("\\", "/") for p in sub_df["path"]]
        labels = sub_df["class_id"].values
        return paths, labels

    train_paths, train_ids = paths_labels(train_df)
    test_paths, test_ids   = paths_labels(test_df)

    # ── 파인튜닝 변수 로드 ──
    npz = np.load(str(npz_path), allow_pickle=True)
    all_var_names = sorted(npz.files)
    log.info("Loaded %d fine-tuned variables from npz", len(all_var_names))

    results = []

    # ── Step 0: 원본 YAMNet (파인튜닝 없음) ──────────────────────────
    log.info("=" * 60)
    log.info("Step 0: 원본 YAMNet (파인튜닝 없음)")
    model = hub.load(hub_url)

    log.info("  Extracting train embeddings (%d files)…", len(train_paths))
    X_tr_raw, tr_valid = extract_embeddings_batch(model, train_paths, sr)
    y_tr = remap_labels(train_ids[tr_valid])

    log.info("  Extracting test embeddings (%d files)…", len(test_paths))
    X_te_raw, te_valid = extract_embeddings_batch(model, test_paths, sr)
    y_te = remap_labels(test_ids[te_valid])

    metrics = linear_probe(X_tr_raw, y_tr, X_te_raw, y_te)
    metrics["step"] = "원본 YAMNet\n(파인튜닝 없음)"
    metrics["layers_finetuned"] = "없음"
    results.append(metrics)
    log.info("  Accuracy=%.3f  siren_R=%.3f  car_horn_R=%.3f",
             metrics["accuracy"], metrics["siren_recall"], metrics["car_horn_recall"])

    # ── Step 1~6: 누적 레이어 파인튜닝 ────────────────────────────────
    yamnet_vars = {v: model.variables for v in []}
    # TF 변수를 이름으로 매핑
    var_map = {v.name.split(":")[0]: v for v in model.variables}

    for step_name, prefixes in LAYER_GROUPS.items():
        log.info("=" * 60)
        log.info("Step: %s", step_name)

        # 해당 레이어 그룹의 변수를 파인튜닝 값으로 교체
        applied = 0
        for var_name_key, val in npz.items():
            matched = any(var_name_key.startswith(p) or
                          any(p in var_name_key for p in prefixes)
                          for p in prefixes)
            if not matched:
                continue
            # TF 변수 이름 패턴에서 찾기
            for tf_var in model.variables:
                vn = tf_var.name
                # layer9__depthwise_conv__bn__beta_0 → yamnet/layer9/depthwise_conv/...
                # 이름 매핑 시도
                key_clean = var_name_key.replace("__", "/").rstrip("_0")
                if key_clean in vn or var_name_key.replace("__", "/") in vn:
                    tf_var.assign(val)
                    applied += 1
                    break

        log.info("  Applied %d variable updates", applied)

        log.info("  Extracting train embeddings…")
        X_tr, tr_v = extract_embeddings_batch(model, train_paths, sr)
        y_tr2 = remap_labels(train_ids[tr_v])

        log.info("  Extracting test embeddings…")
        X_te, te_v = extract_embeddings_batch(model, test_paths, sr)
        y_te2 = remap_labels(test_ids[te_v])

        metrics = linear_probe(X_tr, y_tr2, X_te, y_te2)
        metrics["step"] = step_name
        metrics["layers_finetuned"] = ", ".join(p.strip("_") for p in prefixes)
        results.append(metrics)
        log.info("  Accuracy=%.3f  siren_R=%.3f  car_horn_R=%.3f",
                 metrics["accuracy"], metrics["siren_recall"], metrics["car_horn_recall"])

    return results


def main(config_path: str = "config.yaml"):
    results = run_ablation(config_path)

    print("\n" + "=" * 80)
    print("LAYER ABLATION STUDY — 레이어별 파인튜닝 기여도")
    print("=" * 80)
    header = f"{'단계':<20} {'파인튜닝 레이어':<30} {'Acc':>7} {'Siren_R':>9} {'Horn_R':>8} {'Siren_F1':>9}"
    print(header)
    print("-" * 80)
    for r in results:
        print(f"{r['step']:<20} {r['layers_finetuned']:<30} "
              f"{r['accuracy']:>7.3f} {r['siren_recall']:>9.3f} "
              f"{r['car_horn_recall']:>8.3f} {r['siren_f1']:>9.3f}")

    # 가장 큰 Siren Recall 향상이 일어난 레이어 찾기
    recalls = [r["siren_recall"] for r in results]
    deltas = [recalls[i+1] - recalls[i] for i in range(len(recalls)-1)]
    best_layer_idx = int(np.argmax(deltas)) + 1
    print(f"\n★ Siren Recall 최대 향상: {results[best_layer_idx]['step']} "
          f"(+{deltas[best_layer_idx-1]:.3f})")

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    main(args.config)
