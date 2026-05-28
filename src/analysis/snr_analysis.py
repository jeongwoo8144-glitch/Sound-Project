"""
SNR Sensitivity Analysis — 노이즈 크기별 감지 확률 분석

교수 피드백:
  - 노이즈 크기에 따라서 크고 작고 없고의 확률 차이 분석

각 SNR 조건에서 모델이 siren/car_horn/background를 얼마나 잘 감지하는지 측정.
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

CLASS_ID2NAME = {1: "car_horn", 8: "siren", 99: "background"}
CLASS_ID2IDX  = {1: 0, 8: 1, 99: 2}   # classifier output index

SNR_LEVELS_DB = [None, 20, 15, 10, 5, 0, -5, -10]  # None = clean

def load_wav(path, sr=16000):
    try:
        import soundfile as sf
        data, fsr = sf.read(str(path), dtype="float32", always_2d=False)
        if fsr != sr:
            import librosa
            data = librosa.resample(data, orig_sr=fsr, target_sr=sr)
        return (data.mean(axis=1) if data.ndim == 2 else data).astype(np.float32)
    except Exception as e:
        return None

def mix_snr(signal, noise, snr_db):
    """signal + noise를 목표 SNR로 합성."""
    sp = np.mean(signal**2) + 1e-9
    np_ = np.mean(noise**2) + 1e-9
    scale = np.sqrt(sp / (10**(snr_db/10)) / np_)
    out = signal + scale * noise[:len(signal)]
    mx = np.max(np.abs(out))
    return (out / mx).astype(np.float32) if mx > 1 else out

def get_noise_clips(straffic_dir, sr, clip_len):
    """STRAFFIC에서 노이즈 클립 풀 생성."""
    clips = []
    for wav in sorted(Path(straffic_dir).glob("*.wav"))[:6]:
        data = load_wav(wav, sr)
        if data is None: continue
        for s in range(0, len(data)-clip_len, clip_len):
            clips.append(data[s:s+clip_len])
    if not clips:
        rng = np.random.default_rng(0)
        clips = [rng.normal(0, 0.03, clip_len).astype(np.float32) for _ in range(30)]
        log.warning("STRAFFIC 없음 → 가우시안 노이즈 사용")
    return clips

def main(config_path="config.yaml"):
    import tensorflow as tf, tensorflow_hub as hub
    from src.utils.config import load_config

    cfg  = load_config(config_path)
    sr   = cfg["audio"]["sample_rate"]

    # ── 모델 로드 ──
    log.info("YAMNet 로드 중...")
    yamnet = hub.load(cfg["yamnet"]["tfhub_url"])

    # fine-tuned 가중치 적용
    npz_path = PROJECT / cfg["yamnet"]["finetuned_dir"] / "yamnet_vars.npz"
    npz = np.load(str(npz_path), allow_pickle=True)
    applied = 0
    try:
        variables = list(yamnet.variables)
    except AttributeError:
        try:
            variables = list(yamnet.trainable_variables) + list(yamnet.non_trainable_variables)
        except AttributeError:
            variables = []
            log.warning("yamnet.variables 접근 불가 — 가중치 적용 건너뜀")
    for var in variables:
        vn = var.name
        for key, val in npz.items():
            # 이름 변환: layer9__pointwise_conv__kernel_0 → layer9/pointwise_conv/kernel
            parts = key.rstrip("_0").split("__")
            candidate = "/".join(parts)
            if candidate in vn or key.rstrip("_0").replace("__","/") in vn:
                try:
                    var.assign(val)
                    applied += 1
                except: pass
                break
    log.info("Fine-tuned 가중치 %d개 적용", applied)

    log.info("분류기 로드 중...")
    clf = tf.keras.models.load_model(
        str(PROJECT / "models" / "custom_classifier_finetuned.h5"))

    # ── 테스트 데이터 ──
    manifest = pd.read_csv(str(PROJECT / cfg["dataset"]["processed_dir"] / "manifest.csv"))
    test_df = manifest[
        manifest["fold"].isin(cfg["dataset"]["test_folds"]) &
        (manifest["snr_db"] == "clean")
    ].copy()
    log.info("테스트 파일 %d개", len(test_df))

    # ── 노이즈 풀 ──
    clip_len = int(sr * 4.0)
    noise_pool = get_noise_clips(PROJECT / cfg["dataset"]["straffic_dir"], sr, clip_len)
    rng = np.random.default_rng(42)

    rows = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        wav_path = PROJECT / row["path"].replace("\\","/")
        signal = load_wav(wav_path, sr)
        if signal is None: continue
        class_id   = int(row["class_id"])
        class_name = CLASS_ID2NAME.get(class_id, "?")
        target_idx = CLASS_ID2IDX.get(class_id, -1)

        for snr in SNR_LEVELS_DB:
            if snr is None:
                audio, label = signal.copy(), "clean"
            else:
                noise = rng.choice(noise_pool)
                if len(noise) < len(signal):
                    noise = np.tile(noise, int(np.ceil(len(signal)/len(noise))))
                audio = mix_snr(signal, noise, snr)
                label = f"{snr:+d}dB"

            _, emb, _ = yamnet(tf.constant(audio))
            emb_mean  = emb.numpy().mean(axis=0, keepdims=True)
            probs     = clf.predict(emb_mean, verbose=0)[0]
            pred_idx  = int(np.argmax(probs))
            correct   = (pred_idx == target_idx)

            rows.append({
                "class": class_name, "snr": label,
                "snr_val": 999 if snr is None else snr,
                "p_horn": float(probs[0]),
                "p_siren": float(probs[1]),
                "p_bg": float(probs[2]),
                "target_prob": float(probs[target_idx]) if target_idx>=0 else 0,
                "correct": correct,
            })

        if (i+1) % 30 == 0:
            log.info("  %d/%d 처리 완료", i+1, len(test_df))

    df = pd.DataFrame(rows)

    # ── 요약 ──
    order = ["clean","+20dB","+15dB","+10dB","+5dB","0dB","-5dB","-10dB"]
    print("\n" + "="*85)
    print("  SNR 노이즈 크기별 감지 확률 분석")
    print("="*85)

    for cls in ["siren", "car_horn", "background"]:
        sub = df[df["class"]==cls]
        print(f"\n[{cls.upper()}]  (N={len(sub)//len(SNR_LEVELS_DB)}개)")
        print(f"  {'SNR':>8}  {'평균확률':>8}  {'감지율(Recall)':>14}  시각화")
        print("  "+"-"*65)
        for snr_label in order:
            s = sub[sub["snr"]==snr_label]
            if s.empty: continue
            mean_p = s["target_prob"].mean()
            recall = s["correct"].mean()
            bar = "#"*int(recall*25) + "."*(25-int(recall*25))
            marker = " ← 현재threshold" if (cls=="siren" and snr_label=="clean") else ""
            print(f"  {snr_label:>8}  {mean_p:>8.3f}  {recall:>14.3f}  |{bar}|{marker}")

    print("\n★ 핵심 인사이트:")
    siren_clean = df[(df["class"]=="siren")&(df["snr"]=="clean")]["correct"].mean()
    siren_10    = df[(df["class"]=="siren")&(df["snr"]=="+10dB")]["correct"].mean()
    siren_0     = df[(df["class"]=="siren")&(df["snr"]=="0dB")]["correct"].mean()
    siren_m5    = df[(df["class"]=="siren")&(df["snr"]=="-5dB")]["correct"].mean()
    print(f"  Siren Recall: clean={siren_clean:.3f}  +10dB={siren_10:.3f}  "
          f"0dB={siren_0:.3f}  -5dB={siren_m5:.3f}")
    print(f"  → 노이즈 증가 시 siren recall {'감소' if siren_0<siren_clean else '유지'}")

    return df

if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    os.chdir(PROJECT)
    main(args.config)
