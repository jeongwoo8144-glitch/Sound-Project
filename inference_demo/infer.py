"""
ADAS Sound Detector — Windows 추론 스크립트
==========================================
사용법:
  # 오디오 파일 분류
  python infer.py --file audio.wav

  # 마이크 실시간 감지
  python infer.py --mic

  # 마이크 + 감도 조정 (0.0~1.0, 낮을수록 민감)
  python infer.py --mic --threshold 0.3

필요 설치:
  pip install tensorflow tensorflow-hub librosa sounddevice numpy
"""

import argparse
import sys
import time
import numpy as np

# ── 설정 ────────────────────────────────────────────────────────
MODEL_PATH   = "adas_detector.tflite"   # 스크립트와 같은 폴더에 위치
SAMPLE_RATE  = 16000                    # YAMNet 요구사항
WINDOW_SEC   = 4.0                      # 분석 윈도우 (초)
HOP_SEC      = 0.5                      # 마이크 모드 슬라이딩 간격
CLASSES      = ["배경", "경적 (Car Horn)", "사이렌 (Siren)"]
ALERT_EMOJI  = ["⬜", "📯", "🚨"]

# ── 의존성 확인 ─────────────────────────────────────────────────
def check_deps():
    missing = []
    for pkg in ["tensorflow", "tensorflow_hub", "librosa"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("❌ 다음 패키지가 없습니다. 설치 후 재실행하세요:")
        print(f"   pip install {' '.join(missing)}")
        sys.exit(1)

# ── 모델 / YAMNet 로드 ──────────────────────────────────────────
def load_models():
    import tensorflow as tf
    import tensorflow_hub as hub

    print("📦 YAMNet 로딩 중...", end=" ", flush=True)
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    print("완료")

    print("📦 분류기 TFLite 로딩 중...", end=" ", flush=True)
    interp = tf.lite.Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    print("완료\n")

    return yamnet, interp


def get_embedding(yamnet, waveform_np: np.ndarray) -> np.ndarray:
    """1D float32 waveform → YAMNet 임베딩 (평균 풀링)"""
    import tensorflow as tf
    waveform = tf.constant(waveform_np, dtype=tf.float32)
    _, embeddings, _ = yamnet(waveform)
    return embeddings.numpy().mean(axis=0)   # (1024,)


def classify(yamnet, interp, waveform_np: np.ndarray, threshold: float):
    """파형 → 클래스 예측"""
    emb = get_embedding(yamnet, waveform_np)

    inp  = interp.get_input_details()
    outp = interp.get_output_details()
    interp.set_tensor(inp[0]['index'], emb.reshape(1, -1).astype(np.float32))
    interp.invoke()
    probs = interp.get_tensor(outp[0]['index'])[0]

    pred_idx  = int(np.argmax(probs))
    pred_conf = float(probs[pred_idx])

    # threshold 이하면 배경으로 처리
    if pred_conf < threshold and pred_idx != 0:
        pred_idx  = 0
        pred_conf = float(probs[0])

    return pred_idx, pred_conf, probs


# ── 파일 모드 ───────────────────────────────────────────────────
def run_file(yamnet, interp, filepath: str, threshold: float):
    import librosa
    print(f"🎵 파일 로딩: {filepath}")
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)

    n = int(SAMPLE_RATE * WINDOW_SEC)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    else:
        y = y[:n]

    print(f"   길이: {len(y)/SAMPLE_RATE:.1f}초  |  threshold: {threshold}\n")

    pred_idx, conf, probs = classify(yamnet, interp, y, threshold)

    print("─" * 40)
    print("  분류 결과")
    print("─" * 40)
    for i, (cls, prob) in enumerate(zip(CLASSES, probs)):
        bar   = "█" * int(prob * 30)
        mark  = " ← 예측" if i == pred_idx else ""
        emoji = ALERT_EMOJI[i]
        print(f"  {emoji} {cls:<20} {prob*100:5.1f}%  {bar}{mark}")
    print("─" * 40)

    if pred_idx == 0:
        print("✅ 결과: 일반 배경음입니다.")
    else:
        print(f"⚠️  결과: {ALERT_EMOJI[pred_idx]} {CLASSES[pred_idx]} 감지! (신뢰도: {conf*100:.1f}%)")


# ── 마이크 실시간 모드 ──────────────────────────────────────────
def run_mic(yamnet, interp, threshold: float):
    try:
        import sounddevice as sd
    except ImportError:
        print("❌ sounddevice 미설치:  pip install sounddevice")
        sys.exit(1)

    n_window = int(SAMPLE_RATE * WINDOW_SEC)
    n_hop    = int(SAMPLE_RATE * HOP_SEC)
    buf      = np.zeros(n_window, dtype=np.float32)
    lock_msg = ""

    print(f"🎙️  마이크 실시간 감지 시작  (threshold={threshold}, Ctrl+C로 종료)\n")
    print("─" * 50)

    def callback(indata, frames, t, status):
        nonlocal buf, lock_msg
        chunk = indata[:, 0].astype(np.float32)
        buf   = np.roll(buf, -len(chunk))
        buf[-len(chunk):] = chunk

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        blocksize=n_hop, callback=callback):
        try:
            last_alert = ""
            while True:
                time.sleep(HOP_SEC)
                pred_idx, conf, probs = classify(yamnet, interp, buf.copy(), threshold)
                ts = time.strftime("%H:%M:%S")

                if pred_idx == 0:
                    status = "⬜ 배경음"
                    if last_alert:
                        last_alert = ""
                else:
                    status = f"{ALERT_EMOJI[pred_idx]} {CLASSES[pred_idx]} ({conf*100:.0f}%)"
                    if last_alert != CLASSES[pred_idx]:
                        print(f"\n{'='*50}")
                        print(f"  ⚠️  ALERT: {ALERT_EMOJI[pred_idx]} {CLASSES[pred_idx]}!")
                        print(f"  신뢰도: {conf*100:.1f}%")
                        print(f"{'='*50}")
                        last_alert = CLASSES[pred_idx]

                sys.stdout.write(f"\r  [{ts}]  {status:<35}")
                sys.stdout.flush()
        except KeyboardInterrupt:
            print("\n\n감지 종료.")


# ── 진입점 ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ADAS Sound Detector")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", metavar="WAV", help="분류할 오디오 파일 경로")
    group.add_argument("--mic",  action="store_true", help="마이크 실시간 감지")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="감지 임계값 0.0~1.0 (기본 0.4)")
    args = parser.parse_args()

    check_deps()
    yamnet, interp = load_models()

    if args.file:
        run_file(yamnet, interp, args.file, args.threshold)
    else:
        run_mic(yamnet, interp, args.threshold)


if __name__ == "__main__":
    main()
