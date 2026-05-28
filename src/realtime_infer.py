"""Real-time ADAS sound detection inference engine.

Step 6 of the ADAS Sound Detector pipeline.

Thread architecture
-------------------
::

    PyAudio callback  ──(Queue)──► Inference thread
         (Thread 1)                    │
                                  RingBuffer (15,600 samples)
                                  TFLite INT8 interpreter
                                  AdaptiveThreshold
                                  MajorityVoter
                                  Alert callback  →  컬러 터미널 출력

CLI modes
---------
* Live microphone::

      python -m src.realtime_infer --model models/adas_detector_int8.tflite

* WAV file simulation (no microphone required)::

      python -m src.realtime_infer --wav test.wav

* Latency benchmark (100 runs, p50/p95/p99)::

      python -m src.realtime_infer --benchmark

개선사항:
  - 클래스 이름이 config.yaml의 target_classes에서 자동으로 로드됩니다.
    (--labels 기본값이 config 기반으로 자동 설정)
  - ANSI 색상 코드를 사용한 시각적 알림 출력 (외부 라이브러리 불필요).
  - config.yaml inference.alert_colors로 클래스별 색상 지정 가능.
  - load_config / setup_logging 중복 제거 → utils.config에서 import.
"""

from __future__ import annotations

import argparse
import collections
import logging
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Deque, Optional

import numpy as np

from .utils.config import get_class_names, load_config, setup_logging


# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------

_ANSI = {
    "red":     "\033[91m",
    "yellow":  "\033[93m",
    "green":   "\033[92m",
    "cyan":    "\033[96m",
    "magenta": "\033[95m",
    "bold":    "\033[1m",
    "reset":   "\033[0m",
}


def _colorize(text: str, *codes: str) -> str:
    """Wrap *text* with ANSI color/style codes."""
    prefix = "".join(_ANSI.get(c, "") for c in codes)
    return f"{prefix}{text}{_ANSI['reset']}"


def _print_alert(class_name: str, confidence: float, threshold: float, color: str = "yellow") -> None:
    """Print a visually prominent alert to stdout."""
    bar = "=" * 52
    line1 = _colorize(bar, "bold", color)
    line2 = _colorize(f"  ALERT: {class_name.upper()}", "bold", color)
    line3 = _colorize(f"  confidence={confidence:.3f}  threshold={threshold:.3f}", color)
    line4 = _colorize(bar, "bold", color)
    print(f"\n{line1}\n{line2}\n{line3}\n{line4}\n", flush=True)


# ---------------------------------------------------------------------------
# TFLite interpreter loader
# ---------------------------------------------------------------------------

def _load_interpreter_class() -> Any:
    """Return a TFLite Interpreter class (tflite_runtime preferred)."""
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore[import]
        return Interpreter
    except ImportError:
        pass

    try:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore[import]
        return Interpreter
    except ImportError:
        raise ImportError(
            "No TFLite runtime found. Install one of:\n"
            "  pip install tflite-runtime        # lightweight, recommended\n"
            "  pip install tensorflow             # full TF (includes TFLite)"
        )


# ---------------------------------------------------------------------------
# YAMNet embedding engine (TF-Hub)
# ---------------------------------------------------------------------------

class YAMNetEngine:
    """Loads YAMNet from TF-Hub and extracts mean-pooled 1024-d embeddings.

    Call ``embed(waveform)`` with a float32 waveform of any length ≥ 15600.
    YAMNet splits audio into 0.96s frames internally and mean-pools them.
    """

    def __init__(
        self,
        tfhub_url: str,
        logger: logging.Logger,
        finetuned_dir: Optional[str] = None,
    ) -> None:
        try:
            import tensorflow_hub as hub  # type: ignore[import]
            import tensorflow as tf        # type: ignore[import]  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "tensorflow and tensorflow_hub are required for YAMNet embedding.\n"
                "  pip install tensorflow tensorflow-hub"
            ) from exc

        logger.info("Loading YAMNet from %s …", tfhub_url)
        self._model = hub.load(tfhub_url)
        self._tf = tf

        # 파인튜닝 변수 파일이 있으면 가중치를 복원
        if finetuned_dir:
            from pathlib import Path as _Path
            npz_path = _Path(finetuned_dir) / "yamnet_vars.npz"
            if npz_path.exists():
                data = np.load(str(npz_path))
                # Discover variables via a dummy forward pass (same ordering as finetune.py)
                dummy = tf.constant(np.zeros(15600, dtype=np.float32))
                with tf.GradientTape() as _tape:
                    self._model(dummy)
                model_vars = {
                    v.name.replace("/", "__").replace(":", "_"): v
                    for v in _tape.watched_variables()
                }
                restored = 0
                for name, var in model_vars.items():
                    if name in data:
                        var.assign(data[name])
                        restored += 1
                logger.info(
                    "Fine-tuned YAMNet weights restored: %d/%d vars from %s",
                    restored, len(model_vars), npz_path,
                )
            else:
                logger.warning(
                    "finetuned_dir=%s specified but yamnet_vars.npz not found — "
                    "using original TF-Hub weights.", finetuned_dir
                )

        logger.info("YAMNet ready.")

    def embed(self, waveform: np.ndarray) -> np.ndarray:
        """Return mean-pooled 1024-d embedding for *waveform*.

        Args:
            waveform: Float32 mono array, shape ``(N,)``.  N ≥ 15600.

        Returns:
            Float32 array of shape ``(1024,)``.
        """
        wav_tf = self._tf.constant(waveform.astype(np.float32))
        _, embeddings, _ = self._model(wav_tf)
        return self._tf.reduce_mean(embeddings, axis=0).numpy()


# ---------------------------------------------------------------------------
# TFLite classifier engine (embedding → probabilities)
# ---------------------------------------------------------------------------

class TFLiteInferenceEngine:
    """Thin wrapper around the classifier-only TFLite model.

    Input:  1024-d YAMNet embedding (float32 or INT8).
    Output: float32 class probability vector.

    Pipeline in realtime_infer.py:
        waveform → YAMNetEngine.embed() → (1024,) → TFLiteInferenceEngine.infer()
    """

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {model_path}")

        Interpreter = _load_interpreter_class()
        self._interp = Interpreter(model_path=str(model_path))
        self._interp.allocate_tensors()

        inp = self._interp.get_input_details()[0]
        out = self._interp.get_output_details()[0]

        self._inp_idx: int = inp["index"]
        self._out_idx: int = out["index"]
        self._is_int8: bool = inp["dtype"] == np.int8

        def _quant(det: dict) -> tuple[float, int]:
            q = det.get("quantization_parameters", {})
            scales = q.get("scales", [])
            zeros = q.get("zero_points", [])
            s = (float(scales[0]) if len(scales) > 0 else 1.0) or 1.0
            z = int(zeros[0]) if len(zeros) > 0 else 0
            return s, z

        self._inp_scale, self._inp_zero = _quant(inp)
        self._out_scale, self._out_zero = _quant(out)
        self._num_classes: int = int(out["shape"][-1])

    def infer(self, embedding: np.ndarray) -> np.ndarray:
        """Classify a 1024-d embedding and return class probabilities.

        Args:
            embedding: Float32 array of shape ``(1024,)`` from YAMNetEngine.

        Returns:
            Float32 probability array of shape ``(num_classes,)``.
        """
        emb = embedding.reshape(1, -1).astype(np.float32)
        if self._is_int8:
            inp_tensor = np.round(emb / self._inp_scale + self._inp_zero).astype(np.int8)
        else:
            inp_tensor = emb

        self._interp.set_tensor(self._inp_idx, inp_tensor)
        self._interp.invoke()
        raw = self._interp.get_tensor(self._out_idx)

        if self._is_int8:
            return ((raw.astype(np.float32) - self._out_zero) * self._out_scale).flatten()
        return raw.astype(np.float32).flatten()

    @property
    def num_classes(self) -> int:
        return self._num_classes


# ---------------------------------------------------------------------------
# Ring Buffer
# ---------------------------------------------------------------------------

class RingBuffer:
    """Fixed-capacity numpy ring buffer (shift-and-overwrite, thread-safe)."""

    def __init__(self, capacity: int) -> None:
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._lock = threading.Lock()
        self._filled_samples = 0
        self._ready = False

    def push(self, chunk: np.ndarray) -> None:
        chunk = chunk.astype(np.float32)
        n = len(chunk)
        with self._lock:
            if n >= self._capacity:
                self._buf[:] = chunk[-self._capacity:]
            else:
                self._buf[:self._capacity - n] = self._buf[n:]
                self._buf[self._capacity - n:] = chunk
            self._filled_samples += n
            if not self._ready and self._filled_samples >= self._capacity:
                self._ready = True

    def read(self) -> np.ndarray:
        with self._lock:
            return self._buf.copy()

    @property
    def ready(self) -> bool:
        return self._ready


# ---------------------------------------------------------------------------
# Adaptive threshold
# ---------------------------------------------------------------------------

class AdaptiveThreshold:
    """Running adaptive threshold: ``clip(mean + k×std, base, max)``."""

    def __init__(
        self,
        history_size: int = 50,
        k: float = 2.0,
        base: float = 0.5,
        max_val: float = 0.9,
    ) -> None:
        self._history: Deque[float] = collections.deque(maxlen=history_size)
        self._k = k
        self._base = base
        self._max = max_val

    def update(self, confidence: float) -> None:
        self._history.append(float(confidence))

    @property
    def threshold(self) -> float:
        if len(self._history) < 2:
            return self._base
        arr = np.array(self._history, dtype=np.float32)
        raw = float(arr.mean()) + self._k * float(arr.std())
        return float(np.clip(raw, self._base, self._max))

    def is_detected(self, confidence: float) -> bool:
        return confidence >= self.threshold


# ---------------------------------------------------------------------------
# Majority voter (debouncer)
# ---------------------------------------------------------------------------

class MajorityVoter:
    """Fires only after *window* consecutive identical class IDs."""

    def __init__(self, window: int = 3) -> None:
        self._window = window
        self._history: Deque[int] = collections.deque(maxlen=window)

    def vote(self, class_id: int) -> bool:
        self._history.append(class_id)
        if len(self._history) < self._window:
            return False
        return len(set(self._history)) == 1

    def reset(self) -> None:
        self._history.clear()


# ---------------------------------------------------------------------------
# Inference worker
# ---------------------------------------------------------------------------

class InferenceWorker:
    """Consumes audio chunks from *audio_queue*, runs TFLite, fires alerts."""

    def __init__(
        self,
        engine: TFLiteInferenceEngine,
        yamnet: YAMNetEngine,
        ring: RingBuffer,
        audio_queue: "queue.Queue[np.ndarray]",
        cfg: dict,
        label_names: list[str],
        alert_callback: Optional[Callable[[str, float], None]] = None,
        stop_event: Optional[threading.Event] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        infer_cfg = cfg["inference"]
        thresh_cfg = infer_cfg["threshold"]

        self._engine = engine
        self._yamnet = yamnet
        self._ring = ring
        self._queue = audio_queue
        self._label_names = label_names
        self._alert_callback = alert_callback
        self._stop = stop_event or threading.Event()
        self._logger = logger or logging.getLogger(__name__)

        # 클래스별 alert 색상 (config에서 로드, 없으면 yellow)
        self._alert_colors: dict[str, str] = cfg["inference"].get("alert_colors", {})

        # Per-class adaptive thresholds.
        # Each non-background class gets its own AdaptiveThreshold instance
        # initialised with the class-specific base from config (or the global
        # initial if no per-class override exists).
        per_class_bases: dict[str, float] = thresh_cfg.get("per_class", {})
        self._class_thresholds: dict[int, AdaptiveThreshold] = {}
        for idx, name in enumerate(label_names):
            if name == "background":
                continue
            base = per_class_bases.get(name, thresh_cfg["initial"])
            self._class_thresholds[idx] = AdaptiveThreshold(
                history_size=thresh_cfg["history_size"],
                k=thresh_cfg["k"],
                base=base,
                max_val=thresh_cfg["max_value"],
            )
        self._voter = MajorityVoter(window=infer_cfg["debounce_frames"])
        self._latencies: list[float] = []

    def run(self) -> None:
        """Main loop — call this in a dedicated thread.

        Pipeline per inference step:
            ring_buffer(waveform) → YAMNet → (1024,) embedding → TFLite classifier → proba
        """
        self._logger.info("Inference worker started.")
        while not self._stop.is_set():
            try:
                chunk = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._ring.push(chunk)
            if not self._ring.ready:
                continue

            waveform = self._ring.read()

            t0 = time.perf_counter()
            embedding = self._yamnet.embed(waveform)   # YAMNet: ~20-30 ms
            proba = self._engine.infer(embedding)      # TFLite classifier: <1 ms
            latency_ms = (time.perf_counter() - t0) * 1000
            self._latencies.append(latency_ms)

            # ── Per-class independent detection ────────────────────────────
            # Each target class is checked against its own adaptive threshold.
            # This lets a class (e.g. siren) fire even when background has the
            # highest softmax probability — breaking the argmax-first bias that
            # suppresses weak-but-real siren signal at high-noise SNR.
            # Among classes that exceed their threshold, the most confident one wins.
            detected_class: Optional[int] = None
            detected_conf: float = 0.0
            detected_thresh: float = 0.0

            for class_idx, at in self._class_thresholds.items():
                conf = float(proba[class_idx])
                at.update(conf)
                if at.is_detected(conf) and conf > detected_conf:
                    detected_class = class_idx
                    detected_conf = conf
                    detected_thresh = at.threshold

            if detected_class is not None:
                if self._voter.vote(detected_class):
                    class_name = (
                        self._label_names[detected_class]
                        if detected_class < len(self._label_names)
                        else str(detected_class)
                    )
                    color = self._alert_colors.get(class_name, "yellow")
                    self._logger.warning(
                        "ALERT %-12s  conf=%.3f  threshold=%.3f  latency=%.1f ms",
                        class_name, detected_conf, detected_thresh, latency_ms,
                    )
                    _print_alert(class_name, detected_conf, detected_thresh, color)
                    if self._alert_callback:
                        self._alert_callback(class_name, detected_conf)
                    self._voter.reset()
            else:
                self._voter.reset()

        self._logger.info("Inference worker stopped.")

    def latency_stats(self) -> dict[str, float]:
        if not self._latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "count": 0}
        arr = np.array(self._latencies)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(arr.mean()),
            "count": len(arr),
        }


# ---------------------------------------------------------------------------
# PyAudio capture
# ---------------------------------------------------------------------------

class AudioCapture:
    """Streams microphone input into *audio_queue* using a PyAudio callback."""

    def __init__(
        self,
        audio_queue: "queue.Queue[np.ndarray]",
        sample_rate: int,
        hop_samples: int,
        device_index: Optional[int],
        logger: logging.Logger,
    ) -> None:
        self._queue = audio_queue
        self._sample_rate = sample_rate
        self._hop_samples = hop_samples
        self._device_index = device_index
        self._logger = logger
        self._stream: Any = None
        self._pa: Any = None

    def _callback(self, in_data: bytes, frame_count: int, time_info: Any, status: int):
        try:
            import pyaudio  # type: ignore[import]
        except ImportError:
            sys.exit(1)

        chunk = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            self._logger.debug("Audio queue full — chunk dropped.")

        return (None, pyaudio.paContinue)  # type: ignore[attr-defined]

    def start(self) -> None:
        try:
            import pyaudio  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("PyAudio is required: pip install pyaudio") from exc

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._sample_rate,
            input=True,
            frames_per_buffer=self._hop_samples,
            input_device_index=self._device_index,
            stream_callback=self._callback,
        )
        self._stream.start_stream()
        self._logger.info(
            "Audio capture started — device=%s  rate=%d  hop=%d",
            self._device_index, self._sample_rate, self._hop_samples,
        )

    def stop(self) -> None:
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()
        self._logger.info("Audio capture stopped.")


# ---------------------------------------------------------------------------
# WAV file simulation
# ---------------------------------------------------------------------------

def simulate_from_wav(
    wav_path: Path,
    audio_queue: "queue.Queue[np.ndarray]",
    sample_rate: int,
    hop_samples: int,
    logger: logging.Logger,
) -> None:
    """Feed a WAV file into *audio_queue* in real-time–paced chunks."""
    import soundfile as sf  # type: ignore[import]

    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != sample_rate:
        import librosa  # type: ignore[import]
        data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)

    n_chunks = len(data) // hop_samples
    logger.info("Simulating from %s — %.2f s, %d chunks", wav_path.name, len(data) / sample_rate, n_chunks)

    hop_duration_s = hop_samples / sample_rate
    for start in range(0, len(data) - hop_samples + 1, hop_samples):
        audio_queue.put(data[start : start + hop_samples])
        time.sleep(hop_duration_s)

    logger.info("WAV simulation complete.")


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def run_benchmark(
    engine: TFLiteInferenceEngine,
    yamnet: YAMNetEngine,
    ring_samples: int,
    n_runs: int,
    logger: logging.Logger,
) -> dict[str, float]:
    """Measure E2E inference latency (YAMNet + classifier) over *n_runs* runs."""
    rng = np.random.default_rng(0)
    waveform = rng.standard_normal(ring_samples).astype(np.float32) * 0.1

    # warm-up
    for _ in range(3):
        emb = yamnet.embed(waveform)
        engine.infer(emb)

    yamnet_lats: list[float] = []
    clf_lats: list[float] = []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        emb = yamnet.embed(waveform)
        t1 = time.perf_counter()
        engine.infer(emb)
        t2 = time.perf_counter()
        yamnet_lats.append((t1 - t0) * 1000)
        clf_lats.append((t2 - t1) * 1000)

    e2e = np.array(yamnet_lats) + np.array(clf_lats)
    stats = {
        "p50": float(np.percentile(e2e, 50)),
        "p95": float(np.percentile(e2e, 95)),
        "p99": float(np.percentile(e2e, 99)),
        "mean": float(e2e.mean()),
        "yamnet_mean_ms": float(np.mean(yamnet_lats)),
        "classifier_mean_ms": float(np.mean(clf_lats)),
        "count": n_runs,
    }

    logger.info("=" * 50)
    logger.info("BENCHMARK  (%d runs, waveform=%d samples)", n_runs, ring_samples)
    logger.info("  YAMNet  mean: %7.2f ms", stats["yamnet_mean_ms"])
    logger.info("  Classifier  : %7.3f ms", stats["classifier_mean_ms"])
    logger.info("  E2E  p50 : %7.2f ms", stats["p50"])
    logger.info("  E2E  p95 : %7.2f ms", stats["p95"])
    logger.info("  E2E  p99 : %7.2f ms", stats["p99"])
    logger.info("  E2E  mean: %7.2f ms", stats["mean"])
    logger.info("=" * 50)

    return stats


# ---------------------------------------------------------------------------
# Memory reporter
# ---------------------------------------------------------------------------

def report_memory(label: str, logger: logging.Logger) -> float:
    try:
        import psutil  # type: ignore[import]
        rss_mb = psutil.Process().memory_info().rss / (1024 ** 2)
        logger.info("Memory [%s]: %.1f MB RSS", label, rss_mb)
        return rss_mb
    except ImportError:
        logger.debug("psutil not installed — skipping memory report.")
        return -1.0


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def run_live(cfg: dict, args: argparse.Namespace, logger: logging.Logger) -> None:
    """Run the real-time microphone detection loop."""
    infer_cfg = cfg["inference"]
    audio_cfg = cfg["audio"]

    model_path = Path(args.model)
    sample_rate: int = audio_cfg["sample_rate"]
    ring_samples: int = infer_cfg["ring_buffer_samples"]
    hop_samples: int = infer_cfg["hop_samples"]

    if args.threshold is not None:
        infer_cfg["threshold"]["initial"] = args.threshold

    # 클래스 이름: CLI 인자 > config 자동 로드
    # background 클래스가 있으면 맨 뒤에 추가 (classifier와 동일한 순서 유지)
    label_names: list[str] = args.labels or get_class_names(cfg)
    if cfg["dataset"].get("background_classes") and "background" not in label_names:
        label_names = label_names + ["background"]
    logger.info("Class labels: %s", label_names)

    report_memory("startup", logger)

    yamnet = YAMNetEngine(
        cfg["yamnet"]["tfhub_url"], logger,
        finetuned_dir=cfg["yamnet"].get("finetuned_dir"),
    )
    engine = TFLiteInferenceEngine(model_path)
    ring = RingBuffer(ring_samples)
    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
    stop_event = threading.Event()

    worker = InferenceWorker(
        engine=engine, yamnet=yamnet, ring=ring, audio_queue=audio_queue,
        cfg=cfg, label_names=label_names,
        stop_event=stop_event, logger=logger,
    )

    infer_thread = threading.Thread(target=worker.run, daemon=True, name="inference")
    infer_thread.start()

    capture = AudioCapture(
        audio_queue=audio_queue, sample_rate=sample_rate,
        hop_samples=hop_samples, device_index=infer_cfg.get("device_index"),
        logger=logger,
    )

    def _shutdown(signum: int, frame: Any) -> None:
        logger.info("Shutting down …")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        capture.start()
        print(_colorize("Listening … press Ctrl+C to stop.", "cyan"), flush=True)
        while not stop_event.is_set():
            time.sleep(0.2)
    finally:
        stop_event.set()
        capture.stop()
        infer_thread.join(timeout=3)
        stats = worker.latency_stats()
        logger.info(
            "Session latency — p50=%.1f ms  p95=%.1f ms  p99=%.1f ms  (n=%d)",
            stats["p50"], stats["p95"], stats["p99"], stats["count"],
        )
        report_memory("shutdown", logger)


def run_wav_simulation(cfg: dict, args: argparse.Namespace, logger: logging.Logger) -> None:
    """Simulate real-time detection from a WAV file."""
    infer_cfg = cfg["inference"]
    audio_cfg = cfg["audio"]

    model_path = Path(args.model)
    wav_path = Path(args.wav)
    sample_rate: int = audio_cfg["sample_rate"]
    ring_samples: int = infer_cfg["ring_buffer_samples"]
    hop_samples: int = infer_cfg["hop_samples"]

    if args.threshold is not None:
        infer_cfg["threshold"]["initial"] = args.threshold

    label_names: list[str] = args.labels or get_class_names(cfg)
    if cfg["dataset"].get("background_classes") and "background" not in label_names:
        label_names = label_names + ["background"]
    logger.info("Class labels: %s", label_names)

    report_memory("startup", logger)

    yamnet = YAMNetEngine(
        cfg["yamnet"]["tfhub_url"], logger,
        finetuned_dir=cfg["yamnet"].get("finetuned_dir"),
    )
    engine = TFLiteInferenceEngine(model_path)
    ring = RingBuffer(ring_samples)
    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=128)
    stop_event = threading.Event()
    alerts: list[tuple[str, float]] = []

    def alert_cb(class_name: str, confidence: float) -> None:
        alerts.append((class_name, confidence))

    worker = InferenceWorker(
        engine=engine, yamnet=yamnet, ring=ring, audio_queue=audio_queue,
        cfg=cfg, label_names=label_names, alert_callback=alert_cb,
        stop_event=stop_event, logger=logger,
    )

    infer_thread = threading.Thread(target=worker.run, daemon=True, name="inference")
    infer_thread.start()

    simulate_from_wav(wav_path, audio_queue, sample_rate, hop_samples, logger)

    # 큐가 비워질 때까지 대기
    deadline = time.time() + 5.0
    while not audio_queue.empty() and time.time() < deadline:
        time.sleep(0.05)

    stop_event.set()
    infer_thread.join(timeout=3)

    stats = worker.latency_stats()
    logger.info(
        "WAV simulation done — alerts: %d  p50=%.1f ms  p95=%.1f ms  p99=%.1f ms",
        len(alerts), stats["p50"], stats["p95"], stats["p99"],
    )
    report_memory("done", logger)


def run_benchmark_mode(cfg: dict, args: argparse.Namespace, logger: logging.Logger) -> None:
    """Run latency benchmark without audio capture."""
    infer_cfg = cfg["inference"]
    model_path = Path(args.model)
    ring_samples: int = infer_cfg["ring_buffer_samples"]
    n_runs: int = infer_cfg.get("benchmark_runs", 100)

    report_memory("startup", logger)
    yamnet = YAMNetEngine(
        cfg["yamnet"]["tfhub_url"], logger,
        finetuned_dir=cfg["yamnet"].get("finetuned_dir"),
    )
    engine = TFLiteInferenceEngine(model_path)
    report_memory("model loaded", logger)

    run_benchmark(engine, yamnet, ring_samples, n_runs, logger)
    report_memory("done", logger)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADAS Sound Detector – Real-time Inference")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--model", default="models/adas_detector_int8.tflite",
        help="Path to TFLite model",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override base detection threshold (default: from config.yaml)",
    )
    parser.add_argument("--wav", default=None, help="WAV file for simulation mode")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Class label names in dense-label order. "
             "Defaults to target_classes from config.yaml.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config(args.config)
    logger = setup_logging(cfg, __name__)

    if args.benchmark:
        run_benchmark_mode(cfg, args, logger)
    elif args.wav:
        run_wav_simulation(cfg, args, logger)
    else:
        run_live(cfg, args, logger)
