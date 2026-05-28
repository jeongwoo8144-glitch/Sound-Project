"""End-to-end model integration and TFLite INT8 export.

Step 5 of the ADAS Sound Detector pipeline.

Pipeline:
    waveform [15600] (float32)
    → YAMNet   → (num_frames, 1024) embeddings
    → MeanPool → (1024,)
    → Classifier → (num_classes,) probabilities

Exports:
    models/saved_model/               ← TF SavedModel (reference)
    models/adas_detector_int8.tflite  ← INT8 quantized TFLite model
    models/quantization_report.json   ← accuracy / size / latency report

개선사항:
  - load_config / setup_logging 중복 제거 → utils.config에서 import.
  - label_map을 utils.config.get_class_map에서 자동 로드.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import tensorflow as tf  # type: ignore[import]  # must import before sklearn/soundfile on Windows
import numpy as np
import pandas as pd

from .utils.config import get_class_map, load_config, setup_logging


# ---------------------------------------------------------------------------
# End-to-end tf.Module
# ---------------------------------------------------------------------------

class EndToEndModel:
    """Wraps YAMNet + Custom Classifier as a single tf.Module."""

    def __init__(self, yamnet_model: Any, classifier_model: Any) -> None:
        import tensorflow as tf  # type: ignore[import]

        self._tf = tf
        self._yamnet = yamnet_model
        self._classifier = classifier_model
        self._module = self._build_module()

    def _build_module(self) -> Any:
        tf = self._tf
        yamnet = self._yamnet
        classifier = self._classifier

        class _Module(tf.Module):
            def __init__(self) -> None:
                super().__init__()
                # Assign as attributes so TF tracks all variables/resources
                self._yamnet = yamnet
                self._classifier = classifier

            @tf.function(input_signature=[tf.TensorSpec(shape=[64000], dtype=tf.float32)])
            def serving_fn(self, waveform: Any) -> Any:
                # 64,000 samples = 4.0 s @ 16 kHz — matches training clip length.
                # YAMNet produces ~6-7 frames from 4 s; mean-pool matches training.
                _, embeddings, _ = self._yamnet(waveform)
                embedding = tf.reduce_mean(embeddings, axis=0, keepdims=True)
                probabilities = self._classifier(embedding, training=False)
                return tf.squeeze(probabilities, axis=0)

        return _Module()

    @property
    def module(self) -> Any:
        return self._module


# ---------------------------------------------------------------------------
# Asset loading helpers
# ---------------------------------------------------------------------------

def load_yamnet(tfhub_url: str, logger: logging.Logger) -> Any:
    try:
        import tensorflow_hub as hub  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "tensorflow_hub is required: pip install tensorflow tensorflow-hub"
        ) from exc

    logger.info("Loading YAMNet from %s …", tfhub_url)
    model = hub.load(tfhub_url)
    logger.info("YAMNet loaded.")
    return model


def load_classifier(checkpoint_path: Path, logger: logging.Logger) -> Any:
    """Load the trained Keras classifier from checkpoint.

    Args:
        checkpoint_path: Path to ``custom_classifier.h5``.
        logger: Logger instance.

    Returns:
        Loaded ``tf.keras.Model``.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {checkpoint_path}\n"
            "Run classifier.py first."
        )

    try:
        import tensorflow as tf  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("TensorFlow is required.") from exc

    logger.info("Loading classifier from %s …", checkpoint_path)
    model = tf.keras.models.load_model(str(checkpoint_path))
    logger.info("Classifier loaded — input shape: %s", model.input_shape)
    return model


# ---------------------------------------------------------------------------
# Representative dataset
# ---------------------------------------------------------------------------

def load_calibration_waveforms(
    manifest_path: Path,
    project_root: Path,
    sample_rate: int,
    n_samples: int,
    test_folds: list[int],
    seed: int,
    logger: logging.Logger,
) -> np.ndarray:
    """Sample waveforms from training folds for INT8 calibration.

    Excludes test folds to avoid data contamination.

    Returns:
        Float32 array of shape ``(n_loaded, 64000)``.
    """
    import soundfile as sf  # type: ignore[import]

    manifest_df = pd.read_csv(str(manifest_path))
    calib_df = manifest_df[~manifest_df["fold"].isin(test_folds)].copy()
    calib_df = calib_df.sample(min(n_samples, len(calib_df)), random_state=seed)

    min_samples = 64000  # 4.0 s @ 16 kHz — matches serving_fn input & training clips
    waveforms: list[np.ndarray] = []

    for _, row in calib_df.iterrows():
        wav_path = project_root / str(row["path"]).replace("\\", "/")
        try:
            data, _ = sf.read(str(wav_path), dtype="float32", always_2d=False)
            if data.ndim == 2:
                data = data.mean(axis=1)
            if len(data) >= min_samples:
                data = data[:min_samples]
            else:
                data = np.pad(data, (0, min_samples - len(data)))
            waveforms.append(data.astype(np.float32))
        except Exception as exc:
            logger.debug("Skipping calibration file %s: %s", wav_path.name, exc)

    if not waveforms:
        raise RuntimeError(
            "No calibration waveforms could be loaded. "
            "Ensure processed WAV files exist in data/processed/."
        )

    result = np.stack(waveforms, axis=0)
    logger.info("Calibration dataset: %d waveforms, shape=%s", len(result), result.shape)
    return result


def make_representative_dataset(
    waveforms: np.ndarray,
) -> Callable[[], Iterator[list[np.ndarray]]]:
    """Create a representative dataset generator factory for TFLite INT8 calibration."""
    def generator() -> Iterator[list[np.ndarray]]:
        for wav in waveforms:
            yield [wav.astype(np.float32)]
    return generator


# ---------------------------------------------------------------------------
# SavedModel export
# ---------------------------------------------------------------------------

def export_saved_model(module: Any, saved_model_dir: Path, logger: logging.Logger) -> None:
    import tensorflow as tf  # type: ignore[import]

    saved_model_dir.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(
        module, str(saved_model_dir),
        signatures={"serving_default": module.serving_fn},
    )
    logger.info("SavedModel exported → %s", saved_model_dir)


# ---------------------------------------------------------------------------
# TFLite conversion (end-to-end SavedModel — kept for reference only)
# ---------------------------------------------------------------------------

def convert_to_tflite(
    saved_model_dir: Path,
    rep_dataset_fn: Callable[[], Iterator[list[np.ndarray]]],
    logger: logging.Logger,
) -> tuple[bytes, str]:
    """Convert SavedModel to TFLite with INT8 quantization.

    NOTE: YAMNet from TF-Hub contains READ_VARIABLE ops that prevent full
    INT8 quantization.  Falls back to hybrid (dynamic-range) quantization.

    Returns:
        Tuple of ``(tflite_flatbuffer_bytes, quantization_mode_used)``.
    """
    import tensorflow as tf  # type: ignore[import]

    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_dataset_fn
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_bytes = converter.convert()
        logger.info("INT8 full quantization succeeded.")
        return tflite_bytes, "int8"

    except Exception as exc:
        logger.warning(
            "INT8 full quantization failed (%s). "
            "Falling back to hybrid (dynamic-range) quantization.", exc,
        )

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_bytes = converter.convert()
    logger.warning("Using hybrid quantization — model size/latency will be larger than INT8.")
    return tflite_bytes, "hybrid"


# ---------------------------------------------------------------------------
# Classifier-only TFLite conversion
# ---------------------------------------------------------------------------

def convert_classifier_to_tflite(
    classifier: Any,
    X_calib: np.ndarray,
    logger: logging.Logger,
) -> tuple[bytes, str]:
    """Convert ONLY the Dense classifier to TFLite INT8.

    By separating YAMNet (TF-Hub) from the classifier (simple Dense DNN),
    we avoid the READ_VARIABLE op issue that prevents full INT8 quantization
    of the combined model.

    In realtime inference, YAMNet is called first to produce a 1024-d embedding,
    then this classifier TFLite model classifies it.

    Args:
        classifier: Trained ``tf.keras.Model`` (input: 1024-d embedding).
        X_calib: Embedding array ``(N, 1024)`` for INT8 calibration.
        logger: Logger instance.

    Returns:
        Tuple ``(tflite_flatbuffer_bytes, quantization_mode_used)``.
    """
    import tensorflow as tf  # type: ignore[import]

    def representative_dataset() -> Iterator[list[np.ndarray]]:
        for emb in X_calib:
            yield [emb.reshape(1, -1).astype(np.float32)]

    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(classifier)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_bytes = converter.convert()
        logger.info("Classifier INT8 quantization succeeded.")
        return tflite_bytes, "int8"
    except Exception as exc:
        logger.warning("INT8 failed for classifier (%s). Trying float16.", exc)

    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(classifier)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_bytes = converter.convert()
        logger.info("Classifier float16 quantization succeeded.")
        return tflite_bytes, "float16"
    except Exception as exc:
        logger.warning("float16 failed (%s). Using dynamic-range.", exc)

    converter = tf.lite.TFLiteConverter.from_keras_model(classifier)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    logger.warning("Classifier using dynamic-range quantization.")
    return tflite_bytes, "dynamic"


def evaluate_classifier_tflite(
    tflite_bytes: bytes,
    X_test: np.ndarray,
    y_test: np.ndarray,
    logger: logging.Logger,
) -> tuple[float, float]:
    """Evaluate classifier TFLite model on pre-computed embeddings.

    Returns:
        Tuple ``(accuracy, mean_latency_ms)``.
    """
    import tensorflow as tf  # type: ignore[import]

    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()

    def _quant(det: dict) -> tuple[float, int]:
        q = det.get("quantization_parameters", {})
        scales = q.get("scales", [])
        zeros = q.get("zero_points", [])
        s = (float(scales[0]) if len(scales) > 0 else 1.0) or 1.0
        z = int(zeros[0]) if len(zeros) > 0 else 0
        return s, z

    is_int8 = inp_det[0]["dtype"] == np.int8
    inp_scale, inp_zero = _quant(inp_det[0])
    out_scale, out_zero = _quant(out_det[0])

    y_pred: list[int] = []
    latencies: list[float] = []

    for emb in X_test:
        emb_f32 = emb.reshape(1, -1).astype(np.float32)
        emb_in = np.round(emb_f32 / inp_scale + inp_zero).astype(np.int8) if is_int8 else emb_f32

        interpreter.set_tensor(inp_det[0]["index"], emb_in)
        t0 = time.perf_counter()
        interpreter.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)

        out = interpreter.get_tensor(out_det[0]["index"])
        out_f32 = (out.astype(np.float32) - out_zero) * out_scale if is_int8 else out.astype(np.float32)
        y_pred.append(int(np.argmax(out_f32)))

    acc = float(np.mean(np.array(y_pred) == y_test))
    mean_lat = float(np.mean(latencies))
    logger.info("Classifier TFLite accuracy: %.4f  |  mean latency: %.3f ms", acc, mean_lat)
    return acc, mean_lat


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_keras_on_embeddings(
    classifier: Any, X_test: np.ndarray, y_test: np.ndarray, logger: logging.Logger,
) -> float:
    proba = classifier.predict(X_test, verbose=0)
    acc = float(np.mean(np.argmax(proba, axis=1) == y_test))
    logger.info("Keras (float32) test accuracy: %.4f", acc)
    return acc


def evaluate_tflite_on_waveforms(
    tflite_bytes: bytes,
    manifest_test: pd.DataFrame,
    project_root: Path,
    sample_rate: int,
    label_map: dict[int, int],
    logger: logging.Logger,
) -> tuple[float, float]:
    """Run the TFLite end-to-end model on raw test waveforms.

    Returns:
        Tuple ``(accuracy, mean_latency_ms_per_sample)``.
    """
    import soundfile as sf  # type: ignore[import]
    import tensorflow as tf  # type: ignore[import]

    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    def _quant(details):
        q = details.get("quantization_parameters", {})
        scales = q.get("scales", [])
        zeros = q.get("zero_points", [])
        scale = float(scales[0]) if len(scales) > 0 else 1.0
        zero = int(zeros[0]) if len(zeros) > 0 else 0
        return scale if scale != 0.0 else 1.0, zero

    inp_scale, inp_zero = _quant(input_details[0])
    out_scale, out_zero = _quant(output_details[0])
    is_int8 = input_details[0]["dtype"] == np.int8
    # Read expected input length from model (matches serving_fn input_signature)
    min_samples = int(np.prod(input_details[0]["shape"]))

    y_true: list[int] = []
    y_pred: list[int] = []
    latencies: list[float] = []

    for _, row in manifest_test.iterrows():
        wav_path = project_root / str(row["path"]).replace("\\", "/")
        try:
            data, _ = sf.read(str(wav_path), dtype="float32", always_2d=False)
            if data.ndim == 2:
                data = data.mean(axis=1)
            data = data[:min_samples] if len(data) >= min_samples else np.pad(data, (0, min_samples - len(data)))
        except Exception:
            continue

        wav_f32 = data.astype(np.float32)
        wav_in = np.round(wav_f32 / inp_scale + inp_zero).astype(np.int8) if is_int8 else wav_f32

        interpreter.set_tensor(input_details[0]["index"], wav_in)
        t0 = time.perf_counter()
        interpreter.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)

        out = interpreter.get_tensor(output_details[0]["index"])
        out_f32 = (out.astype(np.float32) - out_zero) * out_scale if is_int8 else out.astype(np.float32)

        y_pred.append(int(np.argmax(out_f32)))
        y_true.append(int(label_map[int(row["class_id"])]))

    if not y_true:
        logger.warning("No test waveforms could be loaded — skipping TFLite accuracy.")
        return -1.0, -1.0

    acc = float(np.mean(np.array(y_pred) == np.array(y_true)))
    mean_lat = float(np.mean(latencies))
    logger.info("TFLite (INT8) test accuracy: %.4f  |  mean latency: %.2f ms", acc, mean_lat)
    return acc, mean_lat


# ---------------------------------------------------------------------------
# Quantization report
# ---------------------------------------------------------------------------

def save_quantization_report(report: dict, report_path: Path, logger: logging.Logger) -> None:
    with open(str(report_path), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Quantization report → %s", report_path)


def _check_size(tflite_bytes: bytes, target_mb: float, logger: logging.Logger) -> float:
    size_mb = len(tflite_bytes) / (1024 ** 2)
    if size_mb > target_mb:
        logger.warning("TFLite model size %.2f MB exceeds target %.1f MB.", size_mb, target_mb)
    else:
        logger.info("TFLite model size: %.2f MB (target: %.1f MB) OK", size_mb, target_mb)
    return size_mb


# ---------------------------------------------------------------------------
# Main export pipeline
# ---------------------------------------------------------------------------

def run_export(config_path: str) -> None:
    """Execute model export pipeline (Step 5).

    Args:
        config_path: Path to config.yaml.
    """
    import tensorflow as tf  # type: ignore[import]

    cfg = load_config(config_path)
    logger = setup_logging(cfg, __name__)

    project_root = Path(config_path).parent
    ds_cfg = cfg["dataset"]
    export_cfg = cfg["export"]
    yamnet_cfg = cfg["yamnet"]
    audio_cfg = cfg["audio"]

    model_dir = project_root / export_cfg["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = project_root / ds_cfg["processed_dir"]
    npz_path = project_root / yamnet_cfg["cache_dir"] / "embeddings.npz"
    manifest_path = processed_dir / "manifest.csv"
    checkpoint_path = model_dir / export_cfg["keras_filename"]

    saved_model_dir = model_dir / "saved_model"
    tflite_path = model_dir / "adas_detector_int8.tflite"
    report_path = model_dir / "quantization_report.json"

    test_folds: list[int] = ds_cfg["test_folds"]
    sample_rate: int = audio_cfg["sample_rate"]
    seed: int = cfg["training"]["seed"]
    target_size_mb: float = export_cfg["target_size_mb"]

    # label_map: classifier.py의 build_label_map과 동일한 규칙 적용
    # target_classes (양성) 먼저, background(99) 맨 뒤
    # NOTE: get_class_map()은 이미 background(99)를 포함하므로 중복 추가 금지.
    BACKGROUND_ID = 99
    class_map = get_class_map(cfg)       # {1: 'car_horn', 8: 'siren', 99: 'background'}
    ordered_ids = sorted(cid for cid in class_map if cid != BACKGROUND_ID)  # [1, 8]
    if BACKGROUND_ID in class_map:
        ordered_ids = ordered_ids + [BACKGROUND_ID]        # background 맨 뒤: [1, 8, 99]
    label_map: dict[int, int] = {cid: i for i, cid in enumerate(ordered_ids)}
    # label_map: {1: 0, 8: 1, 99: 2}

    # ── 1. Load classifier ──
    classifier = load_classifier(checkpoint_path, logger)

    # ── 2. Load test embeddings ──
    data = np.load(str(npz_path), allow_pickle=True)
    X_all: np.ndarray = data["X"]
    y_all_raw: np.ndarray = data["y"]
    paths_all: np.ndarray = data["paths"].astype(str)

    manifest_df = pd.read_csv(str(manifest_path))
    manifest_df["path"] = manifest_df["path"].str.replace("\\", "/", regex=False)
    path_to_fold = dict(zip(manifest_df["path"], manifest_df["fold"]))
    paths_norm = np.array([p.replace("\\", "/") for p in paths_all])

    test_mask = np.array([path_to_fold.get(p, -1) in test_folds for p in paths_norm])
    train_mask = ~test_mask
    X_test = X_all[test_mask]
    y_test = np.vectorize(label_map.get)(y_all_raw[test_mask]).astype(np.int32)
    X_calib = X_all[train_mask]  # training embeddings for INT8 calibration

    # ── 3. Keras accuracy baseline ──
    keras_acc = evaluate_keras_on_embeddings(classifier, X_test, y_test, logger)

    # ── 4. Export SavedModel (end-to-end reference, optional) ──
    try:
        yamnet = load_yamnet(yamnet_cfg["tfhub_url"], logger)
        logger.info("Building end-to-end SavedModel (YAMNet + classifier) …")
        e2e = EndToEndModel(yamnet_model=yamnet, classifier_model=classifier)
        export_saved_model(e2e.module, saved_model_dir, logger)
    except Exception as exc:
        logger.warning("SavedModel export skipped: %s", exc)

    # ── 5. Convert classifier-only to TFLite INT8 ──
    # NOTE: Only the Dense classifier is quantized — YAMNet stays in TF-Hub.
    # In realtime_infer.py, YAMNet extracts embeddings first, then this
    # TFLite model classifies them.  This avoids READ_VARIABLE op issues.
    logger.info("Converting classifier → TFLite …")
    tflite_bytes, quant_mode = convert_classifier_to_tflite(classifier, X_calib, logger)
    tflite_path.write_bytes(tflite_bytes)
    logger.info("Classifier TFLite saved → %s", tflite_path)

    size_mb = _check_size(tflite_bytes, target_size_mb, logger)

    # ── 6. TFLite accuracy on test embeddings ──
    tflite_acc, mean_latency_ms = evaluate_classifier_tflite(
        tflite_bytes=tflite_bytes,
        X_test=X_test,
        y_test=y_test,
        logger=logger,
    )

    acc_drop = keras_acc - tflite_acc if tflite_acc >= 0 else None
    if acc_drop is not None and acc_drop > 0.02:
        logger.warning(
            "Quantization accuracy drop %.2f%% exceeds 2%% threshold "
            "(Keras=%.4f  TFLite=%.4f).",
            acc_drop * 100, keras_acc, tflite_acc,
        )

    # ── 7. Latency benchmark (classifier only) ──
    # YAMNet latency on RPi 4B is ~20-30 ms for 4s waveform (separate).
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()

    is_int8_bench = inp_det[0]["dtype"] == np.int8
    q_b = inp_det[0].get("quantization_parameters", {})
    _s = q_b.get("scales", [])
    _z = q_b.get("zero_points", [])
    bench_scale = (float(_s[0]) if len(_s) > 0 else 1.0) or 1.0
    bench_zero = int(_z[0]) if len(_z) > 0 else 0

    test_emb = X_calib[0].reshape(1, -1).astype(np.float32)
    test_in = np.round(test_emb / bench_scale + bench_zero).astype(np.int8) if is_int8_bench else test_emb

    interpreter.set_tensor(inp_det[0]["index"], test_in)
    interpreter.invoke()  # warm-up

    t_runs: list[float] = []
    for _ in range(20):
        interpreter.set_tensor(inp_det[0]["index"], test_in)
        t0 = time.perf_counter()
        interpreter.invoke()
        t_runs.append((time.perf_counter() - t0) * 1000)

    estimated_latency_ms = float(np.median(t_runs))
    target_ms: float = cfg["inference"]["latency_target_ms"]
    logger.info(
        "Classifier TFLite latency (CPU): %.3f ms (median 20 runs)  "
        "[YAMNet ~20-30 ms separate]",
        estimated_latency_ms,
    )
    if estimated_latency_ms > target_ms:
        logger.warning(
            "Classifier latency %.2f ms exceeds target %.0f ms.",
            estimated_latency_ms, target_ms,
        )

    # ── 8. Save report ──
    report = {
        "architecture": "classifier_only_tflite",
        "note": "YAMNet stays as TF-Hub; only the Dense classifier is TFLite-quantized.",
        "quantization_mode": quant_mode,
        "num_classes": len(label_map),
        "class_map": {str(k): v for k, v in class_map.items()},
        "label_map": {str(k): v for k, v in label_map.items()},
        "model_size_mb": round(size_mb, 4),
        "target_size_mb": target_size_mb,
        "size_ok": size_mb <= target_size_mb,
        "keras_test_accuracy": round(keras_acc, 6),
        "tflite_test_accuracy": round(tflite_acc, 6) if tflite_acc >= 0 else None,
        "accuracy_drop": round(float(acc_drop), 6) if acc_drop is not None else None,
        "accuracy_drop_exceeds_threshold": bool(acc_drop is not None and acc_drop > 0.02),
        "classifier_latency_ms": round(estimated_latency_ms, 3),
        "latency_target_ms": target_ms,
        "latency_ok": estimated_latency_ms <= target_ms,
        "n_calibration_embeddings": len(X_calib),
    }
    save_quantization_report(report, report_path, logger)

    logger.info("=" * 60)
    logger.info("EXPORT COMPLETE")
    logger.info("  Mode         : classifier-only TFLite  (YAMNet via TF-Hub)")
    logger.info("  Quantization : %s", quant_mode)
    logger.info("  Classes      : %s", list(class_map.values()))
    logger.info("  Size         : %.4f MB  (classifier only)", size_mb)
    logger.info("  Keras acc    : %.4f", keras_acc)
    if tflite_acc >= 0:
        logger.info("  TFLite acc   : %.4f  (drop: %.2f%%)", tflite_acc, (acc_drop or 0) * 100)
    logger.info("  Classifier latency: %.3f ms  [YAMNet ~20-30 ms separate]", estimated_latency_ms)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADAS Sound Detector – Model Export + TFLite Conversion")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    run_export(_parse_args().config)
