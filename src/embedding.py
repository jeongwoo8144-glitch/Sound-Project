"""YAMNet embedding extraction with file-level caching.

Step 3 of the ADAS Sound Detector pipeline.

Each WAV file in the manifest is processed once:
  waveform (64,000 samples @ 16 kHz)
    → YAMNet  →  (num_frames, 1024) per-frame embeddings
    → mean-pool →  (1024,) embedding vector
    → cached as .npy

On re-runs, cached .npy files are loaded directly, skipping YAMNet inference.
Final output: embeddings.npz  { X: (N, 1024), y: (N,), paths: (N,) }

개선사항:
  - load_config / setup_logging 중복 제거 → utils.config에서 import
  - 배치 처리: yamnet_cfg.batch_size 단위로 여러 파일을 순차 처리하되
    tqdm 진행률을 배치 단위로 갱신하여 전체 속도를 모니터링합니다.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils.config import load_config, setup_logging


# ---------------------------------------------------------------------------
# YAMNet loader
# ---------------------------------------------------------------------------

def load_yamnet(tfhub_url: str, logger: logging.Logger) -> Any:
    """Load YAMNet model from TensorFlow Hub.

    Args:
        tfhub_url: TF-Hub URL, e.g. 'https://tfhub.dev/google/yamnet/1'.
        logger: Logger instance.

    Returns:
        Loaded TF-Hub model callable.

    Raises:
        ImportError: If tensorflow or tensorflow_hub are not installed.
    """
    try:
        import tensorflow_hub as hub  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "tensorflow_hub is required. Install with: pip install tensorflow tensorflow-hub"
        ) from exc

    logger.info("Loading YAMNet from %s …", tfhub_url)
    model = hub.load(tfhub_url)
    logger.info("YAMNet loaded successfully.")
    return model


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embedding(model: Any, waveform: np.ndarray) -> np.ndarray:
    """Run YAMNet inference and return a mean-pooled 1024-d embedding.

    Args:
        model: Loaded YAMNet TF-Hub model.
        waveform: Float32 mono waveform at 16 kHz, shape ``(num_samples,)``.

    Returns:
        Float32 array of shape ``(1024,)``.

    Raises:
        ValueError: If the model returns an unexpected embedding shape.
    """
    import tensorflow as tf  # type: ignore[import]

    _, embeddings, _ = model(tf.constant(waveform))
    embeddings_np: np.ndarray = embeddings.numpy()

    if embeddings_np.ndim != 2 or embeddings_np.shape[1] != 1024:
        raise ValueError(
            f"Unexpected embedding shape from YAMNet: {embeddings_np.shape}. "
            "Expected (num_frames, 1024)."
        )

    return embeddings_np.mean(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Cache path helpers
# ---------------------------------------------------------------------------

def get_cache_path(
    wav_relative: str,
    processed_dir: Path,
    cache_dir: Path,
    project_root: Path | None = None,
) -> Path:
    """Map a manifest wav path to its corresponding .npy cache path.

    Args:
        wav_relative: Path string from manifest, relative to *project_root*.
        processed_dir: Absolute path to the processed data base directory.
        cache_dir: Absolute root directory for ``.npy`` cache files.
        project_root: Absolute project root used to resolve *wav_relative*.

    Returns:
        Absolute cache path with ``.npy`` suffix.
    """
    if project_root is not None:
        wav_abs = (project_root / wav_relative).resolve()
    else:
        wav_abs = Path(wav_relative).resolve()

    try:
        rel = wav_abs.relative_to(processed_dir.resolve())
    except ValueError:
        rel = Path(wav_abs.name)

    return (cache_dir / rel).with_suffix(".npy")


def _load_waveform(wav_path: Path, sample_rate: int) -> np.ndarray | None:
    try:
        import soundfile as sf  # type: ignore[import]

        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if sr != sample_rate:
            import librosa  # type: ignore[import]
            data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)
        if data.ndim == 2:
            data = data.mean(axis=1)
        return data.astype(np.float32)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_manifest(
    manifest_df: pd.DataFrame,
    model: Any,
    project_root: Path,
    processed_dir: Path,
    cache_dir: Path,
    sample_rate: int,
    logger: logging.Logger,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract embeddings for every entry in *manifest_df* with file-level caching.

    캐시 히트 파일은 .npy를 직접 로드합니다.
    캐시 미스 파일은 *batch_size* 단위로 묶어 진행률을 표시하면서 처리합니다.

    Args:
        manifest_df: DataFrame produced by data_pipeline.
        model: Loaded YAMNet model.
        project_root: Absolute path to the project root directory.
        processed_dir: Absolute path to ``data/processed``.
        cache_dir: Absolute path to the embedding cache directory.
        sample_rate: Audio sample rate (Hz).
        logger: Logger instance.
        batch_size: Number of files to process before logging a progress update.

    Returns:
        Tuple of (X: float32 (N, 1024), y: int32 (N,), paths: list[str]).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    embeddings_list: list[np.ndarray] = []
    labels_list: list[int] = []
    paths_list: list[str] = []

    n_cached = 0
    n_inferred = 0
    n_failed = 0

    rows = list(manifest_df.iterrows())
    pbar = tqdm(rows, total=len(rows), desc="Extracting embeddings", unit="file")

    for batch_start in range(0, len(rows), batch_size):
        batch = rows[batch_start : batch_start + batch_size]

        for _, row in batch:
            wav_rel: str = row["path"].replace("\\", "/")
            class_id: int = int(row["class_id"])
            cache_path = get_cache_path(wav_rel, processed_dir, cache_dir, project_root)

            # ── Cache hit ──
            if cache_path.exists():
                try:
                    embedding = np.load(str(cache_path))
                    if embedding.shape == (1024,):
                        embeddings_list.append(embedding)
                        labels_list.append(class_id)
                        paths_list.append(wav_rel)
                        n_cached += 1
                        pbar.update(1)
                        continue
                except Exception as exc:
                    logger.warning("Corrupt cache %s (%s) — re-inferring.", cache_path.name, exc)

            # ── Cache miss: load WAV + run YAMNet ──
            wav_abs = project_root / wav_rel
            waveform = _load_waveform(wav_abs, sample_rate)
            if waveform is None:
                logger.warning("Skipping (load failed): %s", wav_rel)
                n_failed += 1
                pbar.update(1)
                continue

            try:
                embedding = extract_embedding(model, waveform)
            except Exception as exc:
                logger.warning("Skipping (inference failed): %s – %s", wav_rel, exc)
                n_failed += 1
                pbar.update(1)
                continue

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_path), embedding)

            embeddings_list.append(embedding)
            labels_list.append(class_id)
            paths_list.append(wav_rel)
            n_inferred += 1
            pbar.update(1)

        # 배치 완료 후 진행 요약 출력
        done = batch_start + len(batch)
        pbar.set_postfix(cached=n_cached, inferred=n_inferred, failed=n_failed)

    pbar.close()

    logger.info(
        "Embedding extraction done — cached: %d  inferred: %d  failed: %d",
        n_cached, n_inferred, n_failed,
    )

    if not embeddings_list:
        raise RuntimeError("No embeddings were produced. Check your manifest and audio files.")

    X = np.stack(embeddings_list, axis=0).astype(np.float32)
    y = np.array(labels_list, dtype=np.int32)
    return X, y, paths_list


# ---------------------------------------------------------------------------
# Embedding statistics reporter
# ---------------------------------------------------------------------------

def report_embedding_stats(
    X: np.ndarray,
    y: np.ndarray,
    paths: list[str],
    logger: logging.Logger,
) -> None:
    """Log basic statistics of the embedding matrix for sanity-checking."""
    logger.info("=" * 60)
    logger.info("EMBEDDING STATS  X.shape=%s  dtype=%s", X.shape, X.dtype)
    logger.info("  global mean : %.4f", float(X.mean()))
    logger.info("  global std  : %.4f", float(X.std()))
    logger.info("  global min  : %.4f", float(X.min()))
    logger.info("  global max  : %.4f", float(X.max()))

    for class_id in np.unique(y):
        mask = y == class_id
        logger.info(
            "  class %d  N=%d  μ=%.4f  σ=%.4f",
            class_id, int(mask.sum()), float(X[mask].mean()), float(X[mask].std()),
        )

    rng = np.random.default_rng(0)
    idx = int(rng.integers(0, len(X)))
    emb = X[idx]
    logger.info(
        "  random sample [%d]: path=%s  mean=%.4f  std=%.4f",
        idx, paths[idx], float(emb.mean()), float(emb.std()),
    )
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_embedding(config_path: str) -> None:
    """Execute embedding extraction pipeline (Step 3).

    Args:
        config_path: Path to config.yaml.
    """
    cfg = load_config(config_path)
    logger = setup_logging(cfg, __name__)

    project_root = Path(config_path).parent
    ds_cfg = cfg["dataset"]
    audio_cfg = cfg["audio"]
    yamnet_cfg = cfg["yamnet"]

    processed_dir = project_root / ds_cfg["processed_dir"]
    cache_dir = project_root / yamnet_cfg["cache_dir"]
    sample_rate: int = audio_cfg["sample_rate"]
    batch_size: int = yamnet_cfg.get("batch_size", 32)

    manifest_path = processed_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run data_pipeline.py first."
        )
    manifest_df = pd.read_csv(str(manifest_path))
    logger.info("Manifest loaded: %d rows", len(manifest_df))

    model = load_yamnet(yamnet_cfg["tfhub_url"], logger)

    X, y, paths = process_manifest(
        manifest_df=manifest_df,
        model=model,
        project_root=project_root,
        processed_dir=processed_dir,
        cache_dir=cache_dir,
        sample_rate=sample_rate,
        logger=logger,
        batch_size=batch_size,
    )

    report_embedding_stats(X, y, paths, logger)

    npz_path = cache_dir / "embeddings.npz"
    np.savez(str(npz_path), X=X, y=y, paths=np.array(paths, dtype=object))
    logger.info("Saved embeddings → %s  (X=%s, y=%s)", npz_path, X.shape, y.shape)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADAS Sound Detector – YAMNet Embedding Extraction")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    run_embedding(_parse_args().config)
