"""Shared audio utility functions used across the pipeline."""

from __future__ import annotations

import numpy as np


def rms_power(signal: np.ndarray) -> float:
    """Return RMS power of *signal* (epsilon-guarded)."""
    return float(np.sqrt(np.mean(signal ** 2) + 1e-9))


def peak_normalize(signal: np.ndarray) -> np.ndarray:
    """Normalise *signal* so the absolute peak equals 1.0."""
    peak = np.max(np.abs(signal))
    if peak > 1.0:
        return (signal / peak).astype(np.float32)
    return signal.astype(np.float32)


def compute_snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """Measure actual SNR in dB between *signal* and *noise*."""
    p_signal = rms_power(signal) ** 2
    p_noise = rms_power(noise) ** 2
    return 10.0 * float(np.log10(p_signal / p_noise))
