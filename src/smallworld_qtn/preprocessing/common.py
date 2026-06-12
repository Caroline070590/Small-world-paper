"""Shared signal-cleaning utilities used across the preprocessing pipelines.

These helpers are extracted verbatim from the project notebooks. Filtering uses
fourth-order Butterworth filters applied with zero-phase ``filtfilt`` /
``sosfiltfilt``, matching the Methods section of the paper. ``MAX_ABS_Z`` (the
clip applied after robust z-scoring) is exposed as a module-level constant so it
can be overridden per dataset if needed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import signal

# Default clip on the robust z-score (per the notebooks).
MAX_ABS_Z = 10.0
BUTTER_ORDER = 4


def robust_zscore(x: np.ndarray, max_abs_z: float = MAX_ABS_Z) -> np.ndarray:
    """Median/MAD z-score with a fallback to std and a symmetric clip."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale == 0:
        std = np.nanstd(x)
        scale = std if np.isfinite(std) and std > 0 else 1.0
    z = (x - med) / scale
    return np.clip(z, -max_abs_z, max_abs_z)


def flat_fraction(x: np.ndarray) -> float:
    """Fraction of (near-)constant consecutive samples."""
    dx = np.diff(x)
    if dx.size == 0:
        return 1.0
    return float(np.mean(np.abs(dx) < 1e-12))


def interpolate_short_gaps(x: np.ndarray, max_gap_frac: float = 0.01) -> np.ndarray:
    """Linearly interpolate non-finite samples, rejecting overly long gaps."""
    x = np.asarray(x, dtype=float).copy()
    finite = np.isfinite(x)
    n = len(x)

    if finite.sum() == 0:
        return x
    if finite.all():
        return x

    idx = np.arange(n)
    x_interp = x.copy()
    x_interp[~finite] = np.interp(idx[~finite], idx[finite], x[finite])

    gap_mask = ~finite
    if gap_mask.any():
        starts = np.where(np.diff(np.r_[False, gap_mask, False].astype(int)) == 1)[0]
        ends = np.where(np.diff(np.r_[False, gap_mask, False].astype(int)) == -1)[0]
        max_gap = max((e - s) for s, e in zip(starts, ends)) if len(starts) else 0
        if max_gap > max_gap_frac * n:
            raise ValueError(f"Gap too long for safe interpolation: {max_gap} samples.")
    return x_interp


def butter_filter(x: np.ndarray, fs: Optional[float], low: Optional[float],
                  high: Optional[float], order: int = BUTTER_ORDER) -> np.ndarray:
    """Zero-phase Butterworth band/high/low-pass (SOS form).

    Returns ``x`` unchanged if ``fs`` is unknown or the band is invalid.
    """
    if fs is None or not np.isfinite(fs) or fs <= 0:
        return x

    nyq = 0.5 * fs

    if low is not None and high is not None:
        if high >= nyq:
            high = nyq * 0.99
        if low <= 0 or low >= high:
            return x
        sos = signal.butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    elif low is not None:
        if low <= 0 or low >= nyq:
            return x
        sos = signal.butter(order, low / nyq, btype="highpass", output="sos")
    elif high is not None:
        if high <= 0 or high >= nyq:
            return x
        sos = signal.butter(order, high / nyq, btype="lowpass", output="sos")
    else:
        return x

    return signal.sosfiltfilt(sos, x)


def notch_filter_1d(x: np.ndarray, fs: float, f0: float, Q: float = 30.0) -> np.ndarray:
    """Single-frequency IIR notch (zero-phase)."""
    b, a = signal.iirnotch(w0=f0, Q=Q, fs=fs)
    return signal.filtfilt(b, a, x)


def epoch_ptp_keep_mask(x: np.ndarray, fs: Optional[float], epoch_sec: float,
                        ptp_threshold_mult: float = 10.0,
                        do_epoch_qc: bool = True) -> np.ndarray:
    """Robust peak-to-peak epoch rejection mask (median + k * MAD)."""
    if not do_epoch_qc:
        return np.ones(1, dtype=bool)
    if fs is None or not np.isfinite(fs) or fs <= 0:
        return np.ones(1, dtype=bool)

    epoch_len = max(1, int(round(epoch_sec * fs)))
    n_epochs = len(x) // epoch_len
    if n_epochs == 0:
        return np.zeros(0, dtype=bool)

    y = x[: n_epochs * epoch_len]
    epochs = y.reshape(n_epochs, epoch_len)
    ptp = np.ptp(epochs, axis=1)

    med = np.median(ptp)
    mad = np.median(np.abs(ptp - med))
    scale = 1.4826 * mad if mad > 0 else (np.std(ptp) if np.std(ptp) > 0 else 1.0)
    thr = med + ptp_threshold_mult * scale
    return ptp <= thr
