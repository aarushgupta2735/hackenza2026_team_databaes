"""
Audio preprocessing utilities for the language classifier.

Handles: loading, resampling to 16kHz, normalization, silence trimming,
         and padding/trimming to a fixed segment length.
"""

import numpy as np
import librosa

from src.config.settings import SAMPLE_RATE, TARGET_SAMPLES


def load_audio(file_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file and resample to target sample rate.

    Returns a 1-D float32 numpy array.
    """
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    return y.astype(np.float32)


def normalize_audio(y: np.ndarray) -> np.ndarray:
    """Peak-normalize audio to [-1, 1]."""
    peak = np.max(np.abs(y))
    if peak > 0:
        return y / peak
    return y


def trim_silence(y: np.ndarray, top_db: int = 25) -> np.ndarray:
    """Remove leading/trailing silence using librosa."""
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed


def pad_or_trim(y: np.ndarray, target_len: int = TARGET_SAMPLES) -> np.ndarray:
    """Pad with zeros or trim to exactly `target_len` samples."""
    if len(y) > target_len:
        return y[:target_len]
    elif len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)), mode="constant")
    return y


def is_valid_audio(y: np.ndarray, min_duration_s: float = 0.5,
                   min_rms: float = 1e-5, sr: int = SAMPLE_RATE) -> bool:
    """Check whether an audio clip is long enough and not silent."""
    if len(y) < sr * min_duration_s:
        return False
    if np.sqrt(np.mean(y ** 2)) < min_rms:
        return False
    return True


def preprocess_audio(file_path: str, sr: int = SAMPLE_RATE,
                     target_len: int = TARGET_SAMPLES) -> np.ndarray | None:
    """Full preprocessing pipeline for a single audio file.

    Returns a fixed-length float32 array, or None if the clip is invalid.
    """
    y = load_audio(file_path, sr=sr)

    if not is_valid_audio(y, sr=sr):
        return None

    y = trim_silence(y)
    y = normalize_audio(y)
    y = pad_or_trim(y, target_len=target_len)

    return y