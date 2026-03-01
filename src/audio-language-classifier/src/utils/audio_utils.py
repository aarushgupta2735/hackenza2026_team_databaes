"""
Audio utility functions: load, save, validate, visualize.
"""

import os
import numpy as np
import librosa
import soundfile as sf

from src.config.settings import SAMPLE_RATE, TARGET_SAMPLES


def load_audio(file_path: str, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load an audio file, returning (waveform, sample_rate)."""
    y, sr_out = librosa.load(file_path, sr=sr, mono=True)
    return y.astype(np.float32), sr_out


def save_audio(file_path: str, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Save a waveform to a file (wav, flac, etc.)."""
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    sf.write(file_path, audio, sr)


def get_duration(file_path: str) -> float:
    """Return the duration of an audio file in seconds."""
    return librosa.get_duration(path=file_path)


def get_rms(y: np.ndarray) -> float:
    """Return the RMS energy of a waveform."""
    return float(np.sqrt(np.mean(y ** 2)))


def is_valid_audio_file(file_path: str) -> bool:
    """Check if a file is a readable audio file."""
    try:
        y, sr = librosa.load(file_path, sr=None, duration=0.5)
        return len(y) > 0
    except Exception:
        return False


def list_audio_files(directory: str, extensions: tuple = (".wav", ".mp3", ".flac", ".ogg")) -> list[str]:
    """Recursively list all audio files in a directory."""
    files = []
    for root, _, filenames in os.walk(directory):
        for fname in filenames:
            if fname.lower().endswith(extensions):
                files.append(os.path.join(root, fname))
    return sorted(files)


def visualize_audio(audio: np.ndarray, sr: int = SAMPLE_RATE,
                    title: str = "Waveform", save_path: str | None = None):
    """Plot the waveform. Optionally save to file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.linspace(0, len(audio) / sr, num=len(audio))
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, audio, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()