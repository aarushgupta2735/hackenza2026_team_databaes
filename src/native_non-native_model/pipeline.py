"""
pipeline.py — Native / Non-Native Arabic Speaker Classification
================================================================
Full pipeline converted from the Colab notebook
(wave2vec2_lstm+rbf_svm.ipynb).

Stages:
  0. Setup / configuration
  1. Data exploration
  2. Data collection (Renan downloads + optional Common Voice)
  3. Preprocessing (resample → RMS normalize → VAD trim → denoise → duration gate)
  4. Audio splitting (3 s windows, 1 s hop)
  5. Data augmentation (5× minority, 1× majority)
  6A. wav2vec2 embedding extraction (768-dim)
  6B. Prosodic feature extraction (6-dim via Praat/parselmouth)
  7. Feature fusion (774-dim) + StandardScaler + GroupShuffleSplit
  8. Classification  (BiLSTM → 128-dim → SVM RBF)
  9. Evaluation (F1, accuracy, ROC AUC, EER, confusion matrix, 95 % CI)
 10. Output generation (predictions.csv, technical_report.md)

Inference-only mode:
  Uses saved models (lstm_encoder.pt, svm_model.joblib, scaler.joblib)
  to predict on new audio files without retraining.

Usage:
  # Full training pipeline (requires raw audio + internet for CV)
  python pipeline.py --mode train --project-root ./workspace

  # Inference only (uses saved models)
  python pipeline.py --mode predict --audio-csv data.csv --output-dir outputs
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import subprocess
import sys
import tempfile
import time
import warnings
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    StratifiedGroupKFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Model, Wav2Vec2Processor

warnings.filterwarnings("ignore")

# ─── Constants ───────────────────────────────────────────────────
TARGET_SR = 16000
TARGET_RMS = 0.05
MIN_DURATION = 3.0
VAD_TOP_DB = 20
NR_PROP = 0.75
WINDOW_SEC = 3.0
HOP_SEC = 1.0
BATCH_SIZE_EMBED = 16
CHECKPOINT_EVERY = 50
RANDOM_SEED = 42

# LSTM hyper-parameters (must match the saved model)
HIDDEN_DIM = 64
NUM_LAYERS = 1
DROPOUT = 0.5
LSTM_FEAT_DIM = HIDDEN_DIM * 2  # bidirectional → 128
INPUT_DIM = 1030  # 1024 wav2vec2 (large model) + 6 prosodic
WAV2VEC2_DIM = 1024  # wav2vec2-large-xlsr-53-arabic hidden_size

WAV2VEC2_MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


# ═══════════════════════════════════════════════════════════════════
# LSTM ENCODER  (must match notebook's architecture exactly)
# ═══════════════════════════════════════════════════════════════════
class LSTMEncoder(nn.Module):
    """Bidirectional LSTM that encodes a variable-length sequence of
    774-dim segment features into a fixed 128-dim recording-level
    representation."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.drop = nn.Dropout(dropout)
        # Classification head — used during LSTM pre-training only
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return the fixed-size hidden-state vector (hidden_dim*2)."""
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # Concat final forward + backward hidden states
        h_final = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, hidden*2)
        h_final = self.drop(h_final)
        return h_final

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        h = self.encode(x, lengths)
        logits = self.classifier(h).squeeze(-1)
        return logits, h


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, lengths):
        self.sequences = sequences
        self.labels = torch.FloatTensor(labels)
        self.lengths = lengths

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.lengths[idx]


def collate_fn(batch):
    seqs, labels, lengths = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True)
    return padded, torch.stack(labels), torch.LongTensor(lengths)


# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def load_audio(filepath: str | Path) -> tuple[np.ndarray, int]:
    """Load audio; falls back to ffmpeg conversion for non-WAV formats."""
    filepath = str(filepath)
    try:
        y, sr = librosa.load(filepath, sr=TARGET_SR, mono=True)
        return y, sr
    except Exception:
        wav_tmp = tempfile.mktemp(suffix=".wav")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", filepath, "-ar", str(TARGET_SR), "-ac", "1", wav_tmp],
                capture_output=True,
                check=True,
            )
            y, sr = librosa.load(wav_tmp, sr=TARGET_SR, mono=True)
            return y, sr
        finally:
            if os.path.exists(wav_tmp):
                os.unlink(wav_tmp)


def rms_normalize(y: np.ndarray, target_rms: float = TARGET_RMS) -> np.ndarray:
    current_rms = np.sqrt(np.mean(y**2))
    if current_rms < 1e-9:
        return y
    return y * (target_rms / current_rms)


def vad_trim_edges(y: np.ndarray, top_db: int = VAD_TOP_DB) -> np.ndarray:
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed


def reduce_noise(y: np.ndarray, sr: int, prop_decrease: float = NR_PROP) -> np.ndarray:
    import noisereduce as nr
    return nr.reduce_noise(y=y, sr=sr, prop_decrease=prop_decrease, stationary=True)


def preprocess_audio(y: np.ndarray, sr: int) -> np.ndarray | None:
    """Full preprocessing chain: RMS norm → VAD trim → denoise → duration gate."""
    y = rms_normalize(y)
    y = vad_trim_edges(y)
    y = reduce_noise(y, sr)
    duration = len(y) / sr
    if duration < MIN_DURATION:
        return None
    return y


def split_audio(y: np.ndarray, sr: int = TARGET_SR) -> list[np.ndarray]:
    """Split audio into overlapping 3 s segments with 1 s hop."""
    window_samples = int(WINDOW_SEC * sr)
    hop_samples = int(HOP_SEC * sr)
    segments = []
    n_segments = max(0, (len(y) - window_samples) // hop_samples + 1)
    for i in range(n_segments):
        start = i * hop_samples
        end = start + window_samples
        seg = y[start:end]
        if len(seg) < window_samples:
            seg = np.pad(seg, (0, window_samples - len(seg)))
        segments.append(seg)
    return segments


# ═══════════════════════════════════════════════════════════════════
# AUGMENTATION
# ═══════════════════════════════════════════════════════════════════

def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    return librosa.effects.time_stretch(y, rate=rate)


def pitch_shift(y: np.ndarray, sr: int, steps: float) -> np.ndarray:
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)


def add_noise(y: np.ndarray, sigma: float = 0.005) -> np.ndarray:
    noise = np.random.randn(len(y)) * sigma
    return np.clip(y + noise, -1.0, 1.0)


def augment_segment(y: np.ndarray, sr: int, is_minority: bool) -> list[tuple[str, np.ndarray]]:
    """Return augmented variants of a segment.
    Minority (Non-Native): 5 augmentations.
    Majority (Native): 1 augmentation (time stretch only).
    """
    window_samples = int(WINDOW_SEC * sr)
    results = []
    if is_minority:
        augmentations = [
            ("ts_slow", time_stretch(y, 0.9)),
            ("ts_fast", time_stretch(y, 1.1)),
            ("ps_up", pitch_shift(y, sr, +2)),
            ("ps_down", pitch_shift(y, sr, -2)),
            ("noise", add_noise(y)),
        ]
    else:
        augmentations = [("ts_slow", time_stretch(y, 0.9))]

    for name, y_aug in augmentations:
        if len(y_aug) > window_samples:
            y_aug = y_aug[:window_samples]
        elif len(y_aug) < window_samples:
            y_aug = np.pad(y_aug, (0, window_samples - len(y_aug)))
        results.append((name, y_aug))
    return results


# ═══════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def load_wav2vec2(device: str = "cpu"):
    """Load the Arabic wav2vec2 model and processor."""
    print(f"Loading wav2vec2 model ({WAV2VEC2_MODEL_ID})...")
    processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL_ID)
    model = Wav2Vec2Model.from_pretrained(WAV2VEC2_MODEL_ID)
    model = model.to(device)
    model.eval()
    print("Model loaded.")
    return processor, model


def extract_embedding_batch(
    waveforms: list[np.ndarray],
    processor: Wav2Vec2Processor,
    model: Wav2Vec2Model,
    device: str = "cpu",
    batch_size: int = 2,
) -> np.ndarray:
    """Extract wav2vec2 embeddings from a list of waveforms in mini-batches.
    Returns np.ndarray of shape (N, hidden_dim) where hidden_dim = 1024 for large model.
    """
    processed = []
    for y in waveforms:
        if len(y) > 48000:
            y = y[:48000]
        elif len(y) < 48000:
            y = np.pad(y, (0, 48000 - len(y)))
        processed.append(y)

    all_embeddings = []
    for i in range(0, len(processed), batch_size):
        batch = processed[i : i + batch_size]
        inputs = processor(
            batch, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            outputs = model(input_values)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def extract_prosodic_features(audio_path: str) -> np.ndarray:
    """Extract 6 prosodic features from a full utterance WAV file.
    Returns np.array of shape (6,):
    [f0_mean, f0_var, speaking_rate, pause_freq, pause_dur_mean, npvi]
    """
    try:
        import parselmouth
    except ImportError:
        print("WARNING: parselmouth not installed. Returning zero prosodic features.")
        return np.zeros(6, dtype=np.float32)

    try:
        sound = parselmouth.Sound(audio_path)
        duration = sound.end_time - sound.start_time

        # F0 mean and variance
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
        f0_vals = pitch.selected_array["frequency"]
        f0_voiced = f0_vals[f0_vals > 0]
        f0_mean = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        f0_var = float(np.var(f0_voiced)) if len(f0_voiced) > 0 else 0.0

        # Speaking rate (syllable-proxy peaks/sec)
        intensity = sound.to_intensity(minimum_pitch=75)
        int_values = intensity.values.flatten()
        int_mean = float(np.mean(int_values))
        peaks = 0
        for j in range(1, len(int_values) - 1):
            if (
                int_values[j] > int_mean
                and int_values[j] > int_values[j - 1]
                and int_values[j] > int_values[j + 1]
            ):
                peaks += 1
        speaking_rate = peaks / duration if duration > 0 else 0.0

        # Pause frequency and mean pause duration
        pause_threshold_sec = 0.2
        f0_times = [
            pitch.get_time_from_frame_number(i + 1)
            for i in range(pitch.get_number_of_frames())
        ]
        is_voiced = [
            pitch.get_value_at_time(t) is not None and pitch.get_value_at_time(t) > 0
            for t in f0_times
        ]
        pauses = []
        in_pause = False
        pause_start = 0.0
        for t, voiced in zip(f0_times, is_voiced):
            if not voiced and not in_pause:
                in_pause = True
                pause_start = t
            elif voiced and in_pause:
                pause_dur = t - pause_start
                if pause_dur >= pause_threshold_sec:
                    pauses.append(pause_dur)
                in_pause = False
        pause_freq = len(pauses) / duration if duration > 0 else 0.0
        pause_dur_mean = float(np.mean(pauses)) if pauses else 0.0

        # nPVI (Normalised Pairwise Variability Index)
        vowel_durations = []
        in_vowel = False
        vowel_start = 0.0
        for t, voiced in zip(f0_times, is_voiced):
            if voiced and not in_vowel:
                in_vowel = True
                vowel_start = t
            elif not voiced and in_vowel:
                vowel_dur = t - vowel_start
                if vowel_dur >= 0.03:
                    vowel_durations.append(vowel_dur)
                in_vowel = False

        if len(vowel_durations) >= 2:
            durs = np.array(vowel_durations)
            dk = durs[:-1]
            dk1 = durs[1:]
            denom = (dk + dk1) / 2
            valid = denom > 0
            npvi = float(np.mean(np.abs(dk[valid] - dk1[valid]) / denom[valid]) * 100)
        else:
            npvi = 0.0

        return np.array(
            [f0_mean, f0_var, speaking_rate, pause_freq, pause_dur_mean, npvi],
            dtype=np.float32,
        )

    except Exception as e:
        print(f"  Prosodic extraction failed for {audio_path}: {e}")
        return np.zeros(6, dtype=np.float32)


def extract_prosodic_from_waveform(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Extract prosodic features from a waveform array (writes temp WAV)."""
    tmp = tempfile.mktemp(suffix=".wav")
    try:
        sf.write(tmp, y, sr, subtype="PCM_16")
        return extract_prosodic_features(tmp)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ═══════════════════════════════════════════════════════════════════
# GROUPING SEGMENTS → RECORDING-LEVEL SEQUENCES
# ═══════════════════════════════════════════════════════════════════

def group_into_sequences(
    X: np.ndarray,
    parent_ids: np.ndarray,
    aug_types: np.ndarray,
    seg_indices: np.ndarray,
    y: np.ndarray | None = None,
):
    """Group segment features into recording-level sequences.
    Key = (parent_id, aug_type). Returns tensors for LSTM input.
    """
    seq_keys = np.array([f"{pid}__{aug}" for pid, aug in zip(parent_ids, aug_types)])
    unique_keys = np.unique(seq_keys)

    sequences = []
    seq_labels = []
    seq_lengths = []
    seq_key_list = []

    for key in unique_keys:
        mask = seq_keys == key
        indices = seg_indices[mask]
        sort_order = np.argsort(indices)

        seq = X[mask][sort_order]  # (n_segs, 774)
        sequences.append(torch.FloatTensor(seq))
        seq_lengths.append(len(seq))
        seq_key_list.append(key)

        if y is not None:
            label = int(y[mask][0])
            seq_labels.append(label)

    if y is not None:
        return sequences, np.array(seq_labels), seq_lengths, seq_key_list
    return sequences, seq_lengths, seq_key_list


# ═══════════════════════════════════════════════════════════════════
# LSTM FEATURE EXTRACTION (inference)
# ═══════════════════════════════════════════════════════════════════

def extract_lstm_features(
    mdl: LSTMEncoder,
    sequences: list[torch.Tensor],
    lengths: list[int],
    device: str = "cpu",
) -> np.ndarray:
    """Pass recording-level sequences through the LSTM encoder to get
    128-dim feature vectors."""
    mdl.eval()
    features = []
    with torch.no_grad():
        for seq, length in zip(sequences, lengths):
            x = seq.unsqueeze(0).to(device)
            l = torch.LongTensor([length]).to(device)
            h = mdl.encode(x, l)
            features.append(h.cpu().numpy().squeeze(0))
    return np.array(features)


# ═══════════════════════════════════════════════════════════════════
# DATA COLLECTION (Stage 2)
# ═══════════════════════════════════════════════════════════════════

def download_file(url: str, save_path: str, timeout: int = 30) -> dict:
    """Download a single audio file from a URL."""
    import requests

    if os.path.exists(save_path):
        return {"status": "already_exists", "error": None}
    try:
        r = requests.get(url, timeout=timeout, headers=HEADERS)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "text/html" in ctype:
            return {"status": "bad_content_type", "error": "Got HTML page"}
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(r.content)
        return {"status": "downloaded", "error": None}
    except Exception as e:
        return {"status": "error", "error": str(e)[:120]}


def download_common_voice(
    api_key: str,
    save_dir: str,
    n_clips: int = 200,
    min_upvotes: int = 2,
) -> list[dict]:
    """Download native Arabic clips from Mozilla Common Voice API (streaming tar)."""
    import requests
    import tarfile

    CV_DATASET_ID = "cmj8u3os6000tnxxb169x1zdc"
    CV_API_URL = f"https://datacollective.mozillafoundation.org/api/datasets/{CV_DATASET_ID}/download"

    records = []
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print("  CV_API_KEY not set — skipping Common Voice download.")
        return records

    print("Requesting presigned download URL from Mozilla API...")
    try:
        resp = requests.post(
            CV_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        download_url = resp.json()["downloadUrl"]
        print("Got presigned URL — streaming tar archive...")
    except Exception as e:
        print(f"  Common Voice API error: {e}")
        return records

    os.makedirs(save_dir, exist_ok=True)

    try:
        with requests.get(download_url, stream=True, timeout=600) as r:
            r.raise_for_status()
            buf = io.BytesIO()
            for chunk in r.iter_content(chunk_size=8192):
                buf.write(chunk)
            buf.seek(0)

            with tarfile.open(fileobj=buf, mode="r:*") as tar:
                clip_count = 0
                for member in tar:
                    if not member.name.endswith(".mp3"):
                        continue
                    f = tar.extractfile(member)
                    if f is None:
                        continue

                    clip_id = f"cv_{clip_count:06d}"
                    save_path = os.path.join(save_dir, f"{clip_id}.mp3")
                    with open(save_path, "wb") as out:
                        out.write(f.read())

                    records.append({
                        "dp_id": clip_id,
                        "save_path": save_path,
                        "source": "common_voice",
                        "label": 1,
                        "nativity": "Native",
                        "dialect": "Arabic_MSA",
                        "status": "extracted",
                    })
                    clip_count += 1
                    if clip_count >= n_clips:
                        break

        print(f"  Extracted {len(records)} Common Voice clips.")
    except Exception as e:
        print(f"  Common Voice download error: {e}")

    return records


# ═══════════════════════════════════════════════════════════════════
# FULL TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════

def run_training_pipeline(
    project_root: str,
    renan_csv: str | None = None,
    cv_api_key: str = "",
    device: str | None = None,
):
    """Run the complete training pipeline (Stages 1–10)."""
    import requests
    from tqdm import tqdm

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    np.random.seed(RANDOM_SEED)

    project_root = os.path.abspath(project_root)
    data_dir = os.path.join(project_root, "data")
    output_dir = os.path.join(project_root, "outputs")
    embed_dir = os.path.join(data_dir, "embeddings")

    for d in [
        output_dir,
        os.path.join(data_dir, "train", "native"),
        os.path.join(data_dir, "train", "non_native"),
        os.path.join(data_dir, "preprocessed", "native"),
        os.path.join(data_dir, "preprocessed", "non_native"),
        embed_dir,
        os.path.join(data_dir, "segments", "native"),
        os.path.join(data_dir, "segments", "non_native"),
        os.path.join(data_dir, "augmented", "native"),
        os.path.join(data_dir, "augmented", "non_native"),
    ]:
        os.makedirs(d, exist_ok=True)

    # ── Stage 1: Data exploration ─────────────────────────────
    if renan_csv is None:
        renan_csv = os.path.join(project_root, "renan_dataset.csv")
    print("\n" + "=" * 58)
    print("  STAGE 1 — DATA EXPLORATION")
    print("=" * 58)
    df_renan = pd.read_csv(renan_csv)
    df_renan["label"] = df_renan["nativity_status"].map({"Native": 1, "Non-Native": 0})
    print(f"Dataset: {len(df_renan)} rows")
    print(f"Class balance:\n{df_renan['nativity_status'].value_counts().to_string()}")
    print(f"Dialects:\n{df_renan['language'].value_counts().to_string()}")

    # ── Stage 2: Data collection ──────────────────────────────
    print("\n" + "=" * 58)
    print("  STAGE 2 — DATA COLLECTION")
    print("=" * 58)

    df_renan["filename"] = df_renan["dp_id"].apply(lambda x: f"dp_{int(x)}.mp3")
    df_renan["save_dir"] = df_renan["label"].map(
        {1: os.path.join(data_dir, "train", "native"),
         0: os.path.join(data_dir, "train", "non_native")}
    )
    df_renan["save_path"] = df_renan.apply(
        lambda r: os.path.join(r["save_dir"], r["filename"]), axis=1
    )

    manifest_records = []
    for _, row in tqdm(df_renan.iterrows(), total=len(df_renan), desc="Renan files"):
        result = download_file(row["audio_url"], row["save_path"])
        result.update({
            "dp_id": row["dp_id"],
            "source": "renan",
            "label": row["label"],
            "nativity": row["nativity_status"],
            "dialect": row["language"],
            "save_path": row["save_path"],
        })
        manifest_records.append(result)
        time.sleep(0.05)

    # Part B: Common Voice
    cv_records = download_common_voice(
        api_key=cv_api_key,
        save_dir=os.path.join(data_dir, "train", "native"),
        n_clips=200,
    )
    manifest_records.extend(cv_records)

    mf_df = pd.DataFrame(manifest_records)
    mf_path = os.path.join(output_dir, "stage2_manifest.csv")
    mf_df.to_csv(mf_path, index=False)
    ok = mf_df["status"].isin(["downloaded", "already_exists", "extracted"]).sum()
    print(f"Downloaded/available: {ok} / {len(mf_df)}")

    # ── Stage 3: Preprocessing ────────────────────────────────
    print("\n" + "=" * 58)
    print("  STAGE 3 — PREPROCESSING")
    print("=" * 58)

    processable = mf_df[
        mf_df["status"].isin(["downloaded", "already_exists", "extracted"])
        & mf_df["save_path"].notna()
    ].copy()

    preproc_records = []
    for _, row in tqdm(processable.iterrows(), total=len(processable), desc="Preprocessing"):
        dp_id = row["dp_id"]
        subdir = "native" if row["label"] == 1 else "non_native"
        out_path = os.path.join(data_dir, "preprocessed", subdir, f"{dp_id}.wav")

        if os.path.exists(out_path):
            preproc_records.append({
                "dp_id": dp_id, "preproc_status": "ok",
                "preproc_path": out_path, "label": row["label"],
                "nativity": row["nativity"], "dialect": row["dialect"],
                "source": row["source"],
            })
            continue

        try:
            y, sr = load_audio(row["save_path"])
            y_proc = preprocess_audio(y, sr)
            if y_proc is None:
                preproc_records.append({
                    "dp_id": dp_id, "preproc_status": "dropped_too_short",
                    "preproc_path": None, "label": row["label"],
                    "nativity": row["nativity"], "dialect": row["dialect"],
                    "source": row["source"],
                })
                continue
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            sf.write(out_path, y_proc, TARGET_SR, subtype="PCM_16")
            preproc_records.append({
                "dp_id": dp_id, "preproc_status": "ok",
                "preproc_path": out_path, "label": row["label"],
                "nativity": row["nativity"], "dialect": row["dialect"],
                "source": row["source"],
            })
        except Exception as e:
            preproc_records.append({
                "dp_id": dp_id, "preproc_status": "error",
                "preproc_path": None, "label": row["label"],
                "nativity": row["nativity"], "dialect": row["dialect"],
                "source": row["source"],
            })

    preproc_df = pd.DataFrame(preproc_records)
    preproc_df.to_csv(os.path.join(output_dir, "stage3_manifest.csv"), index=False)
    usable = preproc_df[preproc_df["preproc_status"] == "ok"]
    print(f"Usable after preprocessing: {len(usable)}")

    # ── Stage 4: Audio splitting ──────────────────────────────
    print("\n" + "=" * 58)
    print("  STAGE 4 — AUDIO SPLITTING")
    print("=" * 58)

    segment_records = []
    for _, row in tqdm(usable.iterrows(), total=len(usable), desc="Splitting"):
        try:
            y, sr = librosa.load(row["preproc_path"], sr=TARGET_SR, mono=True)
        except Exception as e:
            print(f"  Load error {row['dp_id']}: {e}")
            continue

        segments = split_audio(y, sr)
        subdir = "native" if row["label"] == 1 else "non_native"
        for i, seg in enumerate(segments):
            seg_id = f"{row['dp_id']}_seg{i:04d}"
            seg_path = os.path.join(data_dir, "segments", subdir, f"{seg_id}.wav")
            sf.write(seg_path, seg, TARGET_SR, subtype="PCM_16")
            segment_records.append({
                "seg_id": seg_id, "parent_id": row["dp_id"],
                "seg_index": i, "label": row["label"],
                "nativity": row["nativity"], "dialect": row["dialect"],
                "source": row["source"], "seg_path": seg_path,
            })

    seg_df = pd.DataFrame(segment_records)
    seg_df.to_csv(os.path.join(output_dir, "stage4_manifest.csv"), index=False)
    print(f"Total segments: {len(seg_df)}")

    # ── Stage 5: Data augmentation ────────────────────────────
    print("\n" + "=" * 58)
    print("  STAGE 5 — DATA AUGMENTATION")
    print("=" * 58)

    aug_records = []
    for _, row in tqdm(seg_df.iterrows(), total=len(seg_df), desc="Augmenting"):
        try:
            y, sr = librosa.load(row["seg_path"], sr=TARGET_SR, mono=True)
        except Exception:
            continue

        is_minority = row["label"] == 0
        augmented = augment_segment(y, sr, is_minority)
        subdir = "native" if row["label"] == 1 else "non_native"

        for aug_name, y_aug in augmented:
            aug_id = f"{row['seg_id']}_{aug_name}"
            aug_path = os.path.join(data_dir, "augmented", subdir, f"{aug_id}.wav")
            sf.write(aug_path, y_aug, TARGET_SR, subtype="PCM_16")
            aug_records.append({
                "seg_id": aug_id, "parent_id": row["parent_id"],
                "seg_index": row["seg_index"], "label": row["label"],
                "nativity": row["nativity"], "dialect": row["dialect"],
                "source": "augmented", "seg_path": aug_path,
                "aug_type": aug_name, "orig_seg": row["seg_id"],
            })

    originals = seg_df.assign(aug_type="original", orig_seg=None)
    aug_df = pd.DataFrame(aug_records)
    all_segs = pd.concat([originals, aug_df], ignore_index=True)
    all_segs.to_csv(os.path.join(output_dir, "stage5_manifest.csv"), index=False)
    print(f"Original: {len(seg_df)}, Augmented: {len(aug_df)}, Total: {len(all_segs)}")

    # ── Stage 6A: wav2vec2 embedding extraction ───────────────
    print("\n" + "=" * 58)
    print("  STAGE 6A — WAV2VEC2 EMBEDDINGS")
    print("=" * 58)

    processor, w2v_model = load_wav2vec2(device)

    all_embeddings = []
    all_seg_ids = []
    paths = all_segs["seg_path"].tolist()
    seg_ids = all_segs["seg_id"].tolist()

    for i in tqdm(range(0, len(paths), BATCH_SIZE_EMBED), desc="Extracting embeddings"):
        batch_paths = paths[i : i + BATCH_SIZE_EMBED]
        batch_seg_ids = seg_ids[i : i + BATCH_SIZE_EMBED]
        try:
            waveforms = []
            for p in batch_paths:
                yy, _ = librosa.load(p, sr=TARGET_SR, mono=True)
                waveforms.append(yy)
            embeds = extract_embedding_batch(waveforms, processor, w2v_model, device)
            all_embeddings.append(embeds)
            all_seg_ids.extend(batch_seg_ids)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"OOM at batch {i}")
                torch.cuda.empty_cache()
                raise
            all_embeddings.append(np.zeros((len(batch_paths), WAV2VEC2_DIM)))
            all_seg_ids.extend(batch_seg_ids)

    X_wav2vec = np.vstack(all_embeddings)
    np.save(os.path.join(embed_dir, "wav2vec2_embeddings.npy"), X_wav2vec)
    np.save(os.path.join(embed_dir, "wav2vec2_seg_ids.npy"), np.array(all_seg_ids))
    print(f"Embeddings shape: {X_wav2vec.shape}")

    # Free GPU memory
    del w2v_model, processor
    torch.cuda.empty_cache() if device == "cuda" else None

    # ── Stage 6B: Prosodic feature extraction ─────────────────
    print("\n" + "=" * 58)
    print("  STAGE 6B — PROSODIC FEATURES")
    print("=" * 58)

    utt_df = preproc_df[preproc_df["preproc_status"] == "ok"].copy()
    prosodic_records = []
    for _, row in tqdm(utt_df.iterrows(), total=len(utt_df), desc="Prosodic features"):
        features = extract_prosodic_features(row["preproc_path"])
        prosodic_records.append({
            "dp_id": row["dp_id"], "label": row["label"],
            "f0_mean": features[0], "f0_var": features[1],
            "speaking_rate": features[2], "pause_freq": features[3],
            "pause_dur_mean": features[4], "npvi": features[5],
        })

    prosodic_df = pd.DataFrame(prosodic_records)
    prosodic_df.to_csv(os.path.join(embed_dir, "prosodic_features.csv"), index=False)
    print(f"Prosodic features shape: {prosodic_df.shape}")

    # ── Stage 7: Feature fusion ───────────────────────────────
    print("\n" + "=" * 58)
    print("  STAGE 7 — FEATURE FUSION")
    print("=" * 58)

    prosodic_map = prosodic_df.set_index("dp_id")[
        ["f0_mean", "f0_var", "speaking_rate", "pause_freq", "pause_dur_mean", "npvi"]
    ].to_dict("index")

    seg_id_to_idx = {sid: i for i, sid in enumerate(all_seg_ids)}

    fused_features = []
    labels = []
    parent_ids_arr = []
    seg_ids_final = []
    aug_types_list = []
    seg_indices_list = []

    for _, row in all_segs.iterrows():
        seg_id = row["seg_id"]
        if seg_id not in seg_id_to_idx:
            continue

        w2v_embed = X_wav2vec[seg_id_to_idx[seg_id]]
        parent_id = row["parent_id"]
        base_parent = parent_id.split("_aug")[0] if "_aug" in str(parent_id) else str(parent_id)

        if base_parent in prosodic_map:
            pros_vec = np.array(list(prosodic_map[base_parent].values()), dtype=np.float32)
        else:
            pros_vec = np.zeros(6, dtype=np.float32)

        fused_features.append(np.concatenate([w2v_embed, pros_vec]))
        labels.append(row["label"])
        parent_ids_arr.append(str(parent_id))
        seg_ids_final.append(seg_id)
        aug_types_list.append(row.get("aug_type", "original"))
        seg_indices_list.append(int(row.get("seg_index", 0)))

    X = np.array(fused_features)
    y = np.array(labels)
    groups = np.array(parent_ids_arr)
    aug_types_arr = np.array(aug_types_list)
    seg_indices_arr = np.array(seg_indices_list, dtype=np.int32)

    print(f"Fused feature matrix: {X.shape}")
    print(f"Labels — Native: {(y == 1).sum()}, Non-Native: {(y == 0).sum()}")

    # Group-level Train/Test split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    np.savez(
        os.path.join(output_dir, "stage7_features.npz"),
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        train_idx=train_idx, test_idx=test_idx,
        train_parent_ids=groups[train_idx],
        test_parent_ids=groups[test_idx],
        train_aug_types=aug_types_arr[train_idx],
        test_aug_types=aug_types_arr[test_idx],
        train_seg_indices=seg_indices_arr[train_idx],
        test_seg_indices=seg_indices_arr[test_idx],
    )
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    print(f"Train: {X_train.shape[0]} segments, Test: {X_test.shape[0]} segments")

    # ── Stage 8: Classification ───────────────────────────────
    print("\n" + "=" * 58)
    print("  STAGE 8 — CLASSIFICATION (LSTM + SVM RBF)")
    print("=" * 58)

    train_parent_ids = groups[train_idx]
    test_parent_ids = groups[test_idx]
    train_aug_types = aug_types_arr[train_idx]
    test_aug_types = aug_types_arr[test_idx]
    train_seg_indices = seg_indices_arr[train_idx]
    test_seg_indices = seg_indices_arr[test_idx]

    train_seqs, y_train_seq, train_lens, train_keys = group_into_sequences(
        X_train, train_parent_ids, train_aug_types, train_seg_indices, y_train
    )
    test_seqs, y_test_seq, test_lens, test_keys = group_into_sequences(
        X_test, test_parent_ids, test_aug_types, test_seg_indices, y_test
    )

    print(f"Recording-level: Train={len(train_seqs)}, Test={len(test_seqs)}")

    # Train/val split for LSTM
    train_parent_ids_seq = np.array([k.split("__")[0] for k in train_keys])
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr_idx, val_idx = next(gss_val.split(
        np.arange(len(train_seqs)), y_train_seq, groups=train_parent_ids_seq
    ))

    val_seqs = [train_seqs[i] for i in val_idx]
    val_labels = y_train_seq[val_idx]
    val_lens = [train_lens[i] for i in val_idx]
    actual_train_seqs = [train_seqs[i] for i in tr_idx]
    actual_train_labels = y_train_seq[tr_idx]
    actual_train_lens = [train_lens[i] for i in tr_idx]

    # Build LSTM encoder
    lstm_model = LSTMEncoder(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    n_pos = int((actual_train_labels == 1).sum())
    n_neg = int((actual_train_labels == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_dataset = SequenceDataset(actual_train_seqs, actual_train_labels, actual_train_lens)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_dataset = SequenceDataset(val_seqs, val_labels, val_lens)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Train LSTM
    lstm_path = os.path.join(output_dir, "lstm_encoder.pt")
    EPOCHS = 100
    PATIENCE = 15
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Training LSTM ({EPOCHS} epochs max, patience={PATIENCE})...")
    for epoch in range(EPOCHS):
        lstm_model.train()
        total_train_loss = 0
        n_batches = 0
        for batch_x, batch_y, batch_lens in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits, _ = lstm_model(batch_x, batch_lens)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_train_loss / n_batches

        lstm_model.eval()
        total_val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_lens in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits, _ = lstm_model(batch_x, batch_lens)
                loss = criterion(logits, batch_y)
                total_val_loss += loss.item()
                n_val += 1

        avg_val_loss = total_val_loss / max(n_val, 1)
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3}: train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(lstm_model.state_dict(), lstm_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    print(f"Best val loss: {best_val_loss:.4f}")

    # Extract LSTM features
    X_train_lstm = extract_lstm_features(lstm_model, train_seqs, train_lens, device)
    X_test_lstm = extract_lstm_features(lstm_model, test_seqs, test_lens, device)
    print(f"LSTM features: Train={X_train_lstm.shape}, Test={X_test_lstm.shape}")

    # SVM RBF with GridSearchCV
    print("\nSVM RBF GridSearchCV...")
    param_grid = {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.1, 0.01, 0.001]}
    svm_rbf = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)
    train_groups_seq = np.array([k.split("__")[0] for k in train_keys])
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    search = GridSearchCV(svm_rbf, param_grid=param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1, refit=True)
    search.fit(X_train_lstm, y_train_seq, groups=train_groups_seq)

    best_svm = search.best_estimator_
    print(f"Best C={search.best_params_['C']}, gamma={search.best_params_['gamma']}")
    print(f"Best CV F1: {search.best_score_:.4f}")

    y_pred_test = best_svm.predict(X_test_lstm)
    test_f1 = f1_score(y_test_seq, y_pred_test)
    print(f"Test F1: {test_f1:.4f}")
    print(classification_report(y_test_seq, y_pred_test, target_names=["Non-Native", "Native"]))

    # Save models
    lstm_config = {"input_dim": INPUT_DIM, "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "dropout": DROPOUT}
    joblib.dump(lstm_config, os.path.join(output_dir, "lstm_config.joblib"))
    joblib.dump(best_svm, os.path.join(output_dir, "svm_model.joblib"))
    joblib.dump(search, os.path.join(output_dir, "grid_search_results.joblib"))

    np.savez(
        os.path.join(output_dir, "stage8_test_data.npz"),
        X_test_lstm=X_test_lstm, y_test_seq=y_test_seq,
        test_keys=np.array(test_keys),
    )

    # ── Stage 9: Evaluation ───────────────────────────────────
    print("\n" + "=" * 58)
    print("  STAGE 9 — EVALUATION")
    print("=" * 58)

    y_scores = best_svm.predict_proba(X_test_lstm)[:, 1]
    acc = accuracy_score(y_test_seq, y_pred_test)
    f1 = f1_score(y_test_seq, y_pred_test)
    precision = precision_score(y_test_seq, y_pred_test, zero_division=0)
    recall = recall_score(y_test_seq, y_pred_test, zero_division=0)
    auc = roc_auc_score(y_test_seq, y_scores)
    cm = confusion_matrix(y_test_seq, y_pred_test)

    fpr, tpr, thresholds = roc_curve(y_test_seq, y_scores)
    frr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - frr))
    eer = (fpr[eer_idx] + frr[eer_idx]) / 2
    eer_thr = thresholds[eer_idx]

    n = len(y_test_seq)
    ci_low, ci_high = stats.binom.interval(0.95, n, acc)
    ci_low /= n
    ci_high /= n

    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  ROC AUC   : {auc:.4f}")
    print(f"  EER       : {eer*100:.2f}%")
    print(f"  95% CI    : [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")

    metrics = {
        "accuracy": acc, "f1_score": f1, "precision": precision,
        "recall": recall, "roc_auc": auc, "eer": eer,
        "eer_threshold": eer_thr, "ci_low": ci_low, "ci_high": ci_high,
        "n_test": n, "tn": cm[0, 0], "fp": cm[0, 1], "fn": cm[1, 0], "tp": cm[1, 1],
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, "stage9_metrics.csv"), index=False)

    # ── Stage 10: Output generation ───────────────────────────
    print("\n" + "=" * 58)
    print("  STAGE 10 — OUTPUT GENERATION")
    print("=" * 58)

    probs = best_svm.predict_proba(X_test_lstm)
    confidence = probs.max(axis=1)
    prob_native = probs[:, 1]
    prob_non_native = probs[:, 0]

    recording_ids = [k.split("__")[0] for k in test_keys]
    aug_types_out = [k.split("__")[1] if "__" in k else "original" for k in test_keys]

    results = pd.DataFrame({
        "recording_id": recording_ids,
        "aug_type": aug_types_out,
        "predicted_label": y_pred_test,
        "predicted_class": ["Native" if p == 1 else "Non-Native" for p in y_pred_test],
        "confidence_score": np.round(confidence, 4),
        "prob_native": np.round(prob_native, 4),
        "prob_non_native": np.round(prob_non_native, 4),
        "true_label": y_test_seq,
        "true_class": ["Native" if t == 1 else "Non-Native" for t in y_test_seq],
        "correct": (y_pred_test == y_test_seq),
    })
    results.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    print(f"predictions.csv: {len(results)} rows, {results['correct'].sum()}/{len(results)} correct")

    print("\n" + "=" * 58)
    print("  TRAINING PIPELINE COMPLETE")
    print("=" * 58)
    print(f"Models saved in: {output_dir}")
    return metrics


# ═══════════════════════════════════════════════════════════════════
# INFERENCE-ONLY PIPELINE
# ═══════════════════════════════════════════════════════════════════

def load_saved_models(models_dir: str, device: str = "cpu"):
    """Load pre-trained LSTM encoder, SVM, and scaler from disk.

    The LSTM weights file may be either:
      - lstm_encoder.pt      (plain state_dict)
      - lstm_encoder.pt.zip  (zip archive containing the .pt file)
    """
    # --- LSTM config ---
    config_path = os.path.join(models_dir, "lstm_config.joblib")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"lstm_config.joblib not found in {models_dir}")
    lstm_config = joblib.load(config_path)

    # --- LSTM weights ---
    # The file may be:
    #   - lstm_encoder.pt       (plain state_dict)
    #   - lstm_encoder.pt.zip   (torch.save zip format — torch.load reads it directly)
    pt_path = os.path.join(models_dir, "lstm_encoder.pt")
    zip_path = os.path.join(models_dir, "lstm_encoder.pt.zip")

    if os.path.exists(pt_path):
        weights_path = pt_path
    elif os.path.exists(zip_path):
        weights_path = zip_path
    else:
        raise FileNotFoundError(f"lstm_encoder.pt(.zip) not found in {models_dir}")

    lstm_model = LSTMEncoder(
        input_dim=lstm_config.get("input_dim", INPUT_DIM),
        hidden_dim=lstm_config.get("hidden_dim", HIDDEN_DIM),
        num_layers=lstm_config.get("num_layers", NUM_LAYERS),
        dropout=lstm_config.get("dropout", DROPOUT),
    ).to(device)
    lstm_model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    lstm_model.eval()

    # --- SVM ---
    svm_path = os.path.join(models_dir, "svm_model.joblib")
    if not os.path.exists(svm_path):
        raise FileNotFoundError(f"svm_model.joblib not found in {models_dir}")
    svm_model = joblib.load(svm_path)

    # --- Scaler ---
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"scaler.joblib not found in {models_dir}")
    scaler = joblib.load(scaler_path)

    return lstm_model, svm_model, scaler, lstm_config


def predict_from_urls(
    audio_csv: str,
    models_dir: str,
    output_dir: str,
    device: str | None = None,
):
    """Run inference on a CSV of audio URLs using saved models.

    The CSV must have columns: dp_id, audio_url
    Optionally: nativity_status, language

    Returns a DataFrame of predictions saved to output_dir/predictions.csv
    """
    import requests
    from tqdm import tqdm

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = os.path.join(output_dir, "_tmp_audio")
    os.makedirs(tmp_dir, exist_ok=True)

    # Load models
    print("Loading saved models...")
    lstm_model, svm_model, scaler, lstm_config = load_saved_models(models_dir, device)

    # Load wav2vec2
    processor, w2v_model = load_wav2vec2(device)

    # Read input CSV
    df = pd.read_csv(audio_csv)
    required_cols = {"dp_id", "audio_url"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"CSV must have columns: {required_cols}. Found: {set(df.columns)}")

    print(f"Input: {len(df)} audio files to classify")

    # Download and process each file
    all_results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio"):
        dp_id = row["dp_id"]
        audio_url = row["audio_url"]
        language = row.get("language", "unknown")

        # Download
        ext = os.path.splitext(audio_url.split("?")[0])[-1].lower()
        if ext not in [".wav", ".mp3", ".aac", ".ogg", ".flac", ".m4a"]:
            ext = ".mp3"
        local_path = os.path.join(tmp_dir, f"{dp_id}{ext}")

        if not os.path.exists(local_path):
            try:
                resp = requests.get(audio_url, timeout=30, headers=HEADERS)
                resp.raise_for_status()
                with open(local_path, "wb") as f:
                    f.write(resp.content)
            except Exception as e:
                print(f"  Download failed for {dp_id}: {e}")
                all_results.append({
                    "dp_id": dp_id, "language": language,
                    "predicted_class": "ERROR", "confidence_score": 0,
                    "prob_native": 0, "prob_non_native": 0,
                    "error": str(e)[:100],
                })
                continue

        # Load and preprocess
        try:
            y, sr = load_audio(local_path)
        except Exception as e:
            print(f"  Load failed for {dp_id}: {e}")
            all_results.append({
                "dp_id": dp_id, "language": language,
                "predicted_class": "ERROR", "confidence_score": 0,
                "prob_native": 0, "prob_non_native": 0,
                "error": str(e)[:100],
            })
            continue

        y_proc = preprocess_audio(y, sr)
        if y_proc is None:
            # Too short — try using raw audio
            y_proc = y
            if len(y_proc) / sr < 1.0:
                all_results.append({
                    "dp_id": dp_id, "language": language,
                    "predicted_class": "TOO_SHORT", "confidence_score": 0,
                    "prob_native": 0, "prob_non_native": 0,
                    "error": f"Audio too short ({len(y)/sr:.1f}s)",
                })
                continue

        # Extract prosodic features (from full utterance)
        pros_vec = extract_prosodic_from_waveform(y_proc, sr)

        # Split into segments
        segments = split_audio(y_proc, sr)
        if len(segments) == 0:
            # File shorter than 3s — use entire audio as one segment
            if len(y_proc) > 48000:
                y_proc = y_proc[:48000]
            elif len(y_proc) < 48000:
                y_proc = np.pad(y_proc, (0, 48000 - len(y_proc)))
            segments = [y_proc]

        # Extract wav2vec2 embeddings for all segments
        seg_embeddings = extract_embedding_batch(segments, processor, w2v_model, device)

        # Fuse: concatenate wav2vec2 (768) + prosodic (6) → 774
        fused = np.array([
            np.concatenate([emb, pros_vec]) for emb in seg_embeddings
        ])

        # Scale features
        fused_scaled = scaler.transform(fused)

        # Create a single sequence from all segments (sorted by index)
        seq_tensor = torch.FloatTensor(fused_scaled)
        seq_length = len(fused_scaled)

        # LSTM encode → 128-dim
        with torch.no_grad():
            x = seq_tensor.unsqueeze(0).to(device)
            l = torch.LongTensor([seq_length]).to(device)
            h = lstm_model.encode(x, l)
            lstm_feat = h.cpu().numpy()

        # SVM predict
        pred_label = svm_model.predict(lstm_feat)[0]
        probs = svm_model.predict_proba(lstm_feat)[0]
        confidence = probs.max()
        prob_native = probs[1]
        prob_non_native = probs[0]

        all_results.append({
            "dp_id": dp_id,
            "language": language,
            "predicted_label": int(pred_label),
            "predicted_class": "Native" if pred_label == 1 else "Non-Native",
            "confidence_score": round(float(confidence), 4),
            "prob_native": round(float(prob_native), 4),
            "prob_non_native": round(float(prob_non_native), 4),
            "n_segments": len(segments),
            "error": None,
        })

    results_df = pd.DataFrame(all_results)
    out_path = os.path.join(output_dir, "predictions.csv")
    results_df.to_csv(out_path, index=False)

    # Summary
    valid = results_df[results_df["predicted_class"].isin(["Native", "Non-Native"])]
    print(f"\nPredictions saved: {out_path}")
    print(f"  Total files     : {len(results_df)}")
    print(f"  Successful      : {len(valid)}")
    print(f"  Predicted Native: {(valid['predicted_class'] == 'Native').sum()}")
    print(f"  Predicted Non-N : {(valid['predicted_class'] == 'Non-Native').sum()}")

    # Clean up temp audio
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return results_df


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Native / Non-Native Arabic Speaker Classifier"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ── train ──
    train_p = sub.add_parser("train", help="Run full training pipeline")
    train_p.add_argument("--project-root", required=True, help="Project workspace root")
    train_p.add_argument("--renan-csv", default=None, help="Path to renan_dataset.csv")
    train_p.add_argument("--cv-api-key", default="", help="Mozilla Common Voice API key")
    train_p.add_argument("--device", default=None, choices=["cpu", "cuda"])

    # ── predict ──
    pred_p = sub.add_parser("predict", help="Run inference using saved models")
    pred_p.add_argument("--audio-csv", required=True, help="CSV with dp_id,audio_url columns")
    pred_p.add_argument("--models-dir", required=True, help="Directory with saved models")
    pred_p.add_argument("--output-dir", default="outputs", help="Output directory")
    pred_p.add_argument("--device", default=None, choices=["cpu", "cuda"])

    args = parser.parse_args()

    if args.mode == "train":
        run_training_pipeline(
            project_root=args.project_root,
            renan_csv=args.renan_csv,
            cv_api_key=args.cv_api_key,
            device=args.device,
        )
    elif args.mode == "predict":
        predict_from_urls(
            audio_csv=args.audio_csv,
            models_dir=args.models_dir,
            output_dir=args.output_dir,
            device=args.device,
        )


if __name__ == "__main__":
    main()
