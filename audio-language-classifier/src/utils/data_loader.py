"""
Data loading utilities.

Supports three modes:
  1. LOCAL  — load audio files organized as  data/raw/<lang_code>/*.wav
  2. MDC    — download from Mozilla Data Collective (Common Voice)
  3. STREAM — (legacy) stream from HuggingFace datasets
"""

import os
import glob
import tarfile
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from src.config.settings import (
    SAMPLE_RATE, TARGET_SAMPLES, SAMPLES_PER_LANGUAGE,
    LANGUAGES, CODE_TO_LABEL, PROCESSED_DATA_DIR, RAW_DATA_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, MANIFEST_PATH,
    MDC_DATASET_IDS,
)
from src.preprocessing import preprocess_audio


# ── Local file loading ────────────────────────────────────────────

def load_local_data(raw_dir: str) -> pd.DataFrame:
    """Load audio from local directory structure.

    Expected layout:
        raw_dir/
            ar/  *.wav
            en/  *.wav
            ...

    Returns a manifest DataFrame with columns:
        sample_id, language_code, language, npy_path, label
    """
    records = []

    for lang_code, lang_name in LANGUAGES:
        lang_dir = os.path.join(raw_dir, lang_code)
        if not os.path.isdir(lang_dir):
            print(f"  ⚠ Skipping {lang_name} — directory not found: {lang_dir}")
            continue

        files = sorted([
            f for f in os.listdir(lang_dir)
            if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))
        ])
        print(f"  {lang_name} ({lang_code}): {len(files)} files found")

        for i, fname in enumerate(tqdm(files, desc=lang_name)):
            fpath = os.path.join(lang_dir, fname)
            y = preprocess_audio(fpath)
            if y is None:
                continue

            sample_id = f"{lang_code}_{i:05d}"
            npy_path = os.path.join(PROCESSED_DATA_DIR, f"{sample_id}.npy")
            np.save(npy_path, y)

            records.append({
                "sample_id": sample_id,
                "language_code": lang_code,
                "language": lang_name,
                "npy_path": npy_path,
                "label": CODE_TO_LABEL[lang_code],
            })

    df = pd.DataFrame(records)
    return df


# ── Mozilla Data Collective (MDC) download ────────────────────────

def load_mdc_data(samples_per_lang: int = SAMPLES_PER_LANGUAGE) -> pd.DataFrame:
    """Download Common Voice data from Mozilla Data Collective.

    Requires:
        pip install datacollective
        export MDC_API_KEY=your-api-key-here

    Get your API key at: https://datacollective.mozillafoundation.org/profile/credentials
    Accept dataset terms on the website before downloading.

    Returns a manifest DataFrame.
    """
    from datacollective import save_dataset_to_disk

    records = []

    for lang_code, lang_name in LANGUAGES:
        dataset_id = MDC_DATASET_IDS.get(lang_code, "")
        if not dataset_id:
            print(f"  ⚠ Skipping {lang_name} — no MDC dataset ID configured.")
            print(f"    Find it at: https://datacollective.mozillafoundation.org/datasets?q=common_voice")
            continue

        print(f"\n{'='*50}")
        print(f"  Downloading {lang_name} ({lang_code}) from MDC...")
        print(f"  Dataset ID: {dataset_id}")
        print(f"{'='*50}")

        try:
            # Download dataset archive to data/raw/<lang_code>/
            lang_raw_dir = os.path.join(RAW_DATA_DIR, lang_code)
            os.makedirs(lang_raw_dir, exist_ok=True)

            dataset_path = save_dataset_to_disk(
                dataset_id,
                output_dir=lang_raw_dir,
            )
            print(f"  Downloaded to: {dataset_path}")

            # Extract if it's a tar.gz archive
            if dataset_path and dataset_path.endswith((".tar.gz", ".tgz")):
                print(f"  Extracting archive...")
                with tarfile.open(dataset_path, "r:gz") as tar:
                    tar.extractall(path=lang_raw_dir)

            # Find all audio files (Common Voice uses .mp3)
            audio_files = sorted(
                glob.glob(os.path.join(lang_raw_dir, "**", "*.mp3"), recursive=True)
                + glob.glob(os.path.join(lang_raw_dir, "**", "*.wav"), recursive=True)
                + glob.glob(os.path.join(lang_raw_dir, "**", "*.flac"), recursive=True)
            )

            if not audio_files:
                print(f"  ⚠ No audio files found in {lang_raw_dir}")
                continue

            print(f"  Found {len(audio_files)} audio files, processing up to {samples_per_lang}...")

            count = 0
            for fpath in tqdm(audio_files, desc=lang_name):
                if count >= samples_per_lang:
                    break

                y = preprocess_audio(fpath)
                if y is None:
                    continue

                sample_id = f"{lang_code}_{count:05d}"
                npy_path = os.path.join(PROCESSED_DATA_DIR, f"{sample_id}.npy")
                np.save(npy_path, y)

                records.append({
                    "sample_id": sample_id,
                    "language_code": lang_code,
                    "language": lang_name,
                    "npy_path": npy_path,
                    "label": CODE_TO_LABEL[lang_code],
                })
                count += 1

            print(f"  ✅ {lang_name}: {count} samples processed")

        except Exception as e:
            print(f"  ❌ Failed to load {lang_name}: {e}")
            print(f"     Make sure you have:")
            print(f"       1. pip install datacollective")
            print(f"       2. Set MDC_API_KEY environment variable")
            print(f"       3. Accepted dataset terms on the MDC website")
            continue

    return pd.DataFrame(records)


# ── HuggingFace Common Voice streaming (legacy fallback) ─────────

def load_streaming_data(samples_per_lang: int = SAMPLES_PER_LANGUAGE) -> pd.DataFrame:
    """Stream audio from HuggingFace Common Voice (legacy fallback).

    NOTE: Common Voice has moved to Mozilla Data Collective.
    Use load_mdc_data() instead. This function may stop working
    if Mozilla removes the HuggingFace mirror.

    Requires: pip install datasets

    Returns a manifest DataFrame.
    """
    from datasets import load_dataset, Audio

    print("⚠ Using legacy HuggingFace streaming. Common Voice has moved to:")
    print("  https://datacollective.mozillafoundation.org/datasets?q=common_voice")
    print("  Consider switching to MDC (use_mdc=True).\n")

    records = []

    for lang_code, lang_name in LANGUAGES:
        print(f"\n{'='*50}")
        print(f"  Streaming {lang_name} ({lang_code})...")
        print(f"{'='*50}")

        try:
            ds = load_dataset(
                "mozilla-foundation/common_voice_17_0",
                lang_code,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

            count = 0
            for sample in tqdm(ds, total=samples_per_lang, desc=lang_name):
                if count >= samples_per_lang:
                    break

                audio = sample["audio"]
                y = np.array(audio["array"], dtype=np.float32)

                # Skip short / silent clips
                if len(y) < SAMPLE_RATE * 0.5:
                    continue
                if np.sqrt(np.mean(y ** 2)) < 1e-5:
                    continue

                # Pad or trim
                if len(y) > TARGET_SAMPLES:
                    y = y[:TARGET_SAMPLES]
                elif len(y) < TARGET_SAMPLES:
                    y = np.pad(y, (0, TARGET_SAMPLES - len(y)))

                sample_id = f"{lang_code}_{count:05d}"
                npy_path = os.path.join(PROCESSED_DATA_DIR, f"{sample_id}.npy")
                np.save(npy_path, y)

                records.append({
                    "sample_id": sample_id,
                    "language_code": lang_code,
                    "language": lang_name,
                    "npy_path": npy_path,
                    "label": CODE_TO_LABEL[lang_code],
                })
                count += 1

            print(f"  ✅ {lang_name}: {count} samples saved")

        except Exception as e:
            print(f"  ❌ Failed to load {lang_name}: {e}")
            continue

    return pd.DataFrame(records)


# ── Train / Val / Test split ──────────────────────────────────────

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split.

    Returns (train_df, val_df, test_df).
    """
    train_df, temp_df = train_test_split(
        df, test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df["label"], random_state=42,
    )
    relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test,
        stratify=temp_df["label"], random_state=42,
    )

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    return train_df, val_df, test_df


# ── High-level loader (auto-selects mode) ────────────────────────

def load_data(raw_dir: str | None = None,
              use_mdc: bool = False,
              use_streaming: bool = False,
              samples_per_lang: int = SAMPLES_PER_LANGUAGE) -> pd.DataFrame:
    """Load or stream audio data and return a manifest DataFrame.

    Modes (in priority order):
        1. Cached manifest — reloads from MANIFEST_PATH if it exists
        2. use_mdc=True    — download from Mozilla Data Collective
        3. use_streaming   — legacy HuggingFace streaming
        4. raw_dir         — load from local directory

    If a cached manifest exists at MANIFEST_PATH, reloads it instead.
    """
    if os.path.exists(MANIFEST_PATH):
        print(f"Loading cached manifest: {MANIFEST_PATH}")
        return pd.read_csv(MANIFEST_PATH)

    if use_mdc:
        df = load_mdc_data(samples_per_lang)
    elif use_streaming:
        df = load_streaming_data(samples_per_lang)
    elif raw_dir and os.path.isdir(raw_dir):
        df = load_local_data(raw_dir)
    else:
        raise FileNotFoundError(
            f"No data found. Options:\n"
            f"  1. Set use_mdc=True to download from Mozilla Data Collective\n"
            f"  2. Set use_streaming=True for legacy HuggingFace streaming\n"
            f"  3. Provide raw_dir with audio files as raw_dir/<lang_code>/*.wav"
        )

    df.to_csv(MANIFEST_PATH, index=False)
    print(f"Manifest saved: {MANIFEST_PATH} ({len(df)} samples)")
    return df