"""
Training pipeline for the audio language classifier.

Usage:
    # Train from local data (data/raw/<lang_code>/*.wav)
    python -m src.train

    # Train from HuggingFace Common Voice streaming
    python -m src.train --streaming

    # Train with custom samples per language
    python -m src.train --streaming --samples 200
"""

import argparse
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.language_classifier import LanguageClassifier
from src.config.settings import RAW_DATA_DIR, SAMPLES_PER_LANGUAGE


def train_model(raw_dir: str = RAW_DATA_DIR,
                use_mdc: bool = False,
                use_streaming: bool = False,
                samples_per_lang: int = SAMPLES_PER_LANGUAGE):
    """Run the full training pipeline."""
    print("=" * 55)
    print("  Audio Language Classifier — Training Pipeline")
    print("=" * 55)

    clf = LanguageClassifier()

    # Stage 1: Load data
    print("\n[1/4] Loading data...")
    clf.load_dataset(
        raw_dir=raw_dir,
        use_mdc=use_mdc,
        use_streaming=use_streaming,
        samples_per_lang=samples_per_lang,
    )

    # Stage 2: Extract features
    print("\n[2/4] Extracting wav2vec2 embeddings...")
    clf.extract_features()

    # Stage 3: Train classifiers
    print("\n[3/4] Training classifiers...")
    clf.train()

    # Stage 4: Evaluate
    print("\n[4/4] Evaluating on test set...")
    metrics = clf.evaluate()

    print(f"\n{'='*55}")
    print(f"  ✅ Training complete!")
    print(f"  Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1 : {metrics['macro_f1']:.4f}")
    print(f"{'='*55}")

    return clf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the audio language classifier."
    )
    parser.add_argument(
        "--raw-dir", type=str, default=RAW_DATA_DIR,
        help="Path to raw audio directory (data/raw/<lang_code>/*.wav)",
    )
    parser.add_argument(
        "--mdc", action="store_true",
        help="Download from Mozilla Data Collective (recommended)",
    )
    parser.add_argument(
        "--streaming", action="store_true",
        help="Legacy: stream from HuggingFace Common Voice",
    )
    parser.add_argument(
        "--samples", type=int, default=SAMPLES_PER_LANGUAGE,
        help=f"Samples per language (default: {SAMPLES_PER_LANGUAGE})",
    )

    args = parser.parse_args()
    train_model(
        raw_dir=args.raw_dir,
        use_mdc=args.mdc,
        use_streaming=args.streaming,
        samples_per_lang=args.samples,
    )