"""
Overfitting check — evaluate the trained classifier on completely unseen data.

Pulls fresh samples from the FLEURS **test** split (model was trained on FLEURS
**train** split) and compares train-set accuracy vs held-out test-set accuracy.
A large gap (>10-15 pp) indicates overfitting.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

# ── Setup paths ───────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config.settings import (
    SAMPLE_RATE, TARGET_SAMPLES, LANGUAGES, CODE_TO_LABEL,
    LABEL_TO_NAME, NUM_CLASSES, PROCESSED_DATA_DIR, EMBEDDINGS_DIR,
)
from src.feature_extraction import Wav2VecExtractor
from src.models.classifier_model import EnsembleClassifier

# ── Config ────────────────────────────────────────────────────────
FLEURS_LANG_MAP = {
    "ar":    "ar_eg",
    "en":    "en_us",
    "fr":    "fr_fr",
    "es":    "es_419",
    "zh-CN": "cmn_hans_cn",
}

SAMPLES_PER_LANG = 40   # fresh test samples per language


def stream_fleurs_test(samples_per_lang: int = SAMPLES_PER_LANG):
    """Stream fresh audio from FLEURS **test** split."""
    from datasets import load_dataset, Audio

    records = []

    for lang_code, lang_name in LANGUAGES:
        fleurs_code = FLEURS_LANG_MAP.get(lang_code)
        if not fleurs_code:
            continue

        print(f"  Streaming {lang_name} ({lang_code}) from FLEURS test split...")

        try:
            ds = load_dataset(
                "google/fleurs",
                fleurs_code,
                split="test",            # ← TEST split, not train
                streaming=True,
                trust_remote_code=True,
            )
            ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

            count = 0
            for sample in ds:
                if count >= samples_per_lang:
                    break

                audio = sample["audio"]
                y = np.array(audio["array"], dtype=np.float32)

                if len(y) < SAMPLE_RATE * 0.5:
                    continue
                if np.sqrt(np.mean(y ** 2)) < 1e-5:
                    continue

                if len(y) > TARGET_SAMPLES:
                    y = y[:TARGET_SAMPLES]
                elif len(y) < TARGET_SAMPLES:
                    y = np.pad(y, (0, TARGET_SAMPLES - len(y)))

                sample_id = f"eval_{lang_code}_{count:05d}"
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

            print(f"    ✅ {lang_name}: {count} test samples")

        except Exception as e:
            print(f"    ❌ {lang_name}: {e}")
            continue

    return pd.DataFrame(records)


def evaluate_on_train_data(clf: EnsembleClassifier, extractor: Wav2VecExtractor):
    """Re-evaluate on the original training manifest for comparison."""
    manifest = pd.read_csv(os.path.join(PROJECT_ROOT, "outputs", "manifest.csv"))

    npy_paths = manifest["npy_path"].tolist()
    labels = manifest["label"].values

    # Extract embeddings
    cache = os.path.join(EMBEDDINGS_DIR, "train_eval_embeddings.npy")
    X_all = extractor.extract_from_npy_files(npy_paths, cache_path=cache)
    X_scaled = clf.transform(X_all)

    preds = clf.predict(X_scaled)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    return acc, f1


def main():
    print("=" * 60)
    print("  OVERFITTING CHECK — Fresh FLEURS Test Split Evaluation")
    print("=" * 60)

    # Load trained model
    print("\n[1/4] Loading trained model...")
    clf = EnsembleClassifier()
    clf.load("best")

    # Load extractor
    print("\n[2/4] Loading wav2vec2 feature extractor...")
    extractor = Wav2VecExtractor()

    # ── Train-set accuracy (for comparison) ───────────────────────
    print("\n[3/4] Evaluating on TRAINING data (for overfitting comparison)...")
    train_acc, train_f1 = evaluate_on_train_data(clf, extractor)
    print(f"  Train Accuracy : {train_acc*100:.2f}%")
    print(f"  Train Macro F1 : {train_f1:.4f}")

    # ── Fresh test-set accuracy ───────────────────────────────────
    print(f"\n[4/4] Streaming {SAMPLES_PER_LANG} fresh samples/lang from FLEURS test split...")
    test_df = stream_fleurs_test(SAMPLES_PER_LANG)

    if len(test_df) == 0:
        print("❌ No test data loaded!")
        return

    # Extract embeddings for fresh test data
    print(f"\n  Extracting embeddings for {len(test_df)} test samples...")
    npy_paths = test_df["npy_path"].tolist()
    labels = test_df["label"].values

    cache = os.path.join(EMBEDDINGS_DIR, "fresh_test_embeddings.npy")
    if os.path.exists(cache):
        os.remove(cache)  # Always recompute for fresh data

    X_test = extractor.extract_from_npy_files(npy_paths, cache_path=cache)
    X_test_scaled = clf.transform(X_test)

    # Predict
    preds = clf.predict(X_test_scaled)

    # ── Results ───────────────────────────────────────────────────
    target_names = [LABEL_TO_NAME[i] for i in range(NUM_CLASSES)]

    test_acc = accuracy_score(labels, preds)
    test_f1 = f1_score(labels, preds, average="macro")

    print("\n" + "=" * 60)
    print("  FRESH TEST SET — Classification Report")
    print("=" * 60)
    print(classification_report(labels, preds, target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(f"{'':>12}", end="")
    for name in target_names:
        print(f"{name:>10}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{target_names[i]:>12}", end="")
        for val in row:
            print(f"{val:>10}", end="")
        print()

    # ── Overfitting analysis ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  OVERFITTING ANALYSIS")
    print("=" * 60)
    print(f"  {'Metric':<20} {'Train':>10} {'Test (fresh)':>15} {'Gap':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Accuracy':<20} {train_acc*100:>9.2f}% {test_acc*100:>14.2f}% {(train_acc-test_acc)*100:>9.2f}%")
    print(f"  {'Macro F1':<20} {train_f1:>10.4f} {test_f1:>15.4f} {(train_f1-test_f1):>10.4f}")

    gap = (train_acc - test_acc) * 100
    if gap > 15:
        print(f"\n  ⚠️  OVERFITTING DETECTED — {gap:.1f}% accuracy gap!")
        print(f"  Recommendations:")
        print(f"    • Increase training samples (--samples 200+)")
        print(f"    • Add regularization to LightGBM")
        print(f"    • Consider cross-validation")
    elif gap > 8:
        print(f"\n  ⚡ MILD OVERFITTING — {gap:.1f}% accuracy gap.")
        print(f"  Consider increasing training data.")
    else:
        print(f"\n  ✅ NO SIGNIFICANT OVERFITTING — {gap:.1f}% accuracy gap is acceptable.")

    print()

    # Cleanup eval npy files
    for _, row in test_df.iterrows():
        try:
            os.remove(row["npy_path"])
        except OSError:
            pass
    for cache_file in ["fresh_test_embeddings.npy", "train_eval_embeddings.npy"]:
        try:
            os.remove(os.path.join(EMBEDDINGS_DIR, cache_file))
        except OSError:
            pass


if __name__ == "__main__":
    main()
