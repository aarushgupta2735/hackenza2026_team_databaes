"""
LanguageClassifier — high-level orchestrator.

Ties together data loading, feature extraction, training, and inference
into a single easy-to-use class.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from src.config.settings import (
    RAW_DATA_DIR, EMBEDDINGS_DIR, OUTPUT_DIR,
    LABEL_TO_NAME, LABEL_TO_CODE, NUM_CLASSES,
    SAMPLE_RATE, TARGET_SAMPLES,
)
from src.feature_extraction import Wav2VecExtractor
from src.models.classifier_model import EnsembleClassifier
from src.preprocessing import preprocess_audio
from src.utils.data_loader import load_data, split_data


class LanguageClassifier:
    """End-to-end audio language classifier.

    Usage:
        clf = LanguageClassifier()
        clf.load_data(use_streaming=True)
        clf.extract_features()
        clf.train()
        result = clf.predict("path/to/audio.wav")
    """

    def __init__(self):
        self.extractor: Wav2VecExtractor | None = None
        self.classifier = EnsembleClassifier()
        self.manifest: pd.DataFrame | None = None
        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        # Feature matrices (scaled)
        self.X_train = self.y_train = None
        self.X_val = self.y_val = None
        self.X_test = self.y_test = None

    # ── 1. Data loading ───────────────────────────────────────────

    def load_dataset(self, raw_dir: str | None = None,
                     use_mdc: bool = False,
                     use_streaming: bool = False,
                     samples_per_lang: int = 500):
        """Load audio data (MDC, local, or streamed) and split."""
        self.manifest = load_data(
            raw_dir=raw_dir or RAW_DATA_DIR,
            use_mdc=use_mdc,
            use_streaming=use_streaming,
            samples_per_lang=samples_per_lang,
        )
        self.train_df, self.val_df, self.test_df = split_data(self.manifest)
        return self.manifest

    # ── 2. Feature extraction ─────────────────────────────────────

    def extract_features(self):
        """Extract wav2vec2 embeddings and build scaled feature matrices."""
        if self.manifest is None:
            raise RuntimeError("Call load_dataset() first.")

        if self.extractor is None:
            self.extractor = Wav2VecExtractor()

        all_df = pd.concat([self.train_df, self.val_df, self.test_df],
                           ignore_index=True)
        npy_paths = all_df["npy_path"].tolist()

        cache = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
        X_all = self.extractor.extract_from_npy_files(npy_paths, cache_path=cache)

        # Map back to splits
        sid_to_idx = {sid: i for i, sid in enumerate(all_df["sample_id"])}

        def _build(split_df):
            idxs = [sid_to_idx[s] for s in split_df["sample_id"] if s in sid_to_idx]
            return X_all[idxs], split_df["label"].values[:len(idxs)]

        X_tr_raw, self.y_train = _build(self.train_df)
        X_va_raw, self.y_val = _build(self.val_df)
        X_te_raw, self.y_test = _build(self.test_df)

        # Scale
        self.X_train = self.classifier.fit_scaler(X_tr_raw)
        self.X_val = self.classifier.transform(X_va_raw)
        self.X_test = self.classifier.transform(X_te_raw)

        print(f"Feature matrices — Train: {self.X_train.shape}, "
              f"Val: {self.X_val.shape}, Test: {self.X_test.shape}")

    # ── 3. Training ───────────────────────────────────────────────

    def train(self):
        """Train all classifiers and pick the best."""
        if self.X_train is None:
            raise RuntimeError("Call extract_features() first.")

        self.classifier.train_all(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            self.X_test, self.y_test,
        )
        self.classifier.save()

    # ── 4. Evaluation ─────────────────────────────────────────────

    def evaluate(self) -> dict:
        """Evaluate on the test set and print a full report."""
        if self.X_test is None:
            raise RuntimeError("No test data available.")

        preds = self.classifier.predict(self.X_test)
        target_names = [LABEL_TO_NAME[i] for i in range(NUM_CLASSES)]

        print("\n" + classification_report(self.y_test, preds,
                                           target_names=target_names))

        acc = accuracy_score(self.y_test, preds)
        f1 = f1_score(self.y_test, preds, average="macro")

        return {"accuracy": acc, "macro_f1": f1, "predictions": preds}

    # ── 5. Single-file inference ──────────────────────────────────

    def predict(self, audio_path: str) -> dict:
        """Predict the language of a single audio file.

        Returns dict with keys: language, language_code, confidence, all_probs
        """
        # Load the model if not already trained
        if self.classifier.best_model is None:
            self.classifier.load("best")

        if self.extractor is None:
            self.extractor = Wav2VecExtractor()

        # Preprocess
        y = preprocess_audio(audio_path)
        if y is None:
            raise ValueError(f"Audio file is too short or silent: {audio_path}")

        # Extract embedding
        embedding = self.extractor.extract_single(y)  # (1024,)
        embedding_sc = self.classifier.transform(embedding.reshape(1, -1))

        # Predict
        pred_label = int(self.classifier.predict(embedding_sc)[0])
        probs = self.classifier.predict_proba(embedding_sc)

        result = {
            "language": LABEL_TO_NAME[pred_label],
            "language_code": LABEL_TO_CODE[pred_label],
            "confidence": float(probs[0, pred_label]) if probs is not None else 1.0,
        }

        if probs is not None:
            result["all_probs"] = {
                LABEL_TO_NAME[i]: float(p) for i, p in enumerate(probs[0])
            }

        return result