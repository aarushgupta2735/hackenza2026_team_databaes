"""
Multi-class language classifiers.

Three models trained on wav2vec2 embeddings:
  1. LightGBM   (primary — fast, high-dim friendly)
  2. SVM RBF    (secondary — subsampled for speed)
  3. Logistic Regression (baseline)
"""

import numpy as np
import joblib
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from src.config.settings import (
    LGB_PARAMS, LGB_EARLY_STOPPING,
    SVM_PARAMS, MAX_SVM_SAMPLES,
    LR_PARAMS,
    SCALER_PATH, LGB_MODEL_PATH, SVM_MODEL_PATH, LR_MODEL_PATH,
    BEST_MODEL_PATH,
)


class EnsembleClassifier:
    """Trains and evaluates LightGBM, SVM, and LR classifiers."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.lgb_model = None
        self.svm_model = None
        self.lr_model = None
        self.best_model = None
        self.best_name = None

    # ── Scaling ───────────────────────────────────────────────────

    def fit_scaler(self, X_train: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(X_train)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)

    # ── Training ──────────────────────────────────────────────────

    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train a LightGBM classifier."""
        print("Training LightGBM...")
        self.lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)

        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [lgb.early_stopping(LGB_EARLY_STOPPING, verbose=True)]

        self.lgb_model.fit(X_train, y_train, **fit_kwargs)
        return self.lgb_model

    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train an SVM classifier (subsampled for speed)."""
        print("Training SVM (RBF)...")

        if X_train.shape[0] > MAX_SVM_SAMPLES:
            rng = np.random.RandomState(42)
            idx = rng.choice(X_train.shape[0], MAX_SVM_SAMPLES, replace=False)
            X_sub, y_sub = X_train[idx], y_train[idx]
            print(f"  Subsampled {X_train.shape[0]} → {MAX_SVM_SAMPLES}")
        else:
            X_sub, y_sub = X_train, y_train

        self.svm_model = SVC(**SVM_PARAMS, decision_function_shape="ovr")
        self.svm_model.fit(X_sub, y_sub)
        return self.svm_model

    def train_lr(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train a Logistic Regression baseline."""
        print("Training Logistic Regression...")
        self.lr_model = LogisticRegression(**LR_PARAMS)
        self.lr_model.fit(X_train, y_train)
        return self.lr_model

    def train_all(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray = None, y_val: np.ndarray = None,
                  X_test: np.ndarray = None, y_test: np.ndarray = None):
        """Train all three classifiers and pick the best on test set."""
        self.train_lightgbm(X_train, y_train, X_val, y_val)
        self.train_svm(X_train, y_train)
        self.train_lr(X_train, y_train)

        # Compare on test set (or val if no test)
        eval_X = X_test if X_test is not None else X_val
        eval_y = y_test if y_test is not None else y_val

        if eval_X is not None:
            results = []
            for name, model in [("LightGBM", self.lgb_model),
                                ("SVM", self.svm_model),
                                ("LR", self.lr_model)]:
                preds = model.predict(eval_X)
                acc = accuracy_score(eval_y, preds)
                f1 = f1_score(eval_y, preds, average="macro")
                results.append((name, model, acc, f1))
                print(f"  {name:<20} Acc: {acc*100:.2f}%  F1: {f1:.4f}")

            best = max(results, key=lambda x: x[3])
            self.best_name = best[0]
            self.best_model = best[1]
            print(f"\n🏆 Best model: {self.best_name}")
        else:
            # Default to LightGBM
            self.best_name = "LightGBM"
            self.best_model = self.lgb_model

        return self.best_model

    # ── Prediction ────────────────────────────────────────────────

    def predict(self, X: np.ndarray, model=None) -> np.ndarray:
        """Predict labels. Uses best model if none specified."""
        m = model or self.best_model or self.lgb_model
        return m.predict(X)

    def predict_proba(self, X: np.ndarray, model=None) -> np.ndarray | None:
        """Predict probabilities if the model supports it."""
        m = model or self.best_model or self.lgb_model
        if hasattr(m, "predict_proba"):
            return m.predict_proba(X)
        return None

    # ── Save / Load ───────────────────────────────────────────────

    def save(self):
        """Save all models and the scaler to disk."""
        joblib.dump(self.scaler, SCALER_PATH)
        if self.lgb_model:
            joblib.dump(self.lgb_model, LGB_MODEL_PATH)
        if self.svm_model:
            joblib.dump(self.svm_model, SVM_MODEL_PATH)
        if self.lr_model:
            joblib.dump(self.lr_model, LR_MODEL_PATH)
        if self.best_model:
            joblib.dump(self.best_model, BEST_MODEL_PATH)
        print(f"Models saved to {LGB_MODEL_PATH}, {SVM_MODEL_PATH}, {LR_MODEL_PATH}")

    def load(self, model_type: str = "best"):
        """Load models from disk.

        model_type: 'best', 'lgb', 'svm', or 'lr'
        """
        self.scaler = joblib.load(SCALER_PATH)

        path_map = {
            "best": BEST_MODEL_PATH,
            "lgb": LGB_MODEL_PATH,
            "svm": SVM_MODEL_PATH,
            "lr": LR_MODEL_PATH,
        }
        path = path_map.get(model_type, BEST_MODEL_PATH)
        model = joblib.load(path)

        if model_type == "lgb":
            self.lgb_model = model
        elif model_type == "svm":
            self.svm_model = model
        elif model_type == "lr":
            self.lr_model = model
        else:
            self.best_model = model

        print(f"Loaded model from {path}")
        return model