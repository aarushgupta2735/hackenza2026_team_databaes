"""
Configuration settings for the audio language classifier.

Architecture: wav2vec2-xls-r-300m (frozen) → mean-pooled embeddings → LightGBM / SVM / LR
"""

import os
import torch

# ── Project paths ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_DIR          = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR      = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
METADATA_DIR      = os.path.join(DATA_DIR, "metadata")
EMBEDDINGS_DIR    = os.path.join(DATA_DIR, "embeddings")

MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs")
LOG_DIR      = os.path.join(PROJECT_ROOT, "logs")

# Ensure directories exist
for _d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, METADATA_DIR,
           EMBEDDINGS_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Languages ─────────────────────────────────────────────────────
# Each entry: (lang_code, human_name, MDC dataset ID)
# Find dataset IDs at: https://datacollective.mozillafoundation.org/datasets?q=common_voice
# The ID is the last part of the dataset URL, e.g. .../datasets/cminc35no007no707hql26lzk
LANGUAGES = [
    ("ar",    "Arabic"),
    ("en",    "English"),
    ("fr",    "French"),
    ("es",    "Spanish"),
    ("zh-CN", "Chinese"),
]

# Mozilla Data Collective dataset IDs for Common Voice corpora.
# Look up IDs at https://datacollective.mozillafoundation.org/datasets?q=common_voice
# Replace placeholders with actual dataset IDs from the MDC website.
MDC_DATASET_IDS = {
    "ar":    "cmj8u3os6000tnxxb169x1zdc",  # Arabic
    "en":    "cmihqzerk023co20749miafhq",  # English
    "fr":    "cmj8u48ad0025nxzp2lfaluce",  # French
    "es":    "cmj8u48a70021nxzpqvs3sisu",  # Spanish
    "zh-CN": "cmj8u3q2n00vhnxxbzrjcugwc",  # Chinese
}

LANG_CODES = [code for code, _ in LANGUAGES]
LANG_NAMES = [name for _, name in LANGUAGES]
NUM_CLASSES = len(LANGUAGES)

CODE_TO_LABEL = {code: i for i, (code, _) in enumerate(LANGUAGES)}
LABEL_TO_CODE = {i: code for code, i in CODE_TO_LABEL.items()}
LABEL_TO_NAME = {i: name for i, (_, name) in enumerate(LANGUAGES)}
CODE_TO_NAME  = dict(LANGUAGES)

# ── Audio settings ────────────────────────────────────────────────
SAMPLE_RATE    = 16_000          # wav2vec2 expects 16 kHz
SEGMENT_SEC    = 3.0             # each clip padded/trimmed to this
TARGET_SAMPLES = int(SAMPLE_RATE * SEGMENT_SEC)  # 48 000

# ── Feature extraction (wav2vec2) ─────────────────────────────────
WAV2VEC_MODEL   = "facebook/wav2vec2-xls-r-300m"  # multilingual, 300M params
EMBEDDING_DIM   = 1024                             # hidden size of xls-r-300m
BATCH_SIZE      = 16                               # for embedding extraction
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ── Data split ────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SAMPLES_PER_LANGUAGE = 500       # samples to stream per language from Common Voice

# ── Classifier hyperparameters ────────────────────────────────────
# LightGBM (primary)
LGB_PARAMS = {
    "n_estimators":  500,
    "learning_rate": 0.05,
    "num_leaves":    63,
    "max_depth":     8,
    "class_weight":  "balanced",
    "random_state":  42,
    "n_jobs":        -1,
    "verbose":       -1,
}
LGB_EARLY_STOPPING = 50

# SVM (secondary) — subsampled to MAX_SVM_SAMPLES for speed
SVM_PARAMS = {
    "kernel":       "rbf",
    "C":            10,
    "gamma":        "scale",
    "class_weight": "balanced",
    "probability":  True,
    "random_state": 42,
}
MAX_SVM_SAMPLES = 5000

# Logistic Regression (baseline)
LR_PARAMS = {
    "C":           1.0,
    "class_weight": "balanced",
    "max_iter":    2000,
    "multi_class": "multinomial",
    "random_state": 42,
    "n_jobs":      -1,
}

# ── Model file names ─────────────────────────────────────────────
SCALER_PATH   = os.path.join(MODEL_DIR, "scaler.joblib")
LGB_MODEL_PATH = os.path.join(MODEL_DIR, "lgb_model.joblib")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.joblib")
LR_MODEL_PATH  = os.path.join(MODEL_DIR, "lr_model.joblib")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
MANIFEST_PATH  = os.path.join(OUTPUT_DIR, "manifest.csv")