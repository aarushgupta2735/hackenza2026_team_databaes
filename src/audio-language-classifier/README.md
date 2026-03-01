# Audio Language Classifier

Classifies spoken language from audio using **Facebook MMS-LID-256** (Massively Multilingual Speech — Language Identification) embeddings and an ensemble of classical ML classifiers.

Supports **5 languages**: Arabic, English, French, Spanish, and Chinese — easily extensible to any of the 256 languages supported by the MMS-LID model.

---

## Setup

### Prerequisites

- Python 3.10+
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) (recommended)
- Git

### Installation

**Option A — Conda (recommended)**

```bash
git clone https://github.com/aarushgupta2735/hackenza2026_team_databaes.git
cd hackenza2026_team_databaes

# Create and activate the environment
conda env create -f environment.yml
conda activate native-classifier

# Install additional dependencies
pip install lightgbm protobuf
```

**Option B — pip only**

```bash
git clone https://github.com/aarushgupta2735/hackenza2026_team_databaes.git
cd hackenza2026_team_databaes/src/audio-language-classifier
pip install -r requirements.txt
pip install lightgbm protobuf
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥ 2.6 | Tensor operations, model inference |
| transformers | ≥ 5.0 | MMS-LID model loading |
| datasets | 2.21.0 | Google FLEURS data streaming |
| scikit-learn | ≥ 1.5 | SVM, Logistic Regression, PCA, scaling |
| LightGBM | ≥ 4.0 | Gradient-boosted tree classifier |
| librosa | ≥ 0.10 | Audio I/O, resampling, silence trimming |

---

## Methodology

### Architecture

```
Audio file (.wav / .mp3 / .flac)
        │
        ▼
┌───────────────────────────────┐
│  Preprocessing                │
│  • Resample to 16 kHz         │
│  • Trim silence (librosa)     │
│  • Peak-normalize to [-1, 1]  │
│  • Pad / trim to 3.0 seconds  │
└───────────┬───────────────────┘
            ▼
┌───────────────────────────────────────────────┐
│  facebook/mms-lid-256  (966 M params)         │
│  wav2vec2 fine-tuned for language ID           │
│  Projector output → 1024-dim language vector   │
│  (speaker-invariant, language-discriminative)   │
└───────────┬───────────────────────────────────┘
            ▼
┌───────────────────────────────┐
│  StandardScaler → PCA (64 d)  │
│  97.9 % variance retained     │
└───────────┬───────────────────┘
            ▼
┌───────────────────────────────────┐
│  Ensemble Classifiers             │
│  ├── LightGBM  (regularised)     │
│  ├── SVM RBF   (balanced)        │
│  └── Logistic Regression          │
│                                   │
│  Best model auto-selected on F1   │
└───────────┬───────────────────────┘
            ▼
    Predicted Language + Confidence
```

### Why MMS-LID?

Earlier iterations used **wav2vec2-xls-r-300m** (a general-purpose multilingual speech model). While this produced high internal test accuracy (~91 %), it suffered from **severe overfitting** — accuracy on completely unseen speakers from the FLEURS test split dropped to ~45 % (a 46 pp gap). The frozen xls-r model's embeddings captured speaker and content artefacts rather than language-level features.

**MMS-LID-256** solves this: it is a wav2vec2 model explicitly **fine-tuned for language identification** across 256 languages. Its projector layer produces compact, speaker-invariant embeddings that generalise across unseen speakers, reducing the overfitting gap to under 5 %.

### Pipeline Steps

1. **Audio Preprocessing** — Audio is loaded with `librosa`, resampled to 16 kHz, silence-trimmed, peak-normalised to [-1, 1], and padded or trimmed to exactly 3 seconds (48 000 samples).

2. **Feature Extraction** — Each clip is passed through the **MMS-LID-256** model. Instead of using the raw hidden states, we extract the **projector output** — a 1024-dimensional vector that is optimised for language discrimination.

3. **Dimensionality Reduction** — Embeddings are standardised (`StandardScaler`) then reduced to 64 dimensions via **PCA**, which retains ~98 % of variance while removing noise and improving classifier generalisation.

4. **Ensemble Training** — Three classifiers are trained on the reduced embeddings:
   - **LightGBM** — gradient-boosted trees with heavy regularisation (max depth 4, num leaves 15, L1/L2 = 5.0, subsample 0.6)
   - **SVM (RBF kernel)** — balanced class weights, C = 1.0
   - **Logistic Regression** — multinomial with L2 regularisation (C = 0.1)

5. **Model Selection** — The classifier with the highest macro-F1 on the held-out test split is automatically saved as the "best" model for inference.

### Data

Training data is streamed from [Google FLEURS](https://huggingface.co/datasets/google/fleurs) (train split). No manual download is required.

| Setting | Value |
|---------|-------|
| Samples per language | 500 (default) |
| Total training samples | 2 500 |
| Train / Val / Test split | 70 / 15 / 15 % |
| Audio segment length | 3.0 s at 16 kHz |

---

## Outputs & Metrics

### Internal Test Set (held-out 15 % from training split)

Trained on 500 samples/language (2 500 total), evaluated on 375 held-out samples:

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| LightGBM | 98.93 % | 0.9894 |
| **SVM (RBF)** | **99.47 %** | **0.9946** |
| Logistic Regression | 98.93 % | 0.9894 |

Best model: **SVM** (auto-selected).

**Per-language breakdown (internal test, SVM):**

| Language | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Arabic | 1.00 | 0.97 | 0.99 |
| English | 1.00 | 1.00 | 1.00 |
| French | 0.99 | 1.00 | 0.99 |
| Spanish | 0.99 | 1.00 | 0.99 |
| Chinese | 1.00 | 1.00 | 1.00 |

### Fresh Test Evaluation (unseen FLEURS test split)

To verify the model does **not** overfit, we evaluated on 200 completely fresh samples (40/language) streamed from the FLEURS **test** split — data the model has never seen during training:

| Language | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Arabic | 1.00 | 0.97 | 0.99 |
| English | 1.00 | 1.00 | 1.00 |
| French | 1.00 | 0.97 | 0.99 |
| Spanish | 0.80 | 1.00 | 0.89 |
| Chinese | 1.00 | 0.80 | 0.89 |
| **Overall** | **0.96** | **0.95** | **0.95** |

### Overfitting Analysis

| Metric | Train Set | Fresh Test | Gap |
|--------|-----------|------------|-----|
| Accuracy | 99.40 % | 95.00 % | 4.40 % |
| Macro F1 | 0.9940 | 0.9505 | 0.0435 |

**4.4 % accuracy gap — no significant overfitting.**

---

## Usage

All commands are run from the `src/audio-language-classifier/` directory.

### Train

```bash
# Recommended (500 samples/lang, ~35 min on CPU)
python -m src.main train --streaming --samples 500

# Quick test (50 samples/lang, ~5 min on CPU)
python -m src.main train --streaming --samples 50

# From local audio files (data/raw/<lang_code>/*.wav)
python -m src.main train --raw-dir data/raw
```

### Predict

```bash
# Single file
python -m src.main predict recording.wav

# Multiple files
python -m src.main predict audio1.wav audio2.mp3

# Entire directory
python -m src.main predict --dir path/to/audio/folder/

# JSON output
python -m src.main predict audio1.wav --json
```

### Evaluate

```bash
# Evaluate on held-out test set
python -m src.main evaluate

# Full overfitting check (streams fresh data from FLEURS test split)
python evaluate_overfitting.py
```

### Python API

```python
from src.language_classifier import LanguageClassifier

clf = LanguageClassifier()
result = clf.predict("path/to/audio.wav")

print(result["language"])       # "French"
print(result["language_code"])  # "fr"
print(result["confidence"])     # 0.94
print(result["all_probs"])      # {"Arabic": 0.01, "English": 0.03, ...}
```

---

## Project Structure

```
audio-language-classifier/
├── src/
│   ├── main.py                 # CLI entry point (train / predict / evaluate)
│   ├── language_classifier.py  # High-level orchestrator
│   ├── feature_extraction.py   # MMS-LID embedding extraction
│   ├── preprocessing.py        # Audio preprocessing pipeline
│   ├── train.py                # Training pipeline
│   ├── predict.py              # Inference pipeline
│   ├── config/
│   │   └── settings.py         # All configuration & hyperparameters
│   ├── models/
│   │   └── classifier_model.py # LightGBM + SVM + LR ensemble with PCA
│   └── utils/
│       ├── audio_utils.py      # Audio I/O helpers
│       └── data_loader.py      # Data loading (FLEURS / local / MDC)
├── evaluate_overfitting.py     # Overfitting check on fresh FLEURS test split
├── data/
│   ├── raw/                    # Raw audio: raw/<lang_code>/*.wav
│   ├── processed/              # Preprocessed .npy segments
│   ├── embeddings/             # Cached MMS-LID embeddings
│   └── metadata/
├── models/                     # Saved .joblib model files (gitignored)
├── outputs/                    # Manifest, metrics (gitignored)
├── notebooks/
│   ├── language_classifier_colab.ipynb
│   └── whisper_svm_rbf/
├── tests/
├── requirements.txt
└── setup.py
```

## Configuration

Edit [`src/config/settings.py`](src/config/settings.py):

| Setting | Default | Description |
|---------|---------|-------------|
| `LANGUAGES` | ar, en, fr, es, zh-CN | Languages to classify |
| `SAMPLES_PER_LANGUAGE` | 500 | Training samples per language |
| `WAV2VEC_MODEL` | `facebook/mms-lid-256` | MMS Language-ID model |
| `PCA_COMPONENTS` | 64 | PCA dimensions after embedding |
| `SEGMENT_SEC` | 3.0 | Audio clip length (seconds) |
| `LGB_PARAMS` | see file | LightGBM hyperparameters |
| `SVM_PARAMS` | see file | SVM hyperparameters |
| `LR_PARAMS` | see file | Logistic Regression hyperparameters |

### Adding a New Language

1. Add a `(lang_code, "Name")` tuple to `LANGUAGES` in `settings.py`
2. Add the FLEURS mapping in `data_loader.py` → `FLEURS_LANG_MAP`
3. Re-train: `python -m src.main train --streaming`

Any of the 256 languages supported by MMS-LID can be added.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'src'` | Run from the `audio-language-classifier/` directory |
| `torch.load` security error | Upgrade PyTorch: `pip install torch>=2.6 --index-url https://download.pytorch.org/whl/cpu` |
| `datasets` script error | Use `datasets==2.21.0` (newer versions dropped script-based dataset support) |
| Slow training on CPU | Reduce `--samples` or use a GPU machine |
| MMS-LID download slow | First run downloads ~3.9 GB; cached for subsequent runs |

## Team

**Team Databaes** — Hackenza 2026

## License

MIT License. See [LICENSE](../../LICENSE) for details.