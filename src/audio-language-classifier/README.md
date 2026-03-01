# Audio Language Classifier

Classifies audio samples by language using **wav2vec2-xls-r-300m** embeddings and an ensemble of LightGBM, SVM, and Logistic Regression classifiers.

Supports **5 languages** out of the box: Arabic, English, French, Spanish, and Chinese — easily extensible to any language covered by the wav2vec2 model.

## Architecture

```
Audio file (.wav / .mp3 / .flac)
        │
        ▼
┌──────────────────────────┐
│  Preprocessing           │
│  • Resample to 16 kHz    │
│  • Trim silence          │
│  • Peak-normalize [-1,1] │
│  • Pad / trim to 3 s     │
└──────────┬───────────────┘
           ▼
┌──────────────────────────────────────┐
│  wav2vec2-xls-r-300m  (frozen)       │
│  Mean-pool over time → 1024-dim vec  │
└──────────┬───────────────────────────┘
           ▼
┌──────────────────────────┐
│  StandardScaler          │
└──────────┬───────────────┘
           ▼
┌──────────────────────────────────┐
│  Ensemble Classifiers            │
│  ├── LightGBM   (high-dim)      │
│  ├── SVM RBF    (subsampled)    │
│  └── Logistic Regression        │
│                                  │
│  Best model auto-selected on F1  │
└──────────┬───────────────────────┘
           ▼
   Predicted Language + Confidence
```

## Benchmark (50 samples / language, CPU)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| LightGBM | 81.58 % | 0.8130 |
| SVM (RBF) | 76.32 % | 0.7627 |
| **Logistic Regression** | **92.11 %** | **0.9212** |

> Accuracy improves significantly with more training samples (`--samples 200+`).

## Project Structure

```
audio-language-classifier/
├── src/
│   ├── main.py                 # CLI entry point
│   ├── language_classifier.py  # High-level orchestrator
│   ├── feature_extraction.py   # wav2vec2 embedding extraction
│   ├── preprocessing.py        # Audio preprocessing pipeline
│   ├── train.py                # Training pipeline
│   ├── predict.py              # Inference pipeline
│   ├── config/
│   │   └── settings.py         # All configuration & hyperparameters
│   ├── models/
│   │   └── classifier_model.py # LightGBM + SVM + LR ensemble
│   └── utils/
│       ├── audio_utils.py      # Audio I/O helpers
│       └── data_loader.py      # Data loading (FLEURS / local / MDC)
├── data/
│   ├── raw/                    # Raw audio: raw/<lang_code>/*.wav
│   ├── processed/              # Preprocessed .npy segments
│   ├── embeddings/             # Cached wav2vec2 embeddings
│   └── metadata/
├── models/                     # Saved .joblib model files
├── outputs/                    # Predictions, metrics, manifest
├── notebooks/
│   ├── language_classifier_colab.ipynb
│   └── whisper_svm_rbf/
├── tests/
├── requirements.txt
└── setup.py
```

## Prerequisites

- Python 3.10+
- Conda (recommended) or pip

## Installation

### Option A — Conda (recommended)

```bash
git clone https://github.com/aarushgupta2735/hackenza2026_team_databaes.git
cd hackenza2026_team_databaes

# Create the environment (includes PyTorch, librosa, transformers, etc.)
conda env create -f environment.yml
conda activate native-classifier

# Install additional classifier deps
pip install lightgbm protobuf
```

### Option B — pip only

```bash
git clone https://github.com/aarushgupta2735/hackenza2026_team_databaes.git
cd hackenza2026_team_databaes/src/audio-language-classifier
pip install -r requirements.txt
pip install lightgbm protobuf
```

## Usage

All commands are run from the `src/audio-language-classifier/` directory.

### Train from Google FLEURS (streaming — no download required)

```bash
# Quick test (50 samples/lang ≈ 2 min on CPU)
python -m src.main train --streaming --samples 50

# Full training (200+ samples/lang for better accuracy)
python -m src.main train --streaming --samples 200
```

### Train from local audio files

Organize audio as `data/raw/<lang_code>/*.wav` (e.g. `data/raw/en/`, `data/raw/fr/`):

```bash
python -m src.main train --raw-dir data/raw
```

### Predict the language of audio files

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

### Evaluate on the held-out test set

```bash
python -m src.main evaluate
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

## Configuration

Edit [`src/config/settings.py`](src/config/settings.py) to:

| Setting | Default | Description |
|---------|---------|-------------|
| `LANGUAGES` | ar, en, fr, es, zh-CN | Languages to classify |
| `SAMPLES_PER_LANGUAGE` | 500 | Training samples per language |
| `WAV2VEC_MODEL` | `facebook/wav2vec2-xls-r-300m` | Embedding model |
| `SEGMENT_SEC` | 3.0 | Audio clip length (seconds) |
| `LGB_PARAMS` | see file | LightGBM hyperparameters |
| `SVM_PARAMS` | see file | SVM hyperparameters |

### Adding a new language

1. Add a `(lang_code, "Name")` tuple to `LANGUAGES` in `settings.py`
2. Add the FLEURS mapping in `data_loader.py` → `FLEURS_LANG_MAP`
3. Re-train: `python -m src.main train --streaming`

## Data Sources

| Source | Flag | Notes |
|--------|------|-------|
| [Google FLEURS](https://huggingface.co/datasets/google/fleurs) | `--streaming` | Free, 102 languages, streams from HuggingFace |
| Local audio files | `--raw-dir` | Organize as `data/raw/<lang_code>/*.wav` |
| [Mozilla Data Collective](https://datacollective.mozillafoundation.org/) | `--mdc` | Requires API key & dataset ToS acceptance |

## How It Works

1. **Preprocessing** — Audio is loaded, resampled to 16 kHz, silence-trimmed, peak-normalized, and padded/trimmed to exactly 3 seconds.

2. **Feature Extraction** — Each 3-second clip is passed through the frozen **wav2vec2-xls-r-300m** model. The last hidden states are mean-pooled over the time dimension to produce a single 1024-dimensional embedding.

3. **Classification** — Three classifiers are trained on the embeddings:
   - **LightGBM** — gradient-boosted trees, handles high-dimensional features well
   - **SVM (RBF kernel)** — strong generalization, subsampled for speed
   - **Logistic Regression** — fast baseline with multinomial output

4. **Model Selection** — The model with the highest macro F1 score on the test set is automatically saved as the "best" model for inference.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'src'` | Run from the `audio-language-classifier/` directory |
| `torch.load` security error | Upgrade PyTorch: `pip install torch>=2.6 --index-url https://download.pytorch.org/whl/cpu` |
| `datasets` script error | Use `datasets==2.21.0` (newer versions dropped script-based dataset support) |
| Slow training on CPU | Reduce `--samples` or use a GPU machine |

## Team

**Team Databaes** — Hackenza 2026

## License

MIT License. See [LICENSE](../../LICENSE) for details.