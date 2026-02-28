# Audio Language Classifier

Classifies audio samples by language using **wav2vec2-xls-r-300m** embeddings and an ensemble of LightGBM, SVM, and Logistic Regression classifiers.

Supports **10 languages** out of the box: Arabic, English, French, German, Spanish, Chinese, Hindi, Japanese, Russian, Portuguese.

## Architecture

```
Audio file (.wav/.mp3/.flac)
    │
    ▼
Preprocessing (16kHz, normalize, pad/trim to 3s)
    │
    ▼
wav2vec2-xls-r-300m (frozen, multilingual)
    │  mean-pool over time → 1024-dim embedding
    ▼
StandardScaler
    │
    ▼
Ensemble Classifiers
    ├── LightGBM      (primary — fast, high-dim friendly)
    ├── SVM RBF        (secondary — subsampled for speed)
    └── Logistic Reg.  (baseline)
    │
    ▼
Predicted Language + Confidence
```

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
│       └── data_loader.py      # Data loading (local / Common Voice)
├── data/
│   ├── raw/                    # Raw audio: raw/<lang_code>/*.wav
│   ├── processed/              # Preprocessed .npy segments
│   ├── embeddings/             # Cached wav2vec2 embeddings
│   └── metadata/
├── models/                     # Saved .joblib models
├── outputs/                    # Predictions, metrics, plots
├── tests/
├── requirements.txt
└── setup.py
```

## Installation

```bash
git clone https://github.com/aarushgupta2735/hackenza2026_team_databaes.git
cd audio-language-classifier
pip install -r requirements.txt
```

## Usage

### Train from HuggingFace Common Voice (streaming — no download)

```bash
python -m src.main train --streaming --samples 200
```

### Train from local audio files

Organize audio as `data/raw/<lang_code>/*.wav` (e.g. `data/raw/en/`, `data/raw/fr/`):

```bash
python -m src.main train --raw-dir data/raw
```

### Predict language of audio files

```bash
python -m src.main predict recording.wav
python -m src.main predict --dir path/to/audio/folder/
python -m src.main predict audio1.wav audio2.mp3 --json
```

### Evaluate on test set

```bash
python -m src.main evaluate
```

### Python API

```python
from src.language_classifier import LanguageClassifier

clf = LanguageClassifier()
result = clf.predict("path/to/audio.wav")

print(result["language"])       # "French"
print(result["confidence"])     # 0.94
print(result["all_probs"])      # {"Arabic": 0.01, "English": 0.03, ...}
```

## Configuration

Edit [src/config/settings.py](src/config/settings.py) to:
- Add/remove languages
- Change samples per language
- Adjust classifier hyperparameters
- Switch wav2vec2 model variant

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.