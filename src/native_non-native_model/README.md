# Native / Non-Native Arabic Speaker Classifier

Binary classifier that distinguishes **Native** from **Non-Native** Arabic speakers using audio recordings.

## Architecture

```
Audio (MP3/WAV) → Preprocess → 3s Segments → wav2vec2 Embeddings (1024-dim)
                                             + Prosodic Features (6-dim)
                                             ────────────────────────────
                                             Fused 1030-dim per segment
                                             ↓
                                        BiLSTM Encoder
                                        (variable-length → 128-dim)
                                             ↓
                                        SVM RBF Classifier
                                             ↓
                                    Native / Non-Native
```

### Feature Extraction

**Stream 1 — Phonetic Embeddings (1024-dim)**
- Model: `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` (frozen, no fine-tuning)
- Each 3-second segment → mean-pooled hidden states → 1024-dim vector

**Stream 2 — Prosodic Features (6-dim)**

| Feature | Description |
|---------|-------------|
| F0 mean | Average fundamental frequency |
| F0 variance | Pitch dynamics |
| Speaking rate | Syllable-proxy peaks per second |
| Pause frequency | Pauses (>200 ms) per second |
| Mean pause duration | Average pause length in seconds |
| nPVI | Normalised Pairwise Variability Index (rhythmic stress-timing) |

### Classifier

1. **BiLSTM Temporal Encoder** — Processes variable-length sequences of 1030-dim segment features per recording → fixed 128-dim representation
2. **SVM with RBF Kernel** — Classifies the 128-dim LSTM features with `class_weight='balanced'`

## Performance (Train/Test Split — 92 Recording Sequences)

| Metric | Value |
|--------|-------|
| Accuracy | 84.78% |
| F1 Score | 0.8833 |
| Precision | 0.9138 |
| Recall | 0.8548 |
| ROC AUC | 0.9237 |
| EER | 17.20% |
| 95% CI (Accuracy) | [77.2%, 91.3%] |

**Confusion Matrix:**

|  | Pred Non-Native | Pred Native |
|--|-----------------|-------------|
| **True Non-Native** | 25 | 5 |
| **True Native** | 9 | 53 |

## Data

- **Training:** 160 Renan recordings (Gulf dialects: SA/QA/AE/KW) + 200 Mozilla Common Voice Arabic clips (native reference)
- **Augmentation:** Time stretch (±10%), pitch shift (±2 semitones), Gaussian noise — 5× minority (Non-Native), 1× majority (Native)
- **Preprocessing:** Resample 16 kHz → RMS normalize → VAD trim → denoise → duration gate (≥3 s)
- **Test (unlabelled):** 40 Arabic_AE recordings from `data/Nativity Assessmet Audio Dataset(Test Dataset).csv`

## Saved Models

| File | Description |
|------|-------------|
| `models/lstm_encoder.pt.zip` | BiLSTM encoder weights (2.2 MB) |
| `models/svm_model.joblib` | SVM RBF classifier (170 KB) |
| `models/scaler.joblib` | StandardScaler fitted on training data (25 KB) |
| `models/lstm_config.joblib` | LSTM hyperparameters (input_dim, hidden_dim, etc.) |

## Setup

```bash
# Create conda environment
conda create -n native-classifier python=3.10 -y
conda activate native-classifier

# Install dependencies
pip install -r requirements.txt

# Optional: install ffmpeg for non-WAV audio formats
# Windows: choco install ffmpeg
# Linux:   apt-get install ffmpeg
# macOS:   brew install ffmpeg
```

## Usage

### Inference (using saved models)

```bash
# From project root — runs on the provided test dataset
python main.py

# Custom input
python main.py --audio-csv data/my_test.csv --output-dir results

# Force CPU
python main.py --device cpu
```

### Full Training Pipeline

```bash
# From this directory
python pipeline.py train --project-root ../../workspace --cv-api-key YOUR_KEY
```

### Pipeline via CLI

```bash
# Inference only
python pipeline.py predict \
    --audio-csv ../../data/test.csv \
    --models-dir models \
    --output-dir outputs
```

## Output

Predictions are saved to `outputs/predictions.csv` with columns:

| Column | Description |
|--------|-------------|
| dp_id | Recording identifier |
| language | Dialect (e.g. Arabic_AE) |
| predicted_label | 1 = Native, 0 = Non-Native |
| predicted_class | "Native" or "Non-Native" |
| confidence_score | Classification confidence (0–1) |
| prob_native | Probability of Native class |
| prob_non_native | Probability of Non-Native class |
| n_segments | Number of 3 s segments extracted |

## References

- Baevski et al. (2020). wav2vec 2.0. https://arxiv.org/abs/2006.11477
- Hochreiter & Schmidhuber (1997). Long Short-Term Memory. Neural Computation, 9(8).
- Cortes & Vapnik (1995). Support-Vector Networks. Machine Learning, 20(3).
- jonatasgrosman/wav2vec2-large-xlsr-53-arabic: https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-arabic
- Mozilla Common Voice 24.0 Arabic, CC0-1.0.
