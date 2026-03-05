# Arabic Native vs. Non-Native Speaker Classifier

> **Hackenza 2026 — Team Databaes**

A 10-stage automated ML pipeline that classifies Arabic speakers as **native or non-native** using deep phonetic embeddings fused with handcrafted prosodic features, encoded by a BiLSTM, and classified by an SVM with RBF kernel — no transcription or manual annotation required at inference time.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline Stages](#pipeline-stages)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Security & Reproducibility](#security--reproducibility)
- [Team](#team)

---

## Overview

The system ingests raw Arabic speech audio and outputs binary classifications (`native` / `non-native`) with confidence scores. It targets Gulf Arabic dialects (Kuwaiti, Saudi, Emirati, Qatari) and is designed to generalize across speakers without any transcription or manual labeling at inference time.

**Key Objectives:**
- Accurately classify Arabic speakers using acoustic signals alone
- Generalize across multiple Gulf Arabic dialects
- Produce confidence scores alongside predicted labels for interpretability
- Achieve strong F1-score on a 40-sample held-out test set with 95% confidence intervals

---

## Architecture

```
Audio Input
    │
    ▼
Preprocessing (resample → 16kHz, normalize, VAD, noise reduction)
    │
    ▼
Segmentation (3-second windows with configurable hop size)
    │
    ├──────────────────────────────────────────┐
    ▼                                          ▼
wav2vec2-large-xlsr-53-arabic            Parselmouth / Praat
(frozen transformer encoder)             (prosodic features)
1024-dim embedding per segment           6-dim vector per utterance
    │                                          │
    └──────────────┬───────────────────────────┘
                   ▼
           Feature Fusion (1030-dim)
                   │
                   ▼
           BiLSTM Encoder (128-dim per recording)
                   │
                   ▼
        SVM Classifier (RBF kernel)
                   │
                   ▼
      Native / Non-Native + Confidence Score
```

The core insight is that native/non-native speech differences manifest at **two levels**:
- **Deep phonetic patterns** — captured via `wav2vec2-large-xlsr-53-arabic`, a transformer pretrained on thousands of hours of Arabic speech, used as a frozen feature extractor.
- **Prosodic patterns** — pitch, speaking rate, rhythm, and pauses extracted via `parselmouth`/Praat.

---

## Pipeline Stages

| Stage | Name | Technology | Output |
|-------|------|-----------|--------|
| 1 | Data Exploration | `librosa`, `pandas` | Exploratory report |
| 2 | Data Collection | Hugging Face Datasets | Unified labelled dataset |
| 3 | Preprocessing | `librosa`, `noisereduce`, `webrtcvad` | Clean 16kHz audio |
| 4 | Audio Splitting | `pydub`, `librosa` | 3-sec segments with labels |
| 5 | Augmentation | `librosa` | Augmented training set (time stretch, pitch shift, noise) |
| 6a | wav2vec2 Embeddings | `transformers`, `torch` | 1024-dim vectors per segment |
| 6b | Prosodic Features | `parselmouth` | 6-dim vector per utterance |
| 7 | Feature Fusion | `numpy`, `scikit-learn` | 1030-dim normalized vectors |
| 8 | BiLSTM + SVM | `torch`, `scikit-learn` | Trained model + CV results |
| 9 | Evaluation | `scikit-learn`, `scipy` | Metrics + confusion matrix |
| 10 | Output | `pandas` | Predictions CSV + report |

---

## Model Performance

Evaluated on a **15% held-out test set** (92 recording-level sequences, grouped by speaker to prevent leakage):

| Metric | Value |
|--------|-------|
| **Accuracy** | **84.78%** |
| **F1 Score** | **0.8833** |
| Precision | 0.9138 |
| Recall | 0.8548 |
| ROC AUC | 0.9237 |
| EER | 17.20% |

All metrics are reported with **95% confidence intervals** via bootstrap resampling.

---

## Project Structure

```
hackenza2026_team_databaes/
├── main.py                               # Entry point — inference on test data
├── analyze_data.py                       # Exploratory data analysis (Stage 1)
├── check_env.py                          # Environment validation script
├── read_notebook.py                      # Notebook reader utility
├── environment.yml                       # Conda environment spec
├── technical_report.md                   # Full technical evaluation report
├── data/
│   ├── renan_dataset.csv                 # Training data (178 rows, labelled)
│   └── Nativity Assessment Audio...csv  # Test data (40 rows, unlabelled)
└── src/
    ├── native_non-native_model/          # Core model (primary deliverable)
    │   ├── pipeline.py                   # Full pipeline (train + inference)
    │   ├── requirements.txt
    │   ├── README.md
    │   ├── models/
    │   │   ├── lstm_encoder.pt.zip       # BiLSTM encoder weights (2.2 MB)
    │   │   ├── svm_model.joblib          # Trained SVM classifier
    │   │   ├── scaler.joblib             # StandardScaler for feature normalization
    │   │   └── lstm_config.joblib        # LSTM hyperparameter config
    │   └── outputs/
    │       ├── predictions.csv           # Train/test split results
    │       └── stage9_metrics.csv        # Evaluation metrics
    └── audio-language-classifier/        # Bonus feature — language identification
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/aarushgupta2735/hackenza2026_team_databaes.git
cd hackenza2026_team_databaes

# 2. Set up the environment
conda env create -f environment.yml
conda activate native-classifier

# Or with pip:
pip install -r src/native_non-native_model/requirements.txt

# 3. Run inference on the 40-sample test dataset
python main.py
```

---

## Installation

### Prerequisites

- Python 3.10+
- Conda (recommended) or pip
- CUDA-compatible GPU (optional but recommended for wav2vec2 inference)

### Dependencies

Key libraries used across the pipeline:

| Library | Purpose |
|---------|---------|
| `transformers` | wav2vec2 feature extraction |
| `torch` | BiLSTM encoder |
| `scikit-learn` | SVM classifier, evaluation |
| `librosa` | Audio loading, augmentation |
| `parselmouth` | Prosodic feature extraction |
| `noisereduce` | Background noise reduction |
| `webrtcvad` | Voice activity detection |
| `pydub` | Audio segmentation |
| `pandas`, `numpy` | Data handling |
| `scipy` | Confidence intervals, EER |

---

## Usage

### Inference on New Audio

```python
# Run the full inference pipeline on the test CSV
python main.py
```

Predictions are saved to `src/native_non-native_model/outputs/predictions.csv` with the following columns:

| Column | Description |
|--------|-------------|
| `filename` | Audio file identifier |
| `predicted_label` | `native` or `non-native` |
| `confidence_score` | Probability of predicted class (0–1) |

### Retraining

```python
# Run the full pipeline including training
python src/native_non-native_model/pipeline.py
```

---

## Dataset

- **Training data:** `data/renan_dataset.csv` — 178 labelled recordings provided by Renan Partners Private Limited
- **Test data:** 40 unlabelled Arabic audio recordings across Gulf dialects (Kuwaiti, Saudi, Emirati, Qatari)
- **Data augmentation** is applied to training data only (time stretch, pitch shift, additive noise) — never to validation or test sets, preventing data leakage
- **Speaker-level grouping** is enforced in all train/test splits to prevent speaker identity leakage

> ⚠️ Audio samples from Renan Partners are treated as confidential client data. No audio is uploaded to external services — all processing runs locally.

---

## Feature Engineering

### Deep Phonetic Features (1024-dim)
Extracted using [`facebook/wav2vec2-large-xlsr-53-arabic`](https://huggingface.co/facebook/wav2vec2-large-xlsr-53-arabic) — a transformer pretrained on multilingual speech. Used as a **frozen** feature extractor; not fine-tuned. Produces a 1024-dimensional embedding per 3-second segment.

### Prosodic Features (6-dim)
Extracted via `parselmouth`/Praat:
- F0 (fundamental frequency) mean and standard deviation
- Speaking rate (syllables/second)
- Pause ratio
- Rhythm regularity
- Voice onset timing

These features are language-agnostic and particularly discriminative for nativeness detection.

### BiLSTM Encoding
Segments from a single recording are fed as a sequence into a **Bidirectional LSTM**, which aggregates variable-length segment sequences into a fixed 128-dimensional recording-level representation.

---

## Security & Reproducibility

- All **random seeds** are fixed for augmentation, k-fold splitting, and model initialization
- **Augmentation** is strictly applied to training data only
- **Test set** is isolated before any preprocessing decisions are made
- wav2vec2 model weights are loaded from Hugging Face Hub
- The SVM model and StandardScaler are serialized with `joblib`
- All pipeline stages are **versioned and logged** with timestamps

---

## Team

**Team Databaes — BITS Pilani, Goa Campus**

| Name | ID |
|------|----|
| Priyanshi Arvind Modi | 2023B3A70791G |
| Aarush Gupta | 2023B3A70839G |
| Moutushi Chanda | 2023B1A80659G |
| Priyanshu Gupta | 2023B1AD0669G |

---

*Built for Hackenza 2026*