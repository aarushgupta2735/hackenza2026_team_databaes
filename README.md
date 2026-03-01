# hackenza2026_team_databaes

**Native vs Non-Native Arabic Speaker Classifier**

Binary audio classifier that distinguishes native from non-native Arabic speakers using deep phonetic embeddings (wav2vec2) fused with prosodic features, encoded by a bidirectional LSTM, and classified by an SVM with RBF kernel.

## Quick Start

```bash
# 1. Set up environment
conda create -n native-classifier python=3.10 -y
conda activate native-classifier
pip install -r src/native_non-native_model/requirements.txt

# 2. Run inference on the test dataset (40 unlabelled Arabic recordings)
python main.py
```

## Project Structure

```
hackenza2026_team_databaes/
├── main.py                              # Entry point — inference on test data
├── data/
│   ├── renan_dataset.csv                # Training data (178 rows, labelled)
│   └── Nativity Assessmet Audio ...csv  # Test data (40 rows, unlabelled)
├── src/
│   ├── native_non-native_model/         # Core model (primary deliverable)
│   │   ├── pipeline.py                  # Full pipeline (train + inference)
│   │   ├── models/                      # Pre-trained model weights
│   │   │   ├── lstm_encoder.pt.zip      # BiLSTM encoder (2.2 MB)
│   │   │   ├── svm_model.joblib         # SVM RBF classifier
│   │   │   ├── scaler.joblib            # StandardScaler
│   │   │   └── lstm_config.joblib       # LSTM hyperparameters
│   │   ├── outputs/                     # Predictions & metrics
│   │   │   ├── predictions.csv          # Train/test split results
│   │   │   └── stage9_metrics.csv       # Evaluation metrics
│   │   ├── requirements.txt
│   │   └── README.md
│   └── audio-language-classifier/       # Additional feature — language ID
│       └── ...
├── environment.yml
└── README.md
```

## Model Performance

Evaluated on a 15% held-out test set (92 recording-level sequences, grouped by speaker to prevent leakage):

| Metric | Value |
|--------|-------|
| **Accuracy** | **84.78%** |
| **F1 Score** | **0.8833** |
| Precision | 0.9138 |
| Recall | 0.8548 |
| ROC AUC | 0.9237 |
| EER | 17.20% |

## Architecture

```
Audio → Preprocess → 3s Segments → wav2vec2-xlsr-53-arabic (1024-dim)
                                   + Prosodic Features (6-dim)
                                   → 1030-dim fused per segment
                                   → BiLSTM (128-dim per recording)
                                   → SVM RBF → Native / Non-Native
```

## Team Databaes — Hackenza 2026
