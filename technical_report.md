# Technical Evaluation Report
## Team Databaes — Arabic Native/Non-Native Speech Classification
**Client:** Renan Partners Private Limited
**Date:** March 01, 2026

---

## 1. Executive Summary

A binary classifier was developed to distinguish Native from Non-Native Arabic speakers.
The model combines deep phonetic representations from a pretrained wav2vec2 model with
hand-crafted prosodic features. A bidirectional LSTM encodes recording-level temporal
patterns from variable-length segment sequences, and an SVM with RBF kernel produces
the final decision — chosen for its strong generalisation on small datasets with
compact, high-dimensional feature spaces.

**Final Results on Renan Test Set (n=92 recording sequences):**

| Metric | Value |
|--------|-------|
| Accuracy | 84.78% |
| F1 Score | 0.8833 |
| Precision | 0.9138 |
| Recall | 0.8548 |
| ROC AUC | 0.9237 |
| EER | 17.20% |
| 95% CI (Accuracy) | [77.2%, 91.3%] |

---

## 2. Methodology

### 2.1 Data
- **Training source:** 160 Renan recordings (Gulf dialects: SA/QA/AE/KW) +
  200 Mozilla Common Voice 24.0 Arabic clips (native reference)
- **Augmentation:** Time stretching (±10%), pitch shifting (±2 semitones), Gaussian noise —
  training segments only. Non-native (minority) received 5 augmentations; native received 1.

### 2.2 Feature Extraction

**Stream 1 — Phonetic Embeddings (768-dim)**
- Model: `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` (Baevski et al., 2020)
- Used frozen — no fine-tuning (prevents overfitting on 160 training recordings)
- Each 3-second segment → mean-pooled hidden states → 768-dimensional vector

**Stream 2 — Prosodic Features (6-dim)**

| Feature | Description |
|---------|-------------|
| F0 mean | Average fundamental frequency |
| F0 variance | Pitch dynamics |
| Speaking rate | Syllable-proxy peaks per second |
| Pause frequency | Pauses (>200ms) per second |
| Mean pause duration | Average pause length in seconds |
| nPVI | Normalised Pairwise Variability Index (rhythmic stress-timing) |

### 2.3 Fusion and Normalisation
- Streams concatenated → 774-dimensional vector per segment
- StandardScaler fitted on training data only
- Segments grouped by recording into temporal sequences

### 2.4 Classifier

**Stage 1 — LSTM Temporal Encoder**
- Architecture: Bidirectional LSTM (input_dim=1030, hidden_dim=64, num_layers=1, bidirectional=True)
- Processes variable-length sequences of 774-dim segment features per recording
- Outputs a fixed 128-dim recording-level representation
- Trained end-to-end with BCE loss (class-balanced via pos_weight) + early stopping

**Stage 2 — SVM with RBF Kernel**
- Best hyperparameters: C=1, gamma=0.01
- Best 3-fold CV F1: 0.9388
- `class_weight='balanced'`, `probability=True` (Platt scaling for confidence scores)
- RBF kernel chosen over RF for stronger margin generalisation on the compact
  128-dim LSTM features with limited training recordings

---

## 3. Results

### 3.1 Confusion Matrix

|  | Predicted Non-Native | Predicted Native |
|--|--|--|
| **True Non-Native** | 25 | 5 |
| **True Native** | 9 | 53 |

- **False Positives** (non-native classified as native): 5
- **False Negatives** (native classified as non-native): 9

### 3.2 Statistical Caveat
The 95% CI [77.2%, 91.3%] is relatively wide due to the
small test set (n=92). Results would be more stable with n≥100 test samples.

---

## 4. Known Limitations

1. **Small test set** — wide confidence intervals
2. **Kuwaiti dialect** — only ~1.9% of training data (3 files); may generalise poorly to KW test samples
3. **Short Common Voice clips** — 3–10s clips yield less reliable prosodic estimates
4. **Non-native L1 diversity** — model trained primarily on Gulf-native speakers

---

## 5. References

- Baevski et al. (2020). wav2vec 2.0. https://arxiv.org/abs/2006.11477
- Hochreiter & Schmidhuber (1997). Long Short-Term Memory. Neural Computation, 9(8).
- Cortes & Vapnik (1995). Support-Vector Networks. Machine Learning, 20(3).
- Emara & Shaker (2024). Speech Communication, 157.
- Grabe & Low (2002). Durational variability and the Rhythm Class Hypothesis.
- Sharma et al. (2021). SVM-based speech classification.
- Mozilla Common Voice 24.0 Arabic, CC0-1.0.
- jonatasgrosman/wav2vec2-large-xlsr-53-arabic. https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-arabic

---
*Report generated automatically by Stage 10 — Team Databaes*
