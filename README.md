# 🧠 Neuro-Fuzzy Deception Detection Using Keystroke Hesitation Patterns

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Architecture](https://img.shields.io/badge/architecture-ANFIS%20(Takagi--Sugeno)-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An end-to-end neuro-fuzzy framework that estimates the probability of typed content being **fabricated** vs. **truthful** using behavioral biometrics (*keystroke dynamics & cognitive hesitation patterns*) rather than linguistic/semantic content.

---

## 📌 Executive Summary & Scientific Basis

Formulating a false answer requires greater cognitive effort (*monitoring oneself, inventing details, avoiding contradictions*) than recalling an autobiographical truth. This mental friction physically manifests in micro-keystroke timings:
* **Initial Latency**: Thinking delay before the first keystroke.
* **Flight Time**: Latency between releasing one key and striking the next.
* **Dwell Time**: Duration a key is held depressed.
* **Pause Count**: Hesitations lasting $> 1.0\\text{s}$ mid-sentence.
* **Backspace Ratio**: Frequency of self-corrections and edits.

Rather than relying on opaque black-box deep neural networks, this project utilizes an **Adaptive Neuro-Fuzzy Inference System (ANFIS)** with Takagi-Sugeno fuzzy rules. ANFIS combines the parameter-tuning capabilities of neural networks with the **human-readable interpretability** of fuzzy logic rules.

---

## 🏗️ System Architecture

```
[1] Keystroke Timings
   (down/up timestamps in ms)
            │
            ▼
[2] Feature Extraction
   (dwell, flight, latency, pauses, backspaces)
            │
            ▼
[3] Fuzzification Layer
   (Gaussian Membership Functions: LOW / MEDIUM)
            │
            ▼
[4] Takagi-Sugeno Fuzzy Rules
   (IF latency is MEDIUM AND backspace is LOW THEN score = ...)
            │
            ▼
[5] Deception Scoring Engine
   (Continuous Score 0.0 – 1.0 + Calibrated Decision Cutoff)
            │
   ┌────────┴────────┐
   ▼                 ▼
[Live Streamlit UI] [Benchmark Evaluation vs. SVM & Threshold]
```

---

## 🚀 How to Run the Project

### 1. Prerequisites & Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/sameerrana/Neuro-Fuzzy-Deception-Detection-Using-Keystroke-Hesitation-Patterns.git
cd Neuro-Fuzzy-Deception-Detection-Using-Keystroke-Hesitation-Patterns

pip install -r requirements.txt
```

---

### 2. Run the Interactive Streamlit Web Demo (Phase 5)

Launch the live interactive dashboard:
```bash
streamlit run src/demo/app.py
```
* **Mode 1 (Live Typing)**: Select a truthful or fabrication prompt, type your answer, and receive an instant hesitation score.
* **Mode 2 (Playground Sliders)**: Adjust latency, dwell, flight, and backspace sliders to inspect rule activations in real-time.
* **Mode 3 (Architecture & Evaluation)**: View ROC curves, membership function before/after plots, and the full fuzzy rule base.

---

### 3. Run Instant Single-Sample Scoring (CLI)

Test any custom typing timing profile directly from your terminal:
```bash
python src/scoring/score.py --latency 1400 --dwell 105 --flight 210 --pauses 4 --backspaces 0.12
```

---

### 4. Retrain Models & Reproduce Evaluations

To rerun the full training pipeline, baselines, and sensitivity evaluations:

```bash
# 1. Generate synthetic keystroke dataset
python src/synthetic/generate.py --n_users 10 --sessions 100 --effect 0.3 --seed 42

# 2. Prepare processed features
python src/features/extract.py synthetic --input data/synthetic/synthetic_dataset_effect0.3_seed42.csv --output data/processed/features.csv

# 3. Train ANFIS model and export learned rules
python src/model/train.py --epochs 150 --lr 0.005 --n_mfs 2

# 4. Run baseline benchmarks (SVM & Threshold Rule)
python src/model/baselines.py

# 5. Run 5-seed stability analysis and sensitivity sweep
python src/model/evaluate_comprehensive.py
```

---

## 📊 Experimental Results (Test Set Performance)

All models evaluated on a **subject-held-out split** (evaluating on users unseen during training):

| Model | Accuracy | Recall | F1 Score | ROC-AUC | Interpretability |
|---|---|---|---|---|---|
| **Fixed Threshold (Latency)** | 52.0% | 0.920 | 0.657 | 0.598 | Weak (single metric) |
| **SVM (RBF Kernel)** | 59.0% | 0.590 | 0.590 | 0.608 | ❌ Black box |
| **ANFIS (Our Model)** | 50.5% | **0.980** | **0.664** | 0.561 | ✅ **Transparent Fuzzy Rules** |

### Sample Learned Takagi-Sugeno Rules:
> **Rule 2:** `IF initial_latency is MEDIUM AND mean_dwell is LOW AND mean_flight is LOW AND pause_count is LOW AND backspace_ratio is MEDIUM THEN deception_score ≈ +8.5177 [HIGH (Deceptive)]`

---

## 📁 Repository Structure

```
├── .gitignore                   # Clean ignore rules (excludes caches, OS files)
├── README.md                    # Project documentation & run guide
├── PROJECT_FOUNDATION.md        # Initial research foundation and specs
├── requirements.txt             # Project dependencies
│
├── data/
│   ├── raw/                     # Real keystroke logs (optional pilot)
│   ├── synthetic/               # Generated datasets (log-normal distributions)
│   └── processed/               # Normalized feature matrices
│
├── src/
│   ├── collection/
│   │   ├── prompts.py           # Standard truthful vs. fabrication prompts
│   │   └── logger.py            # Keystroke event listener
│   ├── synthetic/
│   │   └── generate.py          # Synthetic keystroke generator with cognitive load effect
│   ├── features/
│   │   └── extract.py           # Dwell, flight, latency, pause & backspace extractor
│   ├── model/
│   │   ├── anfis_model.py       # Pure-NumPy Takagi-Sugeno ANFIS implementation
│   │   ├── train.py             # Training loop, subject-split & rule export
│   │   ├── baselines.py         # SVM and threshold baseline benchmarks
│   │   └── evaluate_comprehensive.py # Multi-seed stability & sensitivity sweep
│   ├── scoring/
│   │   └── score.py             # Inference module & formatted report generator
│   └── demo/
│       └── app.py               # Interactive Streamlit live demo application
│
└── results/
    ├── anfis_model.npz          # Serialized ANFIS model weights
    ├── scaler.npz               # Fitted feature normalizer parameters
    ├── metrics_summary.json     # Head-to-head model comparison
    ├── comprehensive_evaluation.json # Multi-seed stability & sensitivity metrics
    ├── rules/
    │   ├── learned_rules.txt    # Exported readable fuzzy if-then rules
    │   └── mf_params.txt        # Learned Gaussian membership centers and widths
    └── figures/
        ├── loss_curve.png       # ANFIS training loss convergence
        ├── mf_before.png        # Membership functions before training
        ├── mf_after.png         # Membership functions after training
        ├── roc_comparison_all.png # 3-model ROC curve overlay
        ├── model_comparison_all.png # 3-model performance bar chart
        └── sensitivity_sweep.png # Effect magnitude sensitivity curve
```

---

## ⚖️ Academic Disclosure & Ethics

* **Semi-Synthetic Nature**: Keystroke timing statistics were modeled using log-normal distributions grounded in established behavioral biometrics literature. 
* **Scope**: This repository demonstrates an end-to-end neuro-fuzzy computational pipeline; it does not claim to serve as a definitive legal or forensic lie detector in unstructured real-world scenarios.
