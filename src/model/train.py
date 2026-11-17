"""
src/model/train.py
-------------------
Full training pipeline:
  1. Load processed feature CSV
  2. Split by user (subject-held-out) to prevent data leakage
  3. Normalise features with MinMaxScaler
  4. Train ANFIS
  5. Find optimal threshold on validation set
  6. Evaluate on held-out test set
  7. Export rules, MF params, loss curve, and metrics.json

Usage
-----
  python src/model/train.py                          # defaults
  python src/model/train.py --data data/processed/features.csv
  python src/model/train.py --epochs 200 --lr 0.005 --n_mfs 3
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Lightweight replacements for sklearn (keeps the script dependency-free)
# ---------------------------------------------------------------------------

class MinMaxScaler:
    """MinMax normalisation: scales each feature to [0, 1]."""
    def fit_transform(self, X):
        self.min_ = X.min(axis=0)
        self.range_ = np.where(X.max(axis=0) - self.min_ == 0, 1,
                               X.max(axis=0) - self.min_)
        return (X - self.min_) / self.range_

    def transform(self, X):
        return (X - self.min_) / self.range_

    def save(self, filepath: str):
        np.savez(filepath, min_=self.min_, range_=self.range_)

    @classmethod
    def load(cls, filepath: str):
        scaler = cls()
        data = np.load(filepath)
        scaler.min_ = data["min_"]
        scaler.range_ = data["range_"]
        return scaler

def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))

def _tp_fp_fn(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, fp, fn

def precision_score(y_true, y_pred, zero_division=0):
    tp, fp, _ = _tp_fp_fn(y_true, y_pred)
    return float(tp / (tp + fp)) if (tp + fp) > 0 else float(zero_division)

def recall_score(y_true, y_pred, zero_division=0):
    tp, _, fn = _tp_fp_fn(y_true, y_pred)
    return float(tp / (tp + fn)) if (tp + fn) > 0 else float(zero_division)

def f1_score(y_true, y_pred, zero_division=0):
    p = precision_score(y_true, y_pred, zero_division)
    r = recall_score(y_true, y_pred, zero_division)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else float(zero_division)

def roc_auc_score(y_true, y_prob):
    """Trapezoidal AUC via threshold sweep."""
    thresholds = np.sort(np.unique(y_prob))[::-1]
    tprs, fprs = [0.0], [0.0]
    pos = np.sum(y_true == 1); neg = np.sum(y_true == 0)
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp, fp, fn = _tp_fp_fn(y_true, pred)
        tprs.append(tp / pos if pos else 0)
        fprs.append(fp / neg if neg else 0)
    tprs.append(1.0); fprs.append(1.0)
    trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trap_fn(tprs, fprs))

def confusion_matrix(y_true, y_pred):
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    return np.array([[tn, fp], [fn, tp]])

# Local imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.anfis_model import ANFIS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "initial_latency",
    "mean_dwell",
    "mean_flight",
    "pause_count",
    "backspace_ratio",
]

MF_LABELS = ["LOW", "MEDIUM", "HIGH"]

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
RULES_DIR   = os.path.join(RESULTS_DIR, "rules")


# ---------------------------------------------------------------------------
# Subject-held-out split
# ---------------------------------------------------------------------------

def subject_split(df: pd.DataFrame, train_frac: float = 0.6,
                  val_frac: float = 0.2, seed: int = 42):
    """
    Split users into train / validation / test groups.
    test_frac = 1 - train_frac - val_frac.

    Why by subject?
      If we split by row, samples from the same user appear in both
      train and test, inflating accuracy because the model learns the
      user's typing style rather than the deception signal.
    """
    users = df["user_id"].unique()
    rng   = np.random.default_rng(seed)
    rng.shuffle(users)

    n_train = max(1, int(len(users) * train_frac))
    n_val   = max(1, int(len(users) * val_frac))

    train_users = users[:n_train]
    val_users   = users[n_train:n_train + n_val]
    test_users  = users[n_train + n_val:]

    train_df = df[df["user_id"].isin(train_users)]
    val_df   = df[df["user_id"].isin(val_users)]
    test_df  = df[df["user_id"].isin(test_users)]

    print(f"\n[split] Train users: {len(train_users)} ({len(train_df)} sessions)")
    print(f"[split] Val   users: {len(val_users)} ({len(val_df)} sessions)")
    print(f"[split] Test  users: {len(test_users)} ({len(test_df)} sessions)")

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob, split_name: str) -> dict:
    metrics = {
        "split":     split_name,
        "accuracy":  round(accuracy_score(y_true, y_pred),  4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0),    4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0),        4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob),   4),
    }
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def print_metrics(m: dict) -> None:
    print(f"\n--- {m['split']} metrics ---")
    for k, v in m.items():
        if k not in ("split", "confusion_matrix"):
            print(f"  {k:12s}: {v}")
    cm = np.array(m["confusion_matrix"])
    print(f"  Confusion matrix:\n{cm}")


# ---------------------------------------------------------------------------
# Threshold selection (on validation set)
# ---------------------------------------------------------------------------

def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Sweep thresholds 0.1..0.9 and pick the one that maximises F1
    on the validation set.
    """
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.95, 0.05):
        y_pred = (y_prob >= t).astype(int)
        f = f1_score(y_true, y_pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    print(f"\n[threshold] Best threshold = {best_t:.2f}  (val F1 = {best_f1:.4f})")
    return float(best_t)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_loss_curve(train_losses, val_losses, path):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train MSE")
    if val_losses:
        plt.plot(val_losses, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("ANFIS Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot] Loss curve → {path}")


def plot_membership_functions(anfis: ANFIS, feature_names: list,
                              title_suffix: str, save_path: str):
    """Plot Gaussian MFs for each input feature."""
    n = len(feature_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
    if n == 1:
        axes = [axes]

    colors = ["steelblue", "darkorange", "green"]
    labels = MF_LABELS[:anfis.n_mfs]

    for i, (ax, feat) in enumerate(zip(axes, feature_names)):
        x_range = np.linspace(0, 1, 300)   # normalised [0,1]
        for k in range(anfis.n_mfs):
            c = anfis.centers[i, k]
            s = anfis.sigmas[i, k]
            y = np.exp(-0.5 * ((x_range - c) / (s + 1e-9)) ** 2)
            ax.plot(x_range, y, color=colors[k % 3], label=labels[k])
        ax.set_title(feat, fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Normalised value")
        if i == 0:
            ax.set_ylabel("Membership degree")
        ax.legend(fontsize=7)

    fig.suptitle(f"Membership Functions — {title_suffix}", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] MF plot → {save_path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(data_path: str = "data/processed/features.csv",
          epochs: int    = 100,
          lr: float      = 0.01,
          n_mfs: int     = 3,
          seed: int      = 42):

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RULES_DIR,   exist_ok=True)

    # 1. Load data
    df = pd.read_csv(data_path)
    print(f"\n[train] Loaded {len(df)} sessions from {data_path}")
    print(f"[train] Label distribution:\n{df['label'].value_counts().to_string()}")

    # 2. Subject-held-out split
    train_df, val_df, test_df = subject_split(df, seed=seed)

    def to_XY(split_df):
        X = split_df[FEATURE_NAMES].values.astype(np.float64)
        y = split_df["label"].values.astype(np.float64)
        return X, y

    X_tr, y_tr = to_XY(train_df)
    X_va, y_va = to_XY(val_df)
    X_te, y_te = to_XY(test_df)

    # 3. Normalise (fit scaler on train ONLY — no leakage)
    scaler = MinMaxScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_va   = scaler.transform(X_va)
    X_te   = scaler.transform(X_te)

    # 4. Build & train ANFIS
    print(f"\n[train] Training ANFIS  "
          f"(inputs={len(FEATURE_NAMES)}, n_mfs={n_mfs}, "
          f"rules={n_mfs**len(FEATURE_NAMES)}, lr={lr}, epochs={epochs})")

    anfis = ANFIS(n_inputs=len(FEATURE_NAMES), n_mfs=n_mfs,
                  lr=lr, epochs=epochs)

    # Snapshot MF params before training for before/after plot
    centers_before = anfis.centers.copy()
    sigmas_before  = anfis.sigmas.copy()

    # We need init_params called once before we can snapshot — trigger it
    anfis._init_params(X_tr)
    centers_before = anfis.centers.copy()
    sigmas_before  = anfis.sigmas.copy()

    # Plot MFs before training
    plot_membership_functions(anfis, FEATURE_NAMES,
                              title_suffix="Before Training",
                              save_path=os.path.join(FIGURES_DIR, "mf_before.png"))

    # Train
    anfis.fit(X_tr, y_tr, X_va, y_va, verbose=True)

    # 5. Plot MFs after training & loss curve
    plot_membership_functions(anfis, FEATURE_NAMES,
                              title_suffix="After Training",
                              save_path=os.path.join(FIGURES_DIR, "mf_after.png"))
    plot_loss_curve(anfis.train_losses, anfis.val_losses,
                    os.path.join(FIGURES_DIR, "loss_curve.png"))

    # 6. Find best threshold on validation set
    y_va_prob = anfis.predict_proba(X_va)
    threshold  = find_best_threshold(y_va, y_va_prob)

    # 7. Evaluate
    all_metrics = []

    for X, y, name in [(X_tr, y_tr, "Train"),
                       (X_va, y_va, "Validation"),
                       (X_te, y_te, "Test")]:
        y_prob = anfis.predict_proba(X)
        y_pred = (y_prob >= threshold).astype(int)
        m = compute_metrics(y, y_pred, y_prob, name)
        print_metrics(m)
        all_metrics.append(m)

    # 8. Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "anfis_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"threshold": threshold,
                   "n_mfs": n_mfs,
                   "epochs": epochs,
                   "lr": lr,
                   "metrics": all_metrics}, f, indent=2)
    print(f"\n[train] Metrics → {metrics_path}")

    # 9. Export fuzzy rules
    rules_text = anfis.export_rules(FEATURE_NAMES, MF_LABELS[:n_mfs])
    rules_path = os.path.join(RULES_DIR, "learned_rules.txt")
    with open(rules_path, "w") as f:
        f.write(rules_text)
    print(rules_text)
    print(f"[train] Rules → {rules_path}")

    # 10. MF params summary
    mf_summary = anfis.mf_summary(FEATURE_NAMES, MF_LABELS[:n_mfs])
    print(mf_summary)
    mf_path = os.path.join(RULES_DIR, "mf_params.txt")
    with open(mf_path, "w") as f:
        f.write(mf_summary)
    print(f"[train] MF params → {mf_path}")

    # 11. Save trained model weights & scaler for scoring and demo
    model_save_path = os.path.join(RESULTS_DIR, "anfis_model.npz")
    scaler_save_path = os.path.join(RESULTS_DIR, "scaler.npz")
    anfis.save(model_save_path)
    scaler.save(scaler_save_path)
    print(f"[train] Model weights → {model_save_path}")
    print(f"[train] Scaler → {scaler_save_path}")

    # Return the trained model and scaler for the demo
    return anfis, scaler, threshold


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ANFIS deception model.")
    parser.add_argument("--data",   default="data/processed/features.csv")
    parser.add_argument("--epochs", type=int,   default=100)
    parser.add_argument("--lr",     type=float, default=0.01)
    parser.add_argument("--n_mfs",  type=int,   default=3)
    parser.add_argument("--seed",   type=int,   default=42)
    args = parser.parse_args()

    train(data_path=args.data,
          epochs=args.epochs,
          lr=args.lr,
          n_mfs=args.n_mfs,
          seed=args.seed)
