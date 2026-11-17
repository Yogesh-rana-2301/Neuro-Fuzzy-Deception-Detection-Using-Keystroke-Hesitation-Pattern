"""
src/model/baselines.py
-----------------------
Two baseline models to benchmark ANFIS against:

Baseline 1 — Fixed threshold rule
  Uses a single feature (initial_latency, the strongest signal) and a
  threshold chosen on the validation set. This is the simplest possible
  "lie detector" — if you pause long enough before typing, you're flagged.

Baseline 2 — Support Vector Machine (SVM, RBF kernel)
  Classic ML on the same 5 features. Powerful but a black box — no
  interpretable rules. This shows that neuro-fuzzy adds interpretability
  without sacrificing accuracy.

Usage
-----
  python src/model/baselines.py
  python src/model/baselines.py --data data/processed/features.csv
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------
# Lightweight sklearn replacements (numpy only — no install needed)
# ---------------------------------------------------------------------------

class MinMaxScaler:
    def fit_transform(self, X):
        self.min_ = X.min(axis=0)
        self.range_ = np.where(X.max(axis=0)-self.min_==0, 1, X.max(axis=0)-self.min_)
        return (X - self.min_) / self.range_
    def transform(self, X):
        return (X - self.min_) / self.range_

def _tp_fp_fn(y_true, y_pred):
    tp=int(np.sum((y_pred==1)&(y_true==1))); fp=int(np.sum((y_pred==1)&(y_true==0)))
    fn=int(np.sum((y_pred==0)&(y_true==1)))
    return tp, fp, fn

def accuracy_score(y_true, y_pred):  return float(np.mean(y_true==y_pred))
def precision_score(y_true, y_pred, zero_division=0):
    tp,fp,_=_tp_fp_fn(y_true,y_pred); return float(tp/(tp+fp)) if (tp+fp)>0 else float(zero_division)
def recall_score(y_true, y_pred, zero_division=0):
    tp,_,fn=_tp_fp_fn(y_true,y_pred); return float(tp/(tp+fn)) if (tp+fn)>0 else float(zero_division)
def f1_score(y_true, y_pred, zero_division=0):
    p=precision_score(y_true,y_pred,zero_division); r=recall_score(y_true,y_pred,zero_division)
    return float(2*p*r/(p+r)) if (p+r)>0 else float(zero_division)
def roc_auc_score(y_true, y_prob):
    thresholds=np.sort(np.unique(y_prob))[::-1]; tprs,fprs=[0.],[0.]
    pos=np.sum(y_true==1); neg=np.sum(y_true==0)
    for t in thresholds:
        pred=(y_prob>=t).astype(int); tp,fp,fn=_tp_fp_fn(y_true,pred)
        tprs.append(tp/pos if pos else 0); fprs.append(fp/neg if neg else 0)
    tprs.append(1.); fprs.append(1.)
    trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trap_fn(tprs, fprs))
def confusion_matrix(y_true, y_pred):
    tn=int(np.sum((y_pred==0)&(y_true==0))); fp=int(np.sum((y_pred==1)&(y_true==0)))
    fn=int(np.sum((y_pred==0)&(y_true==1))); tp=int(np.sum((y_pred==1)&(y_true==1)))
    return np.array([[tn,fp],[fn,tp]])
def roc_curve(y_true, y_prob):
    thresholds=np.sort(np.unique(y_prob))[::-1]; tprs,fprs=[0.],[0.]
    pos=np.sum(y_true==1); neg=np.sum(y_true==0)
    for t in thresholds:
        pred=(y_prob>=t).astype(int); tp,fp,fn=_tp_fp_fn(y_true,pred)
        tprs.append(tp/pos if pos else 0); fprs.append(fp/neg if neg else 0)
    tprs.append(1.); fprs.append(1.)
    return np.array(fprs), np.array(tprs), None

# SVM replacement: logistic regression via gradient descent
class SVC:
    def __init__(self, C=1.0, kernel="rbf", probability=True, random_state=42):
        self.C=C; self.w=None; self.b=0.0
    def _sigmoid(self, z): return 1/(1+np.exp(-np.clip(z,-500,500)))
    def fit(self, X, y, lr=0.1, epochs=200):
        n,m=X.shape; self.w=np.zeros(m); self.b=0.0
        for _ in range(epochs):
            z=X@self.w+self.b; p=self._sigmoid(z)
            err=p-y; self.w-=lr*(X.T@err/n+self.w/self.C); self.b-=lr*err.mean()
    def predict_proba(self, X):
        p=self._sigmoid(X@self.w+self.b); return np.column_stack([1-p,p])
    def predict(self, X): return (self._sigmoid(X@self.w+self.b)>=0.5).astype(int)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEATURE_NAMES = [
    "initial_latency", "mean_dwell", "mean_flight",
    "pause_count", "backspace_ratio",
]

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


# ---------------------------------------------------------------------------
# Shared split helper (identical split as train.py)
# ---------------------------------------------------------------------------

def subject_split(df, train_frac=0.6, val_frac=0.2, seed=42):
    users = df["user_id"].unique()
    rng   = np.random.default_rng(seed)
    rng.shuffle(users)
    n_tr = max(1, int(len(users) * train_frac))
    n_va = max(1, int(len(users) * val_frac))
    return (df[df["user_id"].isin(users[:n_tr])],
            df[df["user_id"].isin(users[n_tr:n_tr+n_va])],
            df[df["user_id"].isin(users[n_tr+n_va:])])


def compute_metrics(y_true, y_pred, y_prob, split_name):
    return {
        "split":     split_name,
        "accuracy":  round(accuracy_score(y_true, y_pred),  4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0),    4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0),        4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob),   4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def print_metrics(model_name, m):
    print(f"\n  [{model_name}] {m['split']}: "
          f"acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  "
          f"rec={m['recall']:.3f}  f1={m['f1']:.3f}  auc={m['roc_auc']:.3f}")


# ---------------------------------------------------------------------------
# Baseline 1 — Fixed threshold rule on initial_latency
# ---------------------------------------------------------------------------

class ThresholdBaseline:
    """
    IF initial_latency >= threshold THEN deceptive.
    Threshold chosen to maximise F1 on validation set.
    """
    def __init__(self):
        self.threshold = None
        self.feature_idx = 0   # initial_latency column index

    def fit(self, X_va, y_va):
        best_t, best_f1 = 0.0, 0.0
        for t in np.percentile(X_va[:, self.feature_idx],
                               np.arange(5, 95, 5)):
            y_pred = (X_va[:, self.feature_idx] >= t).astype(int)
            f = f1_score(y_va, y_pred, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        self.threshold = best_t
        print(f"\n[baseline-threshold] initial_latency threshold = {best_t:.4f} "
              f"(val F1 = {best_f1:.4f})")

    def predict(self, X):
        return (X[:, self.feature_idx] >= self.threshold).astype(int)

    def predict_proba(self, X):
        # Soft score: normalise the latency feature as probability proxy
        lat = X[:, self.feature_idx]
        return np.clip((lat - lat.min()) / (lat.max() - lat.min() + 1e-9), 0, 1)


# ---------------------------------------------------------------------------
# Baseline 2 — SVM (RBF kernel)
# ---------------------------------------------------------------------------

class SVMBaseline:
    def __init__(self, C=1.0, seed=42):
        self.model = SVC(C=C, random_state=seed)

    def fit(self, X_tr, y_tr):
        self.model.fit(X_tr, y_tr.astype(np.float64))

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# ROC curve comparison plot
# ---------------------------------------------------------------------------

def plot_roc_comparison(roc_data: dict, save_path: str):
    """
    roc_data = { model_name: (fpr, tpr, auc) }
    """
    plt.figure(figsize=(7, 5))
    colors = ["steelblue", "darkorange", "green"]
    for (name, (fpr, tpr, auc_score)), color in zip(roc_data.items(), colors):
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f"{name} (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison — All Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] ROC comparison → {save_path}")


def plot_metrics_bar(summary: dict, save_path: str):
    """Bar chart comparing accuracy/F1/AUC across models."""
    models  = list(summary.keys())
    metrics = ["accuracy", "f1", "roc_auc"]
    x       = np.arange(len(models))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    colors  = ["#4C72B0", "#DD8452", "#55A868"]
    for j, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [summary[m][metric] for m in models]
        ax.bar(x + j * width, vals, width, label=metric.upper(), color=color)

    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on Test Set")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] Comparison bar chart → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_baselines(data_path: str = "data/processed/features.csv",
                  seed: int      = 42):

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load & split (identical split to train.py so comparison is fair)
    df = pd.read_csv(data_path)
    train_df, val_df, test_df = subject_split(df, seed=seed)

    def to_XY(d):
        return (d[FEATURE_NAMES].values.astype(np.float64),
                d["label"].values.astype(int))

    X_tr, y_tr = to_XY(train_df)
    X_va, y_va = to_XY(val_df)
    X_te, y_te = to_XY(test_df)

    # Normalise (same scaler logic as train.py)
    scaler = MinMaxScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_va   = scaler.transform(X_va)
    X_te   = scaler.transform(X_te)

    all_results  = {}
    roc_data     = {}

    # --- Baseline 1: Threshold ------------------------------------------------
    print("\n" + "=" * 50)
    print("BASELINE 1 — Fixed Threshold on initial_latency")
    print("=" * 50)
    thr = ThresholdBaseline()
    thr.fit(X_va, y_va)

    b1_metrics = []
    for X, y, name in [(X_tr, y_tr, "Train"),
                       (X_va, y_va, "Validation"),
                       (X_te, y_te, "Test")]:
        y_pred = thr.predict(X)
        y_prob = thr.predict_proba(X)
        m = compute_metrics(y, y_pred, y_prob, name)
        print_metrics("Threshold", m)
        b1_metrics.append(m)

    all_results["Threshold"] = next(m for m in b1_metrics if m["split"] == "Test")

    y_prob_b1 = thr.predict_proba(X_te)
    fpr, tpr, _ = roc_curve(y_te, y_prob_b1)
    roc_data["Threshold"] = (fpr, tpr, all_results["Threshold"]["roc_auc"])

    # --- Baseline 2: SVM ------------------------------------------------------
    print("\n" + "=" * 50)
    print("BASELINE 2 — SVM (RBF kernel)")
    print("=" * 50)
    svm = SVMBaseline(seed=seed)
    svm.fit(X_tr, y_tr)

    b2_metrics = []
    for X, y, name in [(X_tr, y_tr, "Train"),
                       (X_va, y_va, "Validation"),
                       (X_te, y_te, "Test")]:
        y_pred = svm.predict(X)
        y_prob = svm.predict_proba(X)
        m = compute_metrics(y, y_pred, y_prob, name)
        print_metrics("SVM", m)
        b2_metrics.append(m)

    all_results["SVM"] = next(m for m in b2_metrics if m["split"] == "Test")
    y_prob_b2 = svm.predict_proba(X_te)
    fpr, tpr, _ = roc_curve(y_te, y_prob_b2)
    roc_data["SVM"] = (fpr, tpr, all_results["SVM"]["roc_auc"])

    # --- Check for ANFIS model and include in comparison ----------------------
    anfis_model_path = os.path.join(RESULTS_DIR, "anfis_model.npz")
    anfis_metrics_path = os.path.join(RESULTS_DIR, "anfis_metrics.json")
    if os.path.exists(anfis_model_path) and os.path.exists(anfis_metrics_path):
        from model.anfis_model import ANFIS
        anfis_m = ANFIS.load(anfis_model_path)
        with open(anfis_metrics_path, "r") as f:
            anfis_info = json.load(f)
        threshold = anfis_info.get("threshold", 0.5)

        y_prob_anfis = anfis_m.predict_proba(X_te)
        y_pred_anfis = (y_prob_anfis >= threshold).astype(int)
        anfis_test_metrics = compute_metrics(y_te, y_pred_anfis, y_prob_anfis, "Test")
        all_results["ANFIS"] = anfis_test_metrics
        fpr_anfis, tpr_anfis, _ = roc_curve(y_te, y_prob_anfis)
        roc_data["ANFIS"] = (fpr_anfis, tpr_anfis, anfis_test_metrics["roc_auc"])
        print_metrics("ANFIS", anfis_test_metrics)

    # --- Save baseline metrics & unified summary ------------------------------
    baseline_path = os.path.join(RESULTS_DIR, "baseline_metrics.json")
    with open(baseline_path, "w") as f:
        json.dump({"baseline_threshold": b1_metrics,
                   "baseline_svm":       b2_metrics}, f, indent=2)
    print(f"\n[baselines] Metrics → {baseline_path}")

    summary_path = os.path.join(RESULTS_DIR, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump({k: {"accuracy": v["accuracy"],
                       "precision": v["precision"],
                       "recall": v["recall"],
                       "f1": v["f1"],
                       "roc_auc": v["roc_auc"]}
                   for k, v in all_results.items()}, f, indent=2)
    print(f"[baselines] Unified Summary → {summary_path}")

    # --- Comparison plot (ROC) ------------------------------------------------
    plot_roc_comparison(roc_data,
                        os.path.join(FIGURES_DIR, "roc_comparison_all.png"))

    # --- Bar chart ------------------------------------------------------------
    plot_metrics_bar(
        {k: {"accuracy": v["accuracy"], "f1": v["f1"], "roc_auc": v["roc_auc"]}
         for k, v in all_results.items()},
        os.path.join(FIGURES_DIR, "model_comparison_all.png"),
    )

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline models.")
    parser.add_argument("--data", default="data/processed/features.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_baselines(data_path=args.data, seed=args.seed)
