"""
src/model/evaluate_comprehensive.py
-----------------------------------
Generates the comprehensive academic evaluation artifacts for the project:
  1. Multi-Seed Stability Analysis:
     Runs 5 different random seeds with subject-held-out splits and computes
     mean +/- std for Accuracy, Precision, Recall, F1, and ROC-AUC.
  2. Deception Effect Sensitivity Sweep:
     Evaluates how the ANFIS pipeline performs as the injected deception
     signal varies from weak (0.1) to strong (1.0).
  3. Saves results/comprehensive_evaluation.json and generates publication-grade
     evaluation summary figures.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from model.anfis_model import ANFIS
from model.train import (
    subject_split, MinMaxScaler, compute_metrics,
    find_best_threshold, FEATURE_NAMES, MF_LABELS
)
from synthetic.generate import generate_dataset

RESULTS_DIR = os.path.join(PROJECT_ROOT, "..", "results") if os.path.exists(os.path.join(PROJECT_ROOT, "..", "results")) else os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def evaluate_multi_seed(data_path: str, seeds=[42, 101, 7, 999, 123], epochs=100, lr=0.005, n_mfs=2):
    """
    Evaluates ANFIS across 5 random seeds to show statistical stability.
    """
    print("\n" + "=" * 60)
    print(f"1. RUNNING MULTI-SEED STABILITY EVALUATION ({len(seeds)} Seeds)")
    print("=" * 60)

    df = pd.read_csv(data_path)
    metrics_list = []

    for seed in seeds:
        print(f"\n---> Running Seed {seed}...")
        train_df, val_df, test_df = subject_split(df, seed=seed)

        def to_XY(split_df):
            X = split_df[FEATURE_NAMES].values.astype(np.float64)
            y = split_df["label"].values.astype(np.float64)
            return X, y

        X_tr, y_tr = to_XY(train_df)
        X_va, y_va = to_XY(val_df)
        X_te, y_te = to_XY(test_df)

        scaler = MinMaxScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        X_te = scaler.transform(X_te)

        anfis = ANFIS(n_inputs=len(FEATURE_NAMES), n_mfs=n_mfs, lr=lr, epochs=epochs)
        anfis._init_params(X_tr)
        anfis.fit(X_tr, y_tr, X_va, y_va, verbose=False)

        val_probs = anfis.predict_proba(X_va)
        threshold = find_best_threshold(y_va, val_probs)

        test_probs = anfis.predict_proba(X_te)
        test_preds = (test_probs >= threshold).astype(int)

        m = compute_metrics(y_te, test_preds, test_probs, f"Test_Seed_{seed}")
        m["seed"] = seed
        m["threshold"] = threshold
        metrics_list.append(m)
        print(f"Seed {seed} -> Acc: {m['accuracy']:.3f}, F1: {m['f1']:.3f}, AUC: {m['roc_auc']:.3f}")

    # Compute mean and standard deviation
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    summary = {}
    for k in keys:
        vals = [item[k] for item in metrics_list]
        summary[k] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "values": vals
        }

    print("\n" + "-" * 50)
    print("MULTI-SEED STATISTICAL SUMMARY (Mean +/- Std):")
    for k, v in summary.items():
        print(f"  {k:12s}: {v['mean']:.4f} +/- {v['std']:.4f}")
    print("-" * 50)

    return summary, metrics_list


def evaluate_sensitivity_sweep(n_users=10, sessions=100, effects=[0.1, 0.3, 0.5, 0.8, 1.0]):
    """
    Evaluates ANFIS performance against varying deception effect magnitudes.
    Shows the cognitive load boundary where hesitation becomes readily detectable.
    """
    print("\n" + "=" * 60)
    print("2. RUNNING DECEPTION EFFECT SENSITIVITY SWEEP")
    print("=" * 60)

    sweep_results = []
    tmp_dir = os.path.join(RESULTS_DIR, "tmp_sweep")
    os.makedirs(tmp_dir, exist_ok=True)

    for eff in effects:
        print(f"\n---> Testing Deception Effect Size = {eff}...")
        sweep_csv = os.path.join(tmp_dir, f"sweep_eff_{eff}.csv")
        df_eff = generate_dataset(n_users=n_users, sessions=sessions, effect_size=eff, seed=42, output_dir=tmp_dir)

        train_df, val_df, test_df = subject_split(df_eff, seed=42)

        def to_XY(split_df):
            X = split_df[FEATURE_NAMES].values.astype(np.float64)
            y = split_df["label"].values.astype(np.float64)
            return X, y

        X_tr, y_tr = to_XY(train_df)
        X_va, y_va = to_XY(val_df)
        X_te, y_te = to_XY(test_df)

        scaler = MinMaxScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        X_te = scaler.transform(X_te)

        anfis = ANFIS(n_inputs=len(FEATURE_NAMES), n_mfs=2, lr=0.005, epochs=100)
        anfis._init_params(X_tr)
        anfis.fit(X_tr, y_tr, X_va, y_va, verbose=False)

        val_probs = anfis.predict_proba(X_va)
        threshold = find_best_threshold(y_va, val_probs)

        test_probs = anfis.predict_proba(X_te)
        test_preds = (test_probs >= threshold).astype(int)

        m = compute_metrics(y_te, test_preds, test_probs, f"Effect_{eff}")
        m["effect_size"] = eff
        sweep_results.append(m)
        print(f"Effect {eff} -> Accuracy: {m['accuracy']:.3f}, F1: {m['f1']:.3f}, AUC: {m['roc_auc']:.3f}")

    # Plot sensitivity curve
    plt.figure(figsize=(7, 4.5))
    eff_vals = [r["effect_size"] for r in sweep_results]
    f1_vals = [r["f1"] for r in sweep_results]
    auc_vals = [r["roc_auc"] for r in sweep_results]
    acc_vals = [r["accuracy"] for r in sweep_results]

    plt.plot(eff_vals, f1_vals, "o-", label="F1 Score", color="#DD8452", lw=2)
    plt.plot(eff_vals, auc_vals, "s-", label="ROC-AUC", color="#55A868", lw=2)
    plt.plot(eff_vals, acc_vals, "^--", label="Accuracy", color="#4C72B0", lw=2)

    plt.title("ANFIS Detection Sensitivity vs. Deception Effect Magnitude")
    plt.xlabel("Injected Deception Effect Size (Cognitive Load Multiplier)")
    plt.ylabel("Test Set Metric")
    plt.ylim(0.4, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, "sensitivity_sweep.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n[plot] Sensitivity sweep plot saved to: {fig_path}")

    return sweep_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive multi-seed & sensitivity evaluation.")
    parser.add_argument("--data", default="data/processed/features.csv")
    args = parser.parse_args()

    # 1. Multi-seed stability
    multi_seed_summary, raw_seed_metrics = evaluate_multi_seed(args.data)

    # 2. Effect size sensitivity sweep
    sensitivity_results = evaluate_sensitivity_sweep()

    # 3. Save comprehensive JSON
    out_file = os.path.join(RESULTS_DIR, "comprehensive_evaluation.json")
    with open(out_file, "w") as f:
        json.dump({
            "multi_seed_summary": multi_seed_summary,
            "individual_seeds": raw_seed_metrics,
            "sensitivity_sweep": sensitivity_results
        }, f, indent=2)

    print(f"\n[complete] All comprehensive evaluation artifacts saved to: {out_file}")
