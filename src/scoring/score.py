"""
src/scoring/score.py
--------------------
Scoring and inference engine for the Neuro-Fuzzy Deception Detection system.

Loads the trained ANFIS model and fitted feature scaler, takes a single
keystroke feature vector (or dictionary of measurements), and outputs:
  - Continuous deception score: [0.0, 1.0]
  - Classification label: "TRUTHFUL" or "DECEPTIVE"
  - Confidence rating: "LOW", "MEDIUM", or "HIGH"
  - Rule-level explanation: which fuzzy rules contributed most to the score

Usage:
  python src/scoring/score.py --latency 1250 --dwell 95 --flight 190 --pauses 3 --backspaces 0.08
"""

import os
import json
import argparse
import numpy as np
import sys

# Ensure project root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from model.anfis_model import ANFIS
from model.train import MinMaxScaler, FEATURE_NAMES

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "..", "results", "anfis_model.npz")
DEFAULT_SCALER_PATH = os.path.join(PROJECT_ROOT, "..", "results", "scaler.npz")
DEFAULT_METRICS_PATH = os.path.join(PROJECT_ROOT, "..", "results", "anfis_metrics.json")

# Fallback paths inside results directory directly
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results") if os.path.exists(os.path.join(PROJECT_ROOT, "results")) else os.path.join(os.path.dirname(PROJECT_ROOT), "results")


class DeceptionScorer:
    """
    Evaluates keystroke hesitation features using the trained ANFIS model.
    """
    def __init__(self,
                 model_path: str = None,
                 scaler_path: str = None,
                 metrics_path: str = None):

        # Resolve paths
        self.model_path = model_path or os.path.join(RESULTS_DIR, "anfis_model.npz")
        self.scaler_path = scaler_path or os.path.join(RESULTS_DIR, "scaler.npz")
        self.metrics_path = metrics_path or os.path.join(RESULTS_DIR, "anfis_metrics.json")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler file not found at: {self.scaler_path}")

        # Load ANFIS and scaler
        self.model = ANFIS.load(self.model_path)
        self.scaler = MinMaxScaler.load(self.scaler_path)

        # Load tuned decision threshold (default to 0.10 if not present)
        self.threshold = 0.10
        if os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, "r") as f:
                    metrics_data = json.load(f)
                    self.threshold = float(metrics_data.get("threshold", 0.10))
            except Exception:
                pass

    def score_features(self,
                       initial_latency: float,
                       mean_dwell: float,
                       mean_flight: float,
                       pause_count: float,
                       backspace_ratio: float) -> dict:
        """
        Evaluate a single session given its 5 keystroke measurements.

        Parameters
        ----------
        initial_latency : ms from prompt display to first keydown
        mean_dwell      : mean key hold duration (ms)
        mean_flight     : mean transition time between keys (ms)
        pause_count     : count of hesitations > 1000 ms
        backspace_ratio : backspaces / total keystrokes (0.0 to 1.0)

        Returns
        -------
        dict with keys:
          - score: float (0.0 to 1.0)
          - label: "DECEPTIVE" or "TRUTHFUL"
          - threshold: float
          - confidence: "LOW", "MEDIUM", "HIGH"
          - top_rules: list of explaining rules
        """
        raw_vec = np.array([[initial_latency, mean_dwell, mean_flight, pause_count, backspace_ratio]], dtype=np.float64)
        norm_vec = self.scaler.transform(raw_vec)

        # Run ANFIS forward pass
        y_prob = float(self.model.predict_proba(norm_vec)[0])
        is_deceptive = (y_prob >= self.threshold)
        label = "DECEPTIVE" if is_deceptive else "TRUTHFUL"

        # Confidence calculation based on distance from the threshold
        distance = abs(y_prob - self.threshold)
        if distance > 0.35:
            confidence = "HIGH"
        elif distance > 0.15:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # Determine dominant active rules
        _, _, w_norm, _, _ = self.model._forward(norm_vec)
        top_rule_indices = np.argsort(w_norm[0])[::-1][:3]
        mf_labels = ["LOW", "MEDIUM"][:self.model.n_mfs]

        active_rules = []
        for idx in top_rule_indices:
            weight = float(w_norm[0, idx])
            if weight > 0.01:
                combo = self.model.rule_index[idx]
                conditions = [f"{FEATURE_NAMES[i]} is {mf_labels[k]}" for i, k in enumerate(combo)]
                bias_w = float(self.model.P[idx, -1])
                verdict = "DECEPTIVE tendency" if bias_w >= 0 else "TRUTHFUL tendency"
                active_rules.append({
                    "rule": f"IF {' AND '.join(conditions)}",
                    "activation_weight": round(weight, 4),
                    "tendency": verdict
                })

        return {
            "score": round(y_prob, 4),
            "label": label,
            "threshold": round(self.threshold, 4),
            "confidence": confidence,
            "dominant_rules": active_rules,
            "raw_inputs": {
                "initial_latency": initial_latency,
                "mean_dwell": mean_dwell,
                "mean_flight": mean_flight,
                "pause_count": pause_count,
                "backspace_ratio": backspace_ratio
            }
        }


def format_score_report(result: dict) -> str:
    """Pretty-print formatted evaluation report."""
    border = "=" * 60
    lines = [
        border,
        " NEURO-FUZZY DECEPTION DETECTION REPORT",
        border,
        f" Outcome          : {result['label']}",
        f" Deception Score  : {result['score']:.4f}  (Range 0.0 - 1.0)",
        f" Decision Cutoff  : {result['threshold']:.4f}",
        f" Confidence Level : {result['confidence']}",
        "-" * 60,
        " Keystroke Features Analyzed:",
        f"   - Initial Latency : {result['raw_inputs']['initial_latency']:.1f} ms",
        f"   - Mean Dwell Time : {result['raw_inputs']['mean_dwell']:.1f} ms",
        f"   - Mean Flight Time: {result['raw_inputs']['mean_flight']:.1f} ms",
        f"   - Pause Count     : {result['raw_inputs']['pause_count']:.1f}",
        f"   - Backspace Ratio : {result['raw_inputs']['backspace_ratio']:.3f}",
        "-" * 60,
        " Most Active Fuzzy Inference Rules:"
    ]

    for i, r in enumerate(result["dominant_rules"], 1):
        lines.append(f"   {i}. {r['rule']}")
        lines.append(f"      Strength: {r['activation_weight'] * 100:.1f}% | Effect: {r['tendency']}")

    lines.append(border)
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate keystroke features with ANFIS.")
    parser.add_argument("--latency", type=float, default=1150.0, help="Initial thinking latency (ms)")
    parser.add_argument("--dwell", type=float, default=95.0, help="Mean key dwell time (ms)")
    parser.add_argument("--flight", type=float, default=180.0, help="Mean flight time (ms)")
    parser.add_argument("--pauses", type=float, default=2.5, help="Count of long pauses")
    parser.add_argument("--backspaces", type=float, default=0.08, help="Backspace keystroke ratio (0.0-1.0)")

    args = parser.parse_args()

    scorer = DeceptionScorer()
    res = scorer.score_features(
        initial_latency=args.latency,
        mean_dwell=args.dwell,
        mean_flight=args.flight,
        pause_count=args.pauses,
        backspace_ratio=args.backspaces
    )

    print(format_score_report(res))
