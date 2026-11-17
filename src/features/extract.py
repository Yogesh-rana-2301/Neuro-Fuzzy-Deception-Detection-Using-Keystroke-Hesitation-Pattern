"""
src/features/extract.py
------------------------
Converts raw keystroke event logs into the 5 numeric feature values that
feed into the ANFIS model.

For the mock-data pipeline the synthetic generator already outputs clean
feature vectors, so this module is used when you want to process REAL
keystroke log files (collected via logger.py or the JS logger).

Raw log format expected (CSV)
------------------------------
key          : str   — the key name or character pressed
event        : str   — "down" or "up"
timestamp_ms : float — milliseconds since epoch (or since session start)
session_id   : str   — groups rows that belong to the same typed answer
prompt_shown_ms : float — timestamp when the prompt appeared on screen

Example CSV
-----------
key,event,timestamp_ms,session_id,prompt_shown_ms
a,down,1000.0,S001,900.0
a,up,1092.0,S001,900.0
n,down,1150.0,S001,900.0
n,up,1238.0,S001,900.0
...

Output
------
One row per session with columns:
  session_id, initial_latency, mean_dwell, mean_flight,
  pause_count, backspace_ratio
"""

import os
import argparse
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAUSE_THRESHOLD_MS = 1000.0   # gaps longer than this count as a "pause"
BACKSPACE_KEYS     = {"backspace", "BackSpace", "Back", "\x08"}


# ---------------------------------------------------------------------------
# Core computation helpers
# ---------------------------------------------------------------------------

def _initial_latency(events_df: pd.DataFrame, prompt_shown_ms: float) -> float:
    """
    Time (ms) from when the prompt appeared to the first key-down event.
    Reflects how long the subject thought before starting to type.
    """
    downs = events_df[events_df["event"] == "down"]
    if downs.empty:
        return np.nan
    first_down = downs["timestamp_ms"].min()
    return max(0.0, first_down - prompt_shown_ms)


def _dwell_times(events_df: pd.DataFrame) -> np.ndarray:
    """
    Key-hold durations: for each key-down, find its matching key-up.
    Returns array of dwell times (ms).
    """
    dwells = []
    downs = events_df[events_df["event"] == "down"].copy()
    ups   = events_df[events_df["event"] == "up"].copy()

    # Match by key name, greedily pair each down with the next up of same key
    for _, down_row in downs.iterrows():
        key = down_row["key"]
        t_down = down_row["timestamp_ms"]
        # Find the earliest key-up for this key AFTER this key-down
        matching_ups = ups[(ups["key"] == key) & (ups["timestamp_ms"] > t_down)]
        if not matching_ups.empty:
            t_up = matching_ups["timestamp_ms"].min()
            dwell = t_up - t_down
            if 10 <= dwell <= 500:   # sanity clamp: ignore hardware glitches
                dwells.append(dwell)
    return np.array(dwells) if dwells else np.array([0.0])


def _flight_times(events_df: pd.DataFrame) -> np.ndarray:
    """
    Inter-key gaps: time from key-up[i] to key-down[i+1].
    Consecutive transitions only.
    """
    downs = events_df[events_df["event"] == "down"]["timestamp_ms"].sort_values().values
    ups   = events_df[events_df["event"] == "up"]["timestamp_ms"].sort_values().values

    flights = []
    n = min(len(ups), len(downs))
    for i in range(n - 1):
        gap = downs[i + 1] - ups[i]
        if -50 <= gap <= 1000:   # allow slight overlap; ignore huge pauses here
            flights.append(gap)
    return np.array(flights) if flights else np.array([0.0])


def _pause_count(events_df: pd.DataFrame, threshold_ms: float = PAUSE_THRESHOLD_MS) -> float:
    """
    Count the number of inter-keystroke gaps longer than threshold_ms.
    These are mid-sentence hesitation pauses (thinking moments).
    """
    downs = events_df[events_df["event"] == "down"]["timestamp_ms"].sort_values().values
    if len(downs) < 2:
        return 0.0
    gaps = np.diff(downs)
    return float(np.sum(gaps > threshold_ms))


def _backspace_ratio(events_df: pd.DataFrame) -> float:
    """
    Fraction of all key-down events that were backspace presses.
    High ratio → lots of self-correction → possible cognitive load.
    """
    downs = events_df[events_df["event"] == "down"]
    if downs.empty:
        return 0.0
    total      = len(downs)
    backspaces = downs["key"].isin(BACKSPACE_KEYS).sum()
    return float(backspaces / total)


# ---------------------------------------------------------------------------
# Session-level extractor
# ---------------------------------------------------------------------------

def extract_session(session_df: pd.DataFrame, prompt_shown_ms: float) -> dict:
    """
    Given all keystroke events for a single session, return a dict of the
    5 feature values.

    Parameters
    ----------
    session_df     : DataFrame with columns [key, event, timestamp_ms]
    prompt_shown_ms: float — when the prompt was first shown to the subject

    Returns
    -------
    dict: {initial_latency, mean_dwell, mean_flight, pause_count, backspace_ratio}
    """
    dwells  = _dwell_times(session_df)
    flights = _flight_times(session_df)

    return {
        "initial_latency":  round(_initial_latency(session_df, prompt_shown_ms), 4),
        "mean_dwell":       round(float(np.mean(dwells)),   4),
        "mean_flight":      round(float(np.mean(flights)),  4),
        "pause_count":      round(_pause_count(session_df), 4),
        "backspace_ratio":  round(_backspace_ratio(session_df), 4),
    }


# ---------------------------------------------------------------------------
# File-level extractor
# ---------------------------------------------------------------------------

def extract_from_file(raw_csv_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Read a raw keystroke log CSV, extract features per session, and
    optionally save to a processed CSV.

    Parameters
    ----------
    raw_csv_path : path to the raw log file
    output_path  : if given, write the feature table here

    Returns
    -------
    pd.DataFrame with one row per session
    """
    df = pd.read_csv(raw_csv_path)

    required_cols = {"key", "event", "timestamp_ms", "session_id", "prompt_shown_ms"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Raw log is missing columns: {missing}")

    records = []
    for session_id, grp in df.groupby("session_id"):
        prompt_ms = grp["prompt_shown_ms"].iloc[0]
        feats = extract_session(grp, prompt_ms)
        feats["session_id"] = session_id
        # Carry over label if present
        if "label" in grp.columns:
            feats["label"] = int(grp["label"].iloc[0])
        if "user_id" in grp.columns:
            feats["user_id"] = grp["user_id"].iloc[0]
        records.append(feats)

    out_df = pd.DataFrame(records)

    # Reorder nicely
    leading = [c for c in ["user_id", "session_id", "label"] if c in out_df.columns]
    features = ["initial_latency", "mean_dwell", "mean_flight", "pause_count", "backspace_ratio"]
    out_df = out_df[leading + features]

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out_df.to_csv(output_path, index=False)
        print(f"[extract] Saved features for {len(out_df)} sessions → {output_path}")

    return out_df


# ---------------------------------------------------------------------------
# Passthrough: synthetic data already has features — just validate & copy
# ---------------------------------------------------------------------------

def prepare_synthetic(synthetic_csv: str, output_path: str) -> pd.DataFrame:
    """
    The synthetic generator already outputs feature vectors. This function
    validates the columns and copies to data/processed/ so the rest of the
    pipeline always reads from the same location.
    """
    df = pd.read_csv(synthetic_csv)
    expected = {"user_id", "session_id", "label",
                "initial_latency", "mean_dwell", "mean_flight",
                "pause_count", "backspace_ratio"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Synthetic file missing columns: {missing}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[extract] Copied {len(df)} synthetic sessions → {output_path}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract keystroke features.")
    sub = parser.add_subparsers(dest="mode")

    # Mode 1: process a real raw log
    real_p = sub.add_parser("real", help="Process a real keystroke log CSV")
    real_p.add_argument("--input",  required=True, help="Path to raw log CSV")
    real_p.add_argument("--output", required=True, help="Where to save features CSV")

    # Mode 2: prepare synthetic data for training
    syn_p = sub.add_parser("synthetic", help="Prepare synthetic data for training")
    syn_p.add_argument("--input",  required=True, help="Path to synthetic CSV")
    syn_p.add_argument("--output", default="data/processed/features.csv")

    args = parser.parse_args()

    if args.mode == "real":
        df = extract_from_file(args.input, args.output)
        print(df.head())
    elif args.mode == "synthetic":
        df = prepare_synthetic(args.input, args.output)
        print(df.describe().round(2))
    else:
        parser.print_help()
