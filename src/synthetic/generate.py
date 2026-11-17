"""
src/synthetic/generate.py
--------------------------
Generates a synthetic labeled keystroke-feature dataset for the
Neuro-Fuzzy Deception Detection project.

No real human data is needed. We simulate the well-documented cognitive-load
effect: lying is mentally harder than truth-telling, which shows up as
measurable hesitation in keystroke timing patterns.

Output columns
--------------
user_id            : str  — synthetic user identifier (e.g. "user_01")
session_id         : str  — unique per answer session
label              : int  — 0 = Truthful, 1 = Deceptive/Fabricated
initial_latency    : float — ms from prompt shown to first keypress
mean_dwell         : float — mean key-hold time (ms)
mean_flight        : float — mean gap between key-release and next key-press (ms)
pause_count        : float — number of gaps > 1000 ms within the answer
backspace_ratio    : float — fraction of keystrokes that were backspaces

Usage
-----
  python src/synthetic/generate.py                   # uses defaults
  python src/synthetic/generate.py --effect 0.5      # stronger deception signal
  python src/synthetic/generate.py --effect 0.1      # weaker signal (harder task)
  python src/synthetic/generate.py --seed 99 --n_users 20 --sessions 150
"""

import argparse
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1.  Baseline typing profile (truthful condition)
#     These values are grounded in published keystroke-dynamics literature.
#     Source: Bergadano et al., 2002; Gunetti & Picardi, 2005
# ---------------------------------------------------------------------------

# Log-normal parameters:  actual_value = exp(mu + sigma * Z)
# We store (mean_ms, cv) where cv = coefficient of variation = std/mean.
# Then: sigma = sqrt(log(1 + cv^2)), mu = log(mean) - sigma^2/2

TRUTHFUL_PROFILE = {
    # feature           mean (ms)   CV     ← controls spread / naturalness
    "initial_latency": (800,        0.40),   # thinking before first keystroke
    "mean_dwell":      (90,         0.25),   # key-hold time
    "mean_flight":     (150,        0.35),   # between consecutive keys
    "pause_count":     (1.5,        0.60),   # long pauses during typing
    "backspace_ratio": (0.05,       0.80),   # self-correction fraction
}

# ---------------------------------------------------------------------------
# 2.  Deception effect multipliers
#     Lying inflates each feature by this factor (before user noise).
#     Literature-informed, but we parametrise the global magnitude so we
#     can run sensitivity experiments (small vs. large effect size).
# ---------------------------------------------------------------------------

DECEPTION_MULTIPLIERS = {
    "initial_latency": 1.40,   # +40% — longer thinking time before writing
    "mean_dwell":      1.10,   # +10% — slightly heavier key presses
    "mean_flight":     1.25,   # +25% — more hesitation between keys
    "pause_count":     1.60,   # +60% — many more mid-sentence pauses
    "backspace_ratio": 1.80,   # +80% — far more self-correction / editing
}


# ---------------------------------------------------------------------------
# 3.  Helper: log-normal sampler
# ---------------------------------------------------------------------------

def _lognormal_samples(mean: float, cv: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw n samples from a log-normal distribution with the given
    arithmetic mean and coefficient of variation (std/mean).

    Why log-normal?
      Typing times are strictly positive and right-skewed — a small
      number of very long pauses pulls the tail out. Log-normal is the
      standard choice in the keystroke-dynamics literature.
    """
    sigma2 = np.log(1 + cv ** 2)
    mu     = np.log(mean) - sigma2 / 2
    return rng.lognormal(mean=mu, sigma=np.sqrt(sigma2), size=n)


# ---------------------------------------------------------------------------
# 4.  Per-user baseline offset
#     Each simulated user has their own natural typing speed (some people
#     are fast typists, some slow). Without this, the model could mistake
#     "user identity" for "deception signal."
# ---------------------------------------------------------------------------

def _user_speed_factor(rng: np.random.Generator) -> float:
    """
    Returns a multiplicative speed factor for one user.
    Drawn from N(1.0, 0.20), clipped to [0.5, 2.0].
    A factor of 0.7 means this user types 30% faster than average.
    A factor of 1.4 means this user types 40% slower than average.
    """
    return float(np.clip(rng.normal(loc=1.0, scale=0.20), 0.50, 2.00))


# ---------------------------------------------------------------------------
# 5.  Core: generate one session's feature vector
# ---------------------------------------------------------------------------

def _generate_session(
    user_speed: float,
    label: int,
    effect_size: float,
    rng: np.random.Generator,
) -> dict:
    """
    Simulate one typing session (one answer to one prompt).

    Parameters
    ----------
    user_speed  : per-user baseline multiplier (see _user_speed_factor)
    label       : 0 = truthful, 1 = deceptive
    effect_size : global scaling of deception multipliers [0 .. 1]
                  0.0 → no deception signal (null hypothesis)
                  1.0 → full literature-sized effect
    rng         : numpy Generator for reproducibility

    Returns
    -------
    dict with feature values (floats)
    """
    row = {}
    for feat, (base_mean, cv) in TRUTHFUL_PROFILE.items():
        # Apply per-user speed offset to timing features
        # (pause_count and backspace_ratio are count/ratio, not raw times,
        #  so the speed factor applies differently)
        if feat in ("initial_latency", "mean_dwell", "mean_flight"):
            adjusted_mean = base_mean * user_speed
        else:
            adjusted_mean = base_mean  # counts / ratios unaffected by speed

        # Apply deception multiplier if this is a fabricated session
        if label == 1:
            multiplier = 1.0 + (DECEPTION_MULTIPLIERS[feat] - 1.0) * effect_size
            adjusted_mean *= multiplier

        # Sample from log-normal
        value = float(_lognormal_samples(adjusted_mean, cv, n=1, rng=rng)[0])

        # Hard clip to sensible ranges so we don't get absurd outliers
        clamps = {
            "initial_latency": (100.0,  5000.0),
            "mean_dwell":      (30.0,   400.0),
            "mean_flight":     (30.0,   600.0),
            "pause_count":     (0.0,    15.0),
            "backspace_ratio": (0.0,    0.60),
        }
        lo, hi = clamps[feat]
        row[feat] = round(float(np.clip(value, lo, hi)), 4)

    row["label"] = label
    return row


# ---------------------------------------------------------------------------
# 6.  Main generator
# ---------------------------------------------------------------------------

def generate_dataset(
    n_users: int    = 10,
    sessions: int   = 100,   # sessions PER USER (half truthful, half deceptive)
    effect_size: float = 0.3,
    seed: int       = 42,
    output_dir: str = "data/synthetic",
) -> pd.DataFrame:
    """
    Generate a full synthetic dataset and save it to disk.

    Parameters
    ----------
    n_users     : number of distinct simulated users
    sessions    : number of sessions per user (split 50/50 truth vs. lie)
    effect_size : magnitude of deception signal [0=none .. 1=full]
    seed        : random seed for reproducibility
    output_dir  : directory to save the CSV

    Returns
    -------
    pd.DataFrame with all sessions
    """
    rng = np.random.default_rng(seed)
    records = []

    n_truthful   = sessions // 2
    n_deceptive  = sessions - n_truthful   # handles odd numbers cleanly

    for u in range(n_users):
        user_id    = f"user_{u+1:02d}"
        user_speed = _user_speed_factor(rng)

        # Truthful sessions
        for s in range(n_truthful):
            row = _generate_session(user_speed, label=0, effect_size=effect_size, rng=rng)
            row["user_id"]    = user_id
            row["session_id"] = f"{user_id}_T{s+1:03d}"
            records.append(row)

        # Deceptive sessions
        for s in range(n_deceptive):
            row = _generate_session(user_speed, label=1, effect_size=effect_size, rng=rng)
            row["user_id"]    = user_id
            row["session_id"] = f"{user_id}_D{s+1:03d}"
            records.append(row)

    df = pd.DataFrame(records)

    # Reorder columns for clarity
    df = df[["user_id", "session_id", "label",
             "initial_latency", "mean_dwell", "mean_flight",
             "pause_count", "backspace_ratio"]]

    # Shuffle rows so truthful/deceptive aren't grouped together
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"synthetic_dataset_effect{effect_size}_seed{seed}.csv")
    df.to_csv(output_path, index=False)
    print(f"[generate] Saved {len(df)} sessions → {output_path}")
    print(f"[generate] Label distribution:\n{df['label'].value_counts().to_string()}")
    print(f"[generate] Feature summary:\n{df.drop(columns=['user_id','session_id','label']).describe().round(2).to_string()}")

    return df


# ---------------------------------------------------------------------------
# 7.  Quick sanity check: does the deception group actually differ?
# ---------------------------------------------------------------------------

def print_effect_check(df: pd.DataFrame) -> None:
    """
    Print mean of each feature by label so we can visually confirm
    the deception signal was injected correctly.
    """
    print("\n[sanity check] Per-label feature means:")
    features = ["initial_latency", "mean_dwell", "mean_flight", "pause_count", "backspace_ratio"]
    grouped  = df.groupby("label")[features].mean().round(2)
    grouped.index = grouped.index.map({0: "Truthful (0)", 1: "Deceptive (1)"})
    print(grouped.to_string())
    print()


# ---------------------------------------------------------------------------
# 8.  CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic keystroke deception dataset."
    )
    parser.add_argument("--n_users",    type=int,   default=10,
                        help="Number of simulated users (default: 10)")
    parser.add_argument("--sessions",   type=int,   default=100,
                        help="Sessions per user (default: 100 → 50 truth + 50 lie)")
    parser.add_argument("--effect",     type=float, default=0.3,
                        help="Deception effect size 0..1 (default: 0.3)")
    parser.add_argument("--seed",       type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output_dir", type=str,   default="data/synthetic",
                        help="Where to save the CSV")
    args = parser.parse_args()

    df = generate_dataset(
        n_users    = args.n_users,
        sessions   = args.sessions,
        effect_size= args.effect,
        seed       = args.seed,
        output_dir = args.output_dir,
    )
    print_effect_check(df)
