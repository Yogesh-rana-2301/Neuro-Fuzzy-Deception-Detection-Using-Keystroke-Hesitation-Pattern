"""
src/model/anfis_model.py
-------------------------
A pure-NumPy ANFIS (Adaptive Neuro-Fuzzy Inference System) implementation.
Uses Takagi-Sugeno style rules with Gaussian membership functions.
Trained via hybrid learning: least-squares for consequent params,
gradient descent for premise params (MF centers & widths).

Why not just use sanfis?
  sanfis requires PyTorch which needs a network install.
  This self-contained NumPy implementation runs anywhere, is easy to
  inspect during Q&A, and matches the ANFIS theory you'll describe in
  your report exactly.

Architecture (for 5 inputs, 3 MFs each)
----------------------------------------
  Layer 1 — Fuzzification        : Gaussian MF μ(x) for each input × MF
  Layer 2 — Rule firing strength : product of MF outputs per rule
  Layer 3 — Normalised firing    : w̄_i = w_i / Σw_j
  Layer 4 — Consequent           : w̄_i × (p_i·x + q_i)   [Takagi-Sugeno]
  Layer 5 — Aggregation          : Σ(layer 4) → scalar output ∈ [0, 1]

Reference: Jang, J.S.R. (1993). ANFIS: Adaptive-network-based fuzzy
           inference system. IEEE Trans. Systems, Man & Cybernetics.
"""

import numpy as np
from itertools import product as cartesian_product


# ---------------------------------------------------------------------------
# Gaussian membership function helpers
# ---------------------------------------------------------------------------

def gaussian_mf(x: np.ndarray, center: float, sigma: float) -> np.ndarray:
    """μ(x) = exp( -(x - center)² / (2·sigma²) )"""
    return np.exp(-0.5 * ((x - center) / (sigma + 1e-9)) ** 2)


def d_gaussian_dc(x, mu, center, sigma):
    """∂μ/∂center"""
    return mu * (x - center) / (sigma ** 2 + 1e-9)


def d_gaussian_ds(x, mu, center, sigma):
    """∂μ/∂sigma"""
    return mu * ((x - center) ** 2) / (sigma ** 3 + 1e-9)


# ---------------------------------------------------------------------------
# ANFIS class
# ---------------------------------------------------------------------------

class ANFIS:
    """
    Adaptive Neuro-Fuzzy Inference System (Takagi-Sugeno, 1st-order).

    Parameters
    ----------
    n_inputs : int   — number of input features (default 5)
    n_mfs    : int   — membership functions per input (default 3)
    lr       : float — learning rate for premise parameters
    epochs   : int   — training epochs
    """

    def __init__(self, n_inputs: int = 5, n_mfs: int = 2,
                 lr: float = 0.005, epochs: int = 150,
                 l2: float = 1e-3):
        self.n_inputs = n_inputs
        self.n_mfs    = n_mfs
        self.lr       = lr
        self.epochs   = epochs
        self.l2       = l2    # L2 regularisation for consequent params

        # Total rules = n_mfs ^ n_inputs  (e.g. 3^5 = 243)
        self.n_rules  = n_mfs ** n_inputs

        # --- Premise parameters (Layer 1) ---
        # centers[i, k] = center of MF k for input i
        # sigmas[i, k]  = width of MF k for input i
        # Initialised uniformly across [0, 1] after feature normalisation
        self.centers = np.zeros((n_inputs, n_mfs))
        self.sigmas  = np.ones((n_inputs, n_mfs)) * 0.5

        # --- Consequent parameters (Layer 4) ---
        # Each rule r has a linear function: p[r,0]*x0 + ... + p[r,n]*xn + p[r,n+1]
        # Shape: (n_rules, n_inputs + 1)   (+1 for the bias term)
        self.P = np.zeros((self.n_rules, n_inputs + 1))

        # Pre-compute which MF combination each rule corresponds to
        # rule_index[r] = (k0, k1, ..., k_{n_inputs-1})
        self.rule_index = list(cartesian_product(range(n_mfs), repeat=n_inputs))

        # Training history
        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []

        self._fitted = False

    # -----------------------------------------------------------------------
    # Initialise premise params from data range
    # -----------------------------------------------------------------------

    def _init_params(self, X: np.ndarray) -> None:
        """
        Spread MF centers evenly across each feature's value range.
        Width = (range) / (n_mfs + 1)  — a reasonable starting sigma.
        """
        for i in range(self.n_inputs):
            lo, hi = X[:, i].min(), X[:, i].max()
            self.centers[i] = np.linspace(lo, hi, self.n_mfs)
            self.sigmas[i]  = np.full(self.n_mfs,
                                      max((hi - lo) / (self.n_mfs + 1), 1e-3))

    # -----------------------------------------------------------------------
    # Forward pass — returns (output, firing_strengths, norm_strengths, phi)
    # -----------------------------------------------------------------------

    def _forward(self, X: np.ndarray):
        """
        X : (N, n_inputs)  — normalised feature matrix

        Returns
        -------
        y_pred : (N,)   — model output
        W      : (N, n_rules) — raw firing strengths
        W_norm : (N, n_rules) — normalised firing strengths
        Phi    : (N, n_rules, n_inputs+1) — consequent regressors
        """
        N = X.shape[0]

        # Layer 1: membership degrees  μ[i, k] for each sample
        # MU shape: (N, n_inputs, n_mfs)
        MU = np.zeros((N, self.n_inputs, self.n_mfs))
        for i in range(self.n_inputs):
            for k in range(self.n_mfs):
                MU[:, i, k] = gaussian_mf(X[:, i],
                                           self.centers[i, k],
                                           self.sigmas[i, k])

        # Layer 2: rule firing strength = product of relevant MF outputs
        # W shape: (N, n_rules)
        W = np.ones((N, self.n_rules))
        for r, combo in enumerate(self.rule_index):
            for i, k in enumerate(combo):
                W[:, r] *= MU[:, i, k]

        # Layer 3: normalised firing strengths
        W_sum  = W.sum(axis=1, keepdims=True) + 1e-9
        W_norm = W / W_sum            # (N, n_rules)

        # Layer 4: consequent — build regressor matrix Phi
        # Phi[n, r, :] = [x0, x1, ..., x_{m-1}, 1]  (same for all r)
        Phi = np.concatenate([X, np.ones((N, 1))], axis=1)   # (N, n_inputs+1)
        Phi = np.tile(Phi[:, None, :], (1, self.n_rules, 1))  # (N, n_rules, n_inputs+1)

        # f_r = P[r] · x  for each rule
        # f shape: (N, n_rules)
        f = np.einsum("nri,ri->nr", Phi, self.P)

        # Layer 5: weighted sum → scalar output per sample
        y_pred = np.sum(W_norm * f, axis=1)   # (N,)

        return y_pred, W, W_norm, Phi, MU

    # -----------------------------------------------------------------------
    # Consequent update — Weighted Least Squares (closed-form, fast)
    # -----------------------------------------------------------------------

    def _update_consequents(self, X: np.ndarray, y: np.ndarray,
                            W_norm: np.ndarray) -> None:
        """
        Solve for P using weighted least squares:
          P* = argmin ||W_norm·Phi·P - y||²
        Stacked form: A·p_vec = b_vec
        """
        N = X.shape[0]
        Phi = np.concatenate([X, np.ones((N, 1))], axis=1)  # (N, m+1)

        # Build A: (N, n_rules * (n_inputs+1))
        m = self.n_inputs + 1
        A = np.zeros((N, self.n_rules * m))
        for r in range(self.n_rules):
            A[:, r*m:(r+1)*m] = W_norm[:, r:r+1] * Phi

        # Solve using ridge regression (L2) — prevents overfit in consequent layer
        I      = np.eye(A.shape[1])
        p_vec  = np.linalg.lstsq(A.T @ A + self.l2 * I, A.T @ y, rcond=None)[0]
        self.P = p_vec.reshape(self.n_rules, m)

    # -----------------------------------------------------------------------
    # Premise update — Gradient Descent
    # -----------------------------------------------------------------------

    def _update_premises(self, X: np.ndarray, y: np.ndarray,
                         y_pred: np.ndarray, W: np.ndarray,
                         W_norm: np.ndarray, MU: np.ndarray) -> None:
        """
        Backpropagate error through layers 3→2→1 to update centers & sigmas.
        Uses mean-squared-error loss.
        """
        N = X.shape[0]
        error = y_pred - y                    # (N,)
        W_sum = W.sum(axis=1) + 1e-9          # (N,)

        # Consequent output per rule
        f = np.einsum("ri,ni->nr", self.P[:, :-1], X) + self.P[:, -1]  # (N, n_rules)

        # ∂Loss/∂y_pred · ∂y_pred/∂W_norm[r]
        # = error * (f_r - y_pred)   (from quotient rule on normalisation)
        d_wrt_w_norm = error[:, None] * (f - y_pred[:, None])           # (N, n_rules)

        # ∂W_norm[r]/∂W[r] = (W_sum - W[r]) / W_sum²
        d_wrt_w = d_wrt_w_norm / W_sum[:, None]                         # (N, n_rules)

        # For each input i and MF k, accumulate gradient
        dc = np.zeros_like(self.centers)   # (n_inputs, n_mfs)
        ds = np.zeros_like(self.sigmas)

        for r, combo in enumerate(self.rule_index):
            for i, k in enumerate(combo):
                # ∂W[r]/∂MU[i,k] = W[r] / MU[i,k]  (product rule)
                safe_mu = MU[:, i, k] + 1e-9
                d_wrt_mu = d_wrt_w[:, r] * (W[:, r] / safe_mu)          # (N,)

                mu_ik = MU[:, i, k]
                c_ik  = self.centers[i, k]
                s_ik  = self.sigmas[i, k]

                dc[i, k] += np.mean(d_wrt_mu * d_gaussian_dc(X[:, i], mu_ik, c_ik, s_ik))
                ds[i, k] += np.mean(d_wrt_mu * d_gaussian_ds(X[:, i], mu_ik, c_ik, s_ik))

        self.centers -= self.lr * dc
        self.sigmas  -= self.lr * ds
        # Keep sigmas positive
        self.sigmas   = np.maximum(self.sigmas, 1e-4)

    # -----------------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------------

    @staticmethod
    def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            verbose: bool = True) -> "ANFIS":
        """
        Train the ANFIS model.

        Parameters
        ----------
        X_train, y_train : training data (arrays, already normalised)
        X_val,   y_val   : optional validation data
        verbose          : print loss every 10 epochs

        Returns
        -------
        self (for chaining)
        """
        self._init_params(X_train)

        for epoch in range(1, self.epochs + 1):
            # Forward
            y_pred, W, W_norm, _, MU = self._forward(X_train)

            # Consequent update (least squares — fast & stable)
            self._update_consequents(X_train, y_train, W_norm)

            # Re-forward with updated consequents
            y_pred, W, W_norm, _, MU = self._forward(X_train)

            # Premise update (gradient descent)
            self._update_premises(X_train, y_train, y_pred, W, W_norm, MU)

            train_loss = self._mse(y_train, y_pred)
            self.train_losses.append(train_loss)

            if X_val is not None:
                y_val_pred, *_ = self._forward(X_val)
                val_loss = self._mse(y_val, y_val_pred)
                self.val_losses.append(val_loss)

            if verbose and (epoch % 10 == 0 or epoch == 1):
                val_str = f"  val_loss={self.val_losses[-1]:.5f}" if X_val is not None else ""
                print(f"  Epoch {epoch:4d}/{self.epochs}  train_loss={train_loss:.5f}{val_str}")

        self._fitted = True
        return self

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return raw ANFIS output clipped to [0, 1] — deception probability."""
        y_pred, *_ = self._forward(X)
        return np.clip(y_pred, 0.0, 1.0)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary labels (1 = deceptive) using threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    # -----------------------------------------------------------------------
    # Rule export — the interpretability deliverable
    # -----------------------------------------------------------------------

    def export_rules(self, feature_names: list[str],
                     mf_labels: list[str] = None) -> str:
        """
        Return a human-readable string of the top-10 most-fired rules
        with their learned Takagi-Sugeno consequent weights.

        Parameters
        ----------
        feature_names : list of feature names, length = n_inputs
        mf_labels     : labels for MFs, default ["LOW", "MEDIUM", "HIGH"]
        """
        if mf_labels is None:
            mf_labels = ["LOW", "MEDIUM", "HIGH"][:self.n_mfs]

        lines = ["=" * 60,
                 "LEARNED ANFIS FUZZY RULES (top 10 by consequent weight)",
                 "=" * 60]

        # Sort rules by absolute value of their bias consequent term
        rule_weights = np.abs(self.P[:, -1])
        top_rules    = np.argsort(rule_weights)[::-1][:10]

        for rank, r in enumerate(top_rules, 1):
            combo = self.rule_index[r]
            cond  = " AND ".join(
                f"{feature_names[i]} is {mf_labels[k]}"
                for i, k in enumerate(combo)
            )
            bias_w = self.P[r, -1]
            sign   = "HIGH (Deceptive)" if bias_w >= 0.5 else \
                     "LOW (Truthful)"   if bias_w <= -0.5 else \
                     "MODERATE"
            lines.append(f"\nRule {rank:2d}: IF {cond}")
            lines.append(f"         THEN deception_score ≈ {bias_w:+.4f}  [{sign}]")

        lines.append("=" * 60)
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Membership function summary
    # -----------------------------------------------------------------------

    def mf_summary(self, feature_names: list[str],
                   mf_labels: list[str] = None) -> str:
        """Print learned MF centers and widths per feature."""
        if mf_labels is None:
            mf_labels = ["LOW", "MEDIUM", "HIGH"][:self.n_mfs]

        lines = ["\nLearned Membership Function Parameters",
                 "-" * 44]
        for i, feat in enumerate(feature_names):
            lines.append(f"\n  {feat}")
            for k, label in enumerate(mf_labels):
                c = self.centers[i, k]
                s = self.sigmas[i, k]
                lines.append(f"    {label:8s}:  center={c:.4f}  sigma={s:.4f}")
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Model Persistence (Save / Load)
    # -----------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Save ANFIS parameters to a .npz archive."""
        np.savez(
            filepath,
            centers=self.centers,
            sigmas=self.sigmas,
            P=self.P,
            n_inputs=self.n_inputs,
            n_mfs=self.n_mfs,
            lr=self.lr,
            epochs=self.epochs,
            l2=self.l2,
        )

    @classmethod
    def load(cls, filepath: str) -> "ANFIS":
        """Load trained ANFIS model from a .npz archive."""
        data = np.load(filepath)
        model = cls(
            n_inputs=int(data["n_inputs"]),
            n_mfs=int(data["n_mfs"]),
            lr=float(data["lr"]),
            epochs=int(data["epochs"]),
            l2=float(data["l2"]),
        )
        model.centers = data["centers"]
        model.sigmas = data["sigmas"]
        model.P = data["P"]
        model._fitted = True
        return model
