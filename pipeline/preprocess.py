"""
Stage 3 — Feature scaling, temporal reshape, and delta feature computation.

Transform order (always applied in this sequence):
  1. scale_features   : StandardScaler on raw 160-dim flat vectors
  2. reshape_temporal : (N, 160) → (N, 20, 8)
  3. compute_deltas   : compute Δ across timesteps → (N, 20, 8)
  4. concatenate      : [absolute | delta] → (N, 20, 16)

The scaler is fit on the training split only and applied to val/test
to prevent data leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pipeline import config


def scale_features(
    X: np.ndarray,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, StandardScaler]:
    """Fit (if scaler is None) or apply a StandardScaler to X.

    Args:
        X: shape (N, 160) float array of raw feature values
        scaler: pre-fit scaler for val/test; None to fit on X

    Returns:
        (X_scaled, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"[preprocess] Scaler fit on {X.shape[0]:,} samples")
    else:
        X_scaled = scaler.transform(X)
        print(f"[preprocess] Scaler applied to {X.shape[0]:,} samples")
    return X_scaled.astype(np.float32), scaler


def reshape_temporal(X_scaled: np.ndarray) -> np.ndarray:
    """Reshape flat (N, 160) into temporal (N, 20, 8).

    Column order in config.FEATURE_COLS guarantees:
      axis-1 = timestep h0..h19, axis-2 = 8 variables per step.
    """
    N = X_scaled.shape[0]
    X_temporal = X_scaled.reshape(N, config.N_TIMESTEPS, config.N_FEATURES_PER_STEP)
    # Reverse temporal axis: CSV stores h0=most recent, h19=oldest.
    # Chronological order (oldest→newest) is required so LSTM hidden state at
    # the final step represents the current mechanical state, and deltas
    # (np.diff) give positive values for rising quantities like separN.
    X_temporal = X_temporal[:, ::-1, :]  # index 0=h19 (oldest), index 19=h0 (newest)
    print(f"[preprocess] Reshaped to {X_temporal.shape} (chronological: idx0=h19 oldest, idx19=h0 newest)")
    return X_temporal


def compute_deltas(X_temporal: np.ndarray) -> np.ndarray:
    """Compute per-step deltas: Δvar = var[t] - var[t-1].

    Step 0 is padded with zeros (no prior state).

    Args:
        X_temporal: shape (N, 20, 8)

    Returns:
        deltas: shape (N, 20, 8)  — zero-padded at axis=1 index 0
    """
    diff = np.diff(X_temporal, axis=1)           # (N, 19, 8)
    pad = np.zeros((X_temporal.shape[0], 1, config.N_FEATURES_PER_STEP), dtype=np.float32)
    deltas = np.concatenate([pad, diff], axis=1)  # (N, 20, 8)
    return deltas.astype(np.float32)


def build_features(
    X_raw: np.ndarray,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, StandardScaler]:
    """Full preprocessing pipeline: scale → reshape → delta → concatenate.

    Args:
        X_raw: shape (N, 160) raw feature values from CSV
        scaler: None to fit (train), pre-fit scaler for val/test

    Returns:
        (X_out, scaler)
        X_out shape: (N, 20, 16)  — 8 absolute + 8 delta per timestep
    """
    X_scaled, scaler = scale_features(X_raw, scaler)
    X_temporal = reshape_temporal(X_scaled)
    X_deltas = compute_deltas(X_temporal)
    X_out = np.concatenate([X_temporal, X_deltas], axis=2)  # (N, 20, 16)
    print(f"[preprocess] Final feature tensor: {X_out.shape}  (N, timesteps, abs+delta)")
    return X_out, scaler


def extract_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split a DataFrame into raw X array and y label array."""
    X = df[config.FEATURE_COLS].values.astype(np.float32)
    y = df[config.LABEL_COL].values.astype(np.float32)
    return X, y
