"""
Stage 6 — Export processed arrays and metadata to data/processed/.

Artifacts written:
  train.npz            — X (N,20,16) and y (N,) for training split
  val.npz              — X and y for validation split
  test.npz             — X and y for test split
  weights_train.npy    — per-sample loss weights (training only)
  scaler.json          — StandardScaler mean/scale arrays (C++ reads this)
  pipeline_config.json — metadata: split sizes, bin counts, D_BIN_EDGES, timestamp
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pipeline import config


def save_split(
    out_dir: str,
    name: str,
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    path = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(path, X=X, y=y)
    print(f"[export] Saved {path}  (X={X.shape}, y={y.shape})")


def save_weights(out_dir: str, weights: np.ndarray) -> None:
    path = os.path.join(out_dir, "weights_train.npy")
    np.save(path, weights)
    print(f"[export] Saved {path}  (shape={weights.shape})")


def save_scaler(out_dir: str, scaler: StandardScaler) -> None:
    """Save scaler as JSON so the C++ preprocessor can load it without Python."""
    path = os.path.join(out_dir, "scaler.json")
    payload = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "n_features": int(scaler.n_features_in_),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[export] Saved {path}  ({len(payload['mean'])} features)")


def save_pipeline_config(
    out_dir: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    def bin_counts(df):
        bins = pd.cut(
            df[config.LABEL_COL],
            bins=config.D_BIN_EDGES,
            labels=config.D_BIN_LABELS,
            right=False,
        )
        return {label: int((bins == label).sum()) for label in config.D_BIN_LABELS}

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_timesteps": config.N_TIMESTEPS,
        "n_features_per_step": config.N_FEATURES_PER_STEP,
        "n_features_with_deltas": config.N_FEATURES_PER_STEP * 2,
        "d_bin_edges": config.D_BIN_EDGES,
        "d_bin_labels": config.D_BIN_LABELS,
        "temporal_order": "chronological — axis-1 index 0 = h19 (oldest), index 19 = h0 (most recent)",
        "random_seed": config.RANDOM_SEED,
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "bin_counts": {
            "train": bin_counts(train_df),
            "val": bin_counts(val_df),
            "test": bin_counts(test_df),
        },
    }
    path = os.path.join(out_dir, "pipeline_config.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[export] Saved {path}")


def export_all(
    out_dir: str,
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    weights: np.ndarray,
    scaler: StandardScaler,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    save_split(out_dir, "train", train_X, train_y)
    save_split(out_dir, "val", val_X, val_y)
    save_split(out_dir, "test", test_X, test_y)
    save_weights(out_dir, weights)
    save_scaler(out_dir, scaler)
    save_pipeline_config(out_dir, train_df, val_df, test_df)
    print(f"\n[export] All artifacts written to {out_dir}/")
