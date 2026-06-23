"""
Stage 5 — tf.data.Dataset builders.

Wraps preprocessed numpy arrays into efficient TF input pipelines.
Training datasets include sample weights and shuffling.
Validation / test datasets are unshuffled and unweighted.
"""

from __future__ import annotations

import numpy as np
from pipeline import config

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    tf = None  # type: ignore[assignment]
    _TF_AVAILABLE = False


def _require_tf() -> None:
    if not _TF_AVAILABLE:
        raise ImportError(
            "tensorflow is required for dataset.py. "
            "Install it with: pip install tensorflow"
        )


def make_train_dataset(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
):
    """Build a shuffled, weighted training dataset.

    Args:
        X: shape (N, 20, 16) preprocessed feature tensors
        y: shape (N,) damage labels
        weights: shape (N,) per-sample loss weights
        batch_size: mini-batch size

    Returns:
        tf.data.Dataset yielding (X_batch, y_batch, weight_batch)
    """
    _require_tf()
    ds = tf.data.Dataset.from_tensor_slices((
        X.astype(np.float32),
        y.astype(np.float32),
        weights.astype(np.float32),
    ))
    ds = ds.shuffle(buffer_size=len(y), seed=config.RANDOM_SEED)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    print(f"[dataset] Train dataset: {len(y):,} samples, batch_size={batch_size}")
    return ds


def make_eval_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    name: str = "eval",
):
    """Build an unshuffled, unweighted evaluation dataset (val or test).

    Returns:
        tf.data.Dataset yielding (X_batch, y_batch)
    """
    _require_tf()
    ds = tf.data.Dataset.from_tensor_slices((
        X.astype(np.float32),
        y.astype(np.float32),
    ))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    print(f"[dataset] {name} dataset: {len(y):,} samples, batch_size={batch_size}")
    return ds
