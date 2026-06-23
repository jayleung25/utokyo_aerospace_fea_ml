"""
Stage 4 — Per-sample loss weights for class-imbalance correction.

The damage distribution is heavily skewed toward D ≈ 0.9–1.0.
Weighting each training sample by the inverse frequency of its D bin
forces the model to treat rare transition-zone samples as equally
important to the abundant high-D samples.

Applied to the TRAINING split only. Val and test are evaluated
unweighted to reflect true population performance.
"""

import numpy as np
from pipeline import config


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Compute inverse-frequency weights per sample, normalized to mean=1.

    Args:
        y: 1-D array of damage values (training labels only)

    Returns:
        weights: float32 array, same length as y, mean ≈ 1.0
    """
    bin_edges = config.D_BIN_EDGES
    n_bins = len(config.D_BIN_LABELS)

    # Assign each sample to a bin index (0..n_bins-1) using the interior edges
    # np.searchsorted on interior edges [0.3, 0.7, 0.9] gives 0-based bin index
    interior_edges = bin_edges[1:-1]  # [0.3, 0.7, 0.9]
    bin_idx = np.searchsorted(interior_edges, y, side='right')
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # Count samples per bin
    bin_counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float32)

    # Inverse frequency: samples in rare bins get higher weight
    total = float(len(y))
    bin_weights = np.where(bin_counts > 0, total / bin_counts, 0.0)

    # Map each sample to its bin weight
    raw_weights = bin_weights[bin_idx].astype(np.float32)

    # Normalize so mean(weights) == 1.0 (keeps effective LR stable)
    weights = raw_weights / raw_weights.mean()

    _print_weight_summary(bin_idx, bin_counts, weights)
    return weights.astype(np.float32)


def _print_weight_summary(
    bin_idx: np.ndarray,
    bin_counts: np.ndarray,
    weights: np.ndarray,
) -> None:
    print("\n[weights] Per-bin sample weights (mean-normalized):")
    print(f"  {'Bin':<15} {'Samples':>8}  {'Weight':>8}")
    print(f"  {'-'*36}")
    for i, label in enumerate(config.D_BIN_LABELS):
        count = int(bin_counts[i])
        if count == 0:
            print(f"  {label:<15} {'0':>8}  {'n/a':>8}")
            continue
        mask = bin_idx == i
        w = weights[mask].mean()
        print(f"  {label:<15} {count:>8,}  {w:>8.3f}×")
    print()
