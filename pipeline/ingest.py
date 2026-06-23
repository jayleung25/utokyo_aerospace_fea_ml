"""
Stage 1 — Ingest and validate raw FEA CSV data.
"""

import pandas as pd
import numpy as np
from pipeline import config


def load_raw(path: str) -> pd.DataFrame:
    """Load raw CSV, validate schema and basic data integrity.

    Prints a summary of row counts and per-bin label distribution.
    Raises ValueError on any schema or integrity violation.
    """
    print(f"[ingest] Loading {path} ...")
    df = pd.read_csv(path)
    print(f"[ingest] Loaded {len(df):,} rows, {len(df.columns)} columns")

    _validate_columns(df)
    _validate_no_nulls(df)
    _validate_label_range(df)
    _validate_mode_mixity(df)
    _print_bin_summary(df)

    return df


# ---------------------------------------------------------------------------
# Internal validators
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame) -> None:
    expected = set(config.FEATURE_COLS + [config.LABEL_COL])
    actual = set(df.columns)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise ValueError(f"[ingest] Missing columns: {sorted(missing)}")
    if extra:
        raise ValueError(f"[ingest] Unexpected columns: {sorted(extra)}")
    print(f"[ingest] Schema OK — {len(config.FEATURE_COLS)} feature cols + label")


def _validate_no_nulls(df: pd.DataFrame) -> None:
    null_counts = df.isnull().sum()
    bad = null_counts[null_counts > 0]
    if not bad.empty:
        raise ValueError(f"[ingest] NaN values found:\n{bad}")
    print("[ingest] No NaN values found")


def _validate_label_range(df: pd.DataFrame) -> None:
    lo = df[config.LABEL_COL].min()
    hi = df[config.LABEL_COL].max()
    if lo < 0.0 or hi > 1.0:
        raise ValueError(
            f"[ingest] label out of [0, 1]: min={lo:.6f}, max={hi:.6f}"
        )
    print(f"[ingest] label range OK — min={lo:.6f}, max={hi:.6f}")


def _validate_mode_mixity(df: pd.DataFrame) -> None:
    """Confirm all modeMixity columns are 0 (DCB-only dataset)."""
    mixity_cols = [c for c in config.FEATURE_COLS if "modeMixity" in c]
    max_mixity = df[mixity_cols].abs().max().max()
    if max_mixity > 0.0:
        print(
            f"[ingest] WARNING: modeMixity is non-zero (max={max_mixity:.6f}). "
            "Dataset contains ENF or MMB samples — re-check assumptions."
        )
    else:
        print("[ingest] modeMixity = 0.0 for all samples (DCB-only, Mode I)")


def _print_bin_summary(df: pd.DataFrame) -> None:
    bins = pd.cut(
        df[config.LABEL_COL],
        bins=config.D_BIN_EDGES,
        labels=config.D_BIN_LABELS,
        right=False,
    )
    counts = bins.value_counts().reindex(config.D_BIN_LABELS)
    total = len(df)
    print("\n[ingest] Damage distribution:")
    print(f"  {'Bin':<15} {'Count':>8}  {'%':>6}")
    print(f"  {'-'*32}")
    for label, count in counts.items():
        print(f"  {label:<15} {count:>8,}  {100*count/total:>5.1f}%")
    print()
