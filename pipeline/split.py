"""
Stage 2 — Stratified train / val / test split.

Stratification is on damage bins (config.D_BIN_EDGES) so that rare
initiation samples (D ≈ 0–0.3) appear in every split rather than
concentrating in a single partition.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from pipeline import config


def stratified_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into train / val / test with stratification on D bins.

    Returns:
        (train_df, val_df, test_df)
    """
    # Assign each sample to a D bin for stratification
    strat_labels = pd.cut(
        df[config.LABEL_COL],
        bins=config.D_BIN_EDGES,
        labels=config.D_BIN_LABELS,
        right=False,
    ).astype(str)

    test_frac = config.SPLIT_RATIOS["test"]
    val_frac = config.SPLIT_RATIOS["val"]
    # val fraction relative to (train + val) pool after test is removed
    val_of_trainval = val_frac / (1.0 - test_frac)

    # Step 1: hold out test set
    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_frac,
        random_state=config.RANDOM_SEED,
    )
    trainval_idx, test_idx = next(sss1.split(df, strat_labels))

    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    trainval_strat = strat_labels.iloc[trainval_idx].reset_index(drop=True)

    # Step 2: split train vs val from the trainval pool
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_of_trainval,
        random_state=config.RANDOM_SEED,
    )
    train_idx, val_idx = next(sss2.split(trainval_df, trainval_strat))

    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    _print_split_summary(train_df, val_df, test_df)
    return train_df, val_df, test_df


def _print_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    total = len(train_df) + len(val_df) + len(test_df)
    print(f"\n[split] Split summary (total {total:,} samples):")
    header = f"  {'Bin':<15} {'Train':>8} {'Val':>8} {'Test':>8}"
    print(header)
    print(f"  {'-'*44}")

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        pass  # column header only, rows below

    for bin_label in config.D_BIN_LABELS:
        def _bin_count(df):
            bins = pd.cut(
                df[config.LABEL_COL],
                bins=config.D_BIN_EDGES,
                labels=config.D_BIN_LABELS,
                right=False,
            )
            return (bins == bin_label).sum()

        tr = _bin_count(train_df)
        va = _bin_count(val_df)
        te = _bin_count(test_df)
        print(f"  {bin_label:<15} {tr:>8,} {va:>8,} {te:>8,}")

    print(f"  {'TOTAL':<15} {len(train_df):>8,} {len(val_df):>8,} {len(test_df):>8,}")
    print()
