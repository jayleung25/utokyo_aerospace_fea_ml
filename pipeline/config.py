"""
Single source of truth for all pipeline constants.
No magic numbers anywhere else in the codebase — import from here.
"""

# History window and per-step feature count
N_TIMESTEPS = 20
N_FEATURES_PER_STEP = 8

# The 8 mechanical state variables recorded at each timestep, in column order
FEATURE_NAMES = [
    "failureIndex",
    "modeMixity",
    "separN",
    "separT1",
    "separT2",
    "tractN",
    "tractT1",
    "tractT2",
]

# 160 input columns: h0_failureIndex, h0_modeMixity, ..., h19_tractT2
FEATURE_COLS = [
    f"h{step}_{var}"
    for step in range(N_TIMESTEPS)
    for var in FEATURE_NAMES
]

LABEL_COL = "label"

# Damage variable bins: initiation / transition / post-peak / failed
# Upper edge is 1.0 + epsilon so pd.cut includes D == 1.0 in the "failed" bin
D_BIN_EDGES = [0.0, 0.3, 0.7, 0.9, 1.0 + 1e-6]
D_BIN_LABELS = ["initiation", "transition", "post_peak", "failed"]

# Train / val / test proportions (must sum to 1.0)
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}

RANDOM_SEED = 42

# Default batch size for tf.data pipelines
DEFAULT_BATCH_SIZE = 128
