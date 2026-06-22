# UTokyo Aerospace FEA/ML — CZM Surrogate Project

## Project Overview

This project replaces computationally expensive Cohesive Zone Model (CZM) constitutive solves in Finite Element Analysis (FEA) with faster ML surrogate models. Instead of computing the damage variable D analytically at every cohesive element integration point every time step, an ML model predicts D directly from the local mechanical state history — delivering ~49.5% CPU time savings (demonstrated with the baseline ANN on 70k data).

**Research context:** University of Tokyo, aerospace composites delamination.

---

## Data Notice

`data/` are **not tracked in git**

In order to execute the pipeline, add the data to the `data/` directory.

---

## Background: FEA and CZM

### Finite Element Analysis (FEA)
FEA is a numerical method that discretizes a physical structure into a mesh of elements, then solves governing PDEs (e.g., equilibrium equations) over that mesh. It is the standard tool for structural analysis of composite aerospace parts — predicting stress, strain, and deformation fields under applied loads.

### Cohesive Zone Model (CZM)
CZM is a constitutive model for interface/delamination failure in layered composites. Each cohesive element represents a thin adhesive interface between composite plies. At every integration point and time step, the CZM subroutine:
1. Takes the local separation vector (opening + shear) and traction state as input.
2. Computes a scalar **damage variable D ∈ [0, 1]** (0 = intact, 1 = fully failed/delaminated).
3. Updates element stiffness accordingly.

Elements used: **UCOH3D8** (8-node 3D cohesive elements).

This per-element, per-timestep constitutive solve is the computational bottleneck in large-scale delamination simulations.

### ML-CZM
The ML surrogate replaces the CZM subroutine call. Given the same mechanical state inputs (formatted as 20-step history, see Variables below), the model predicts D directly. It is embedded in the FEA solver and must:
- Be fast at inference (no model is useful if it's slower than CZM)
- Predict D within physically valid bounds [0, 1]
- Respect monotonicity (D never decreases — damage is irreversible)

---

## Load Testing Methods

Three standard fracture test geometries are used to characterize composite delamination:

### DCB — Double Cantilever Beam (Mode I)
Two cantilever arms bonded at the interface with a pre-crack. Arms are pulled apart symmetrically by equal and opposite forces P. This produces **pure Mode I fracture** — crack opens by normal (tensile) separation, no shear.

```
      ↑ P
  ----+----
  =========  ← cohesive interface (pre-crack → )
  ----+----
      ↓ P
```

**Current training data is exclusively DCB (see modeMixity section below).**

### ENF — End-Notched Flexure (Mode II)
A beam with a pre-crack is supported at both ends and loaded at the center. This produces **pure Mode II fracture** — crack propagates by in-plane shear, no normal opening.

```
       ↓ P
  -----+-----
  =====+===== ← cohesive interface
  |         |
  (support) (support)
```

### MMB — Mixed-Mode Bending (Mixed Mode I + II)
Similar to ENF but with an additional lever arm offset by distance `c` from the center. Adjusting `c` controls the **mode mixture ratio** between Mode I and Mode II, allowing generation of fracture data at any desired blend of opening and shear.

---

## Input Variables

Each data row represents the mechanical state of one cohesive element at one time step. The history formulation stores **20 prior time steps** (h0 = most recent, h19 = oldest), each with 8 fields:

| Variable | Description |
|---|---|
| `failureIndex` | Current failure criterion value (normalized loading relative to strength) |
| `modeMixity` | Mode mixture ratio: 0.0 = pure Mode I (DCB), higher values → more Mode II |
| `separN` | Normal separation — opening displacement (Mode I component) |
| `separT1` | Tangential separation, direction 1 — shear displacement |
| `separT2` | Tangential separation, direction 2 — shear displacement |
| `tractN` | Normal traction — stress in opening direction |
| `tractT1` | Tangential traction, direction 1 — shear stress |
| `tractT2` | Tangential traction, direction 2 — shear stress |

Column naming convention: `h{step}_{variable}` (e.g., `h0_failureIndex`, `h3_tractN`).

**Total input dimensionality: 20 × 8 = 160 features.**
**Target output:** `label` — the damage variable D ∈ [0, 1].

---

## modeMixity Observation — DCB Only (Current Limitation)

In both the 50k and 70k datasets, **every `h*_modeMixity` value is exactly 0.0**. This means:

- Only pure Mode I (DCB) loading scenarios are present in the training data.
- No Mode II (ENF) or mixed-mode (MMB) data has been collected yet.
- The trained models have **zero generalization to ENF or MMB conditions** — applying them outside DCB loading is physically meaningless until more data is collected.

This is a known gap. Future work requires FEA simulation runs under ENF and MMB configurations to expand the training distribution.

---

## Damage Level Distribution — Class Imbalance

The damage variable D in both datasets is **heavily concentrated at D = 0.9–1.0**. This reflects physical reality: in a simulation that runs until delamination propagates, the vast majority of cohesive element time steps occur *after* significant damage has accumulated.

**Implications:**
- The model learns the high-D regime very well (most training signal comes from it).
- **Mid-range damage (D ≈ 0.3–0.7) is underrepresented** — initiation and propagation stages are rare in the data, and model accuracy in this range is lower.
- Damage initiation (D ≈ 0 transitioning upward) is the most physically important regime and the hardest to fit.
- Future data collection should deliberately oversample early loading stages to address this imbalance.

The 70k dataset already improved on this by adding early-loading samples compared to 50k, yielding a 65% MSE reduction.

---

## Baseline: Original ANN Model

The reference implementation is a fully-connected ANN trained on 70k samples.

### Architecture
```
Input(160) → Dense(256) → Dense(128) → Dense(128) → Dense(64) → Dense(32) → Output(1)
```
- Hidden activations: ReLU / Leaky ReLU
- Output activation: linear (continuous regression)

### Training Configuration
| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 5e-4 |
| Adam β1, β2, ε | 0.9, 0.999, 1e-8 |
| LR schedule | Exponential decay (rate=0.999, steps=160,000) |
| Batch size | 128 |
| Epochs | 200 |
| Train / val split | 90% / 10% |
| Weight init std | 0.02 |

### Results

| Dataset | Val MSE | Training Time |
|---|---|---|
| 50k (initial) | 0.000109 | ~484s |
| 70k (improved) | 0.0000383 | ~725s |

**CPU time vs conventional CZM:**
- 50k model: ~34% savings
- 70k model: ~49.5% savings

### Known Issues (Baseline Limitations)
- Predictions occasionally fall outside [0, 1]; physically clamped post-hoc.
- Force vs. crack opening response curves show discrepancies from true CZM, especially in mid-range loading.
- Training loss shows noisy fluctuations — convergence is not always monotone.
- High-D regime dominates the loss; mid-range damage has higher variance errors.
- No generalization to ENF or MMB (all modeMixity = 0).

**The 70k ANN result is the baseline. All new models must beat or match it.**

---

## The Problem

The baseline ANN achieves a very low validation MSE (3.83×10⁻⁵), which looks excellent on paper. Yet when it is embedded back into the FEA solver and run on a DCB simulation, the global structural response diverges from the true CZM result in two measurable ways:

1. **Peak load overestimation.** The ML-CZM force-displacement curve peaks ~25–30% higher than the CZM curve (~103 N vs ~80 N). The surrogate thinks the interface is tougher than it is.
2. **Slower crack propagation.** At the same load increment, ML-CZM predicts a shorter delamination front than CZM. The crack is being held back.

These two symptoms have the same root cause.

### Why Low MSE Does Not Guarantee Physical Accuracy

MSE is computed as an average over all samples. Because ~90% of the dataset has D near 1.0 (already fully damaged elements), the loss is dominated by predictions in that regime — where the answer is nearly always "this element is dead." The model learns that regime very well, and the aggregate MSE looks great.

The physically critical samples are the ones where D is actively transitioning — roughly **D ≈ 0.1 to 0.7**. These are the cohesive elements at the crack tip, in the process zone, currently softening. They control:
- How stiff the interface is at any given moment
- When each element "breaks" and redistributes load to its neighbors
- How fast the crack front advances through the structure

These samples are a small fraction of the training data, so even systematic errors there produce negligible MSE penalty. But those same errors, applied at every process-zone element at every time step, **compound across the entire simulation**. A consistent underprediction of D in the transition zone means those elements stay stiffer longer than they should → the structure carries more load than CZM would allow → peak force is overestimated → crack grows more slowly.

### Summary

The model is accurate where data is abundant and the physics are irrelevant (D ≈ 1), and inaccurate where data is sparse and the physics are critical (D ≈ 0–0.7). MSE as a training objective is blind to this distinction. **Fixing the aggregate loss metric does not fix the physical prediction — the error distribution matters more than the error magnitude.**

---

## Potential Solutions

| Model | Rationale |
|---|---|
| **Linear Regression** | Sanity check — confirms the problem is non-linear |
| **Random Forest** | Strong tabular baseline; interpretable feature importances |
| **XGBoost** | Often best on tabular regression tasks; fast inference |
| **LSTM** | Recurrent; naturally exploits the h0→h19 temporal ordering |
| **GRU** | Lighter recurrent alternative to LSTM; often similar performance |
| **ANN (baseline)** | Fully-connected 160→256→128→128→64→32→1 |

### Additional Directions
- **Hyperparameter optimization:** Optuna or grid search for learning rate, hidden size, depth, dropout.
- **Feature engineering:** Delta features (Δseparation, Δtraction across history steps), normalized inputs.
- **Data augmentation:** Oversample mid-damage range to address imbalance.
- **Physics constraints:** Enforce D monotonicity (D_{t+1} ≥ D_t) as a loss penalty.

---

## Evaluation Metrics

| Metric | Purpose |
|---|---|
| Validation MSE | Primary — match against baseline 0.0000383 |
| Validation MAE | Secondary — interpretable error magnitude |
| Force vs. crack opening fidelity | Physics check — run full FEA with ML-CZM, compare force-displacement curve to CZM ground truth |
| Inference time per element | Efficiency — must be faster than CZM to be useful |
