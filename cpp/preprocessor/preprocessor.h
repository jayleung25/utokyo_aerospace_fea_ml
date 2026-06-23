#pragma once
/**
 * CZM Surrogate — C++ Preprocessor
 *
 * High-performance mirror of pipeline/preprocess.py.
 * Takes a raw 160-element flat feature vector from the FEA solver,
 * applies StandardScaler normalization (from scaler.json), reshapes
 * to (20, 8) temporal format, and computes delta features — producing
 * a (20, 16) tensor ready for inference.
 *
 * Designed to be called at every cohesive element integration point
 * inside an Abaqus UMAT/VUMAT subroutine with zero Python overhead.
 */

#include <string>
#include <vector>
#include <array>
#include <stdexcept>

namespace czm {

// Pipeline constants — must match pipeline/config.py exactly
static constexpr int N_TIMESTEPS         = 20;
static constexpr int N_FEATURES_PER_STEP = 8;
static constexpr int N_FLAT_FEATURES     = N_TIMESTEPS * N_FEATURES_PER_STEP; // 160
static constexpr int N_OUTPUT_FEATURES   = N_FEATURES_PER_STEP * 2;           // 16 (abs + delta)
static constexpr int N_OUTPUT_ELEMENTS   = N_TIMESTEPS * N_OUTPUT_FEATURES;   // 320

/**
 * StandardScaler parameters loaded from scaler.json.
 * Produced by pipeline/export.py, consumed here at inference time.
 */
struct ScalerParams {
    std::vector<float> mean;   // length 160
    std::vector<float> scale;  // length 160
};

/**
 * Load a scaler.json written by pipeline/export.py::save_scaler().
 * Throws std::runtime_error on parse failure.
 */
ScalerParams load_scaler(const std::string& json_path);

/**
 * Apply z-score normalization: x_scaled[i] = (x[i] - mean[i]) / scale[i]
 *
 * @param flat_input  Raw 160-element feature vector (h0_feat0 ... h19_feat7)
 * @param scaler      Loaded scaler parameters
 * @return            160-element normalized vector
 */
std::vector<float> normalize(
    const std::vector<float>& flat_input,
    const ScalerParams& scaler
);

/**
 * Reshape a flat 160-element vector to (20, 8) temporal layout in chronological order.
 * The raw input has h0=most recent, h19=oldest. This function reverses the temporal
 * axis so Row 0 = h19 (oldest) and Row 19 = h0 (most recent/current state).
 * This matches the Python pipeline's reshape_temporal() output exactly.
 *
 * @param normalized  160-element normalized vector (h0 features first, h19 last)
 * @return            (20 × 8) chronological layout: row 0=h19, row 19=h0
 */
std::vector<std::vector<float>> reshape_temporal(
    const std::vector<float>& normalized
);

/**
 * Compute per-timestep delta features.
 * delta[t][f] = temporal[t][f] - temporal[t-1][f]
 * delta[0][f] = 0.0  (zero-padded, no prior state at step 0)
 *
 * @param temporal  (20, 8) temporal layout
 * @return          (20, 8) delta layout
 */
std::vector<std::vector<float>> compute_deltas(
    const std::vector<std::vector<float>>& temporal
);

/**
 * Full preprocessing pipeline: normalize → reshape → delta → concatenate.
 *
 * Output layout (row-major, 320 elements):
 *   [t0_abs0..t0_abs7, t0_d0..t0_d7,  t1_abs0..t1_abs7, t1_d0..t1_d7, ...]
 * i.e. for each timestep: 8 absolute values followed by 8 delta values.
 * This matches the (N, 20, 16) tensor shape produced by pipeline/preprocess.py.
 *
 * @param flat_raw  160-element raw feature vector from FEA solver
 * @param scaler    Loaded scaler parameters
 * @return          320-element flattened (20, 16) tensor, ready for inference
 */
std::vector<float> build_input_tensor(
    const std::vector<float>& flat_raw,
    const ScalerParams& scaler
);

} // namespace czm
