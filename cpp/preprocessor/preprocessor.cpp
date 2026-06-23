/**
 * CZM Surrogate — C++ Preprocessor Implementation
 *
 * STL-only (no external dependencies except optional nlohmann/json).
 * Minimal JSON parser included for scaler.json loading.
 */

#include "preprocessor.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <cassert>
#include <iostream>

namespace czm {

// ---------------------------------------------------------------------------
// Minimal JSON parser — only handles the flat structure of scaler.json:
//   { "mean": [...], "scale": [...], "n_features": N }
// ---------------------------------------------------------------------------

namespace detail {

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Extract a JSON float array value by key from a simple flat JSON object.
// Assumes the array fits on one logical "line" and uses standard formatting.
static std::vector<float> parse_float_array(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto key_pos = json.find(search);
    if (key_pos == std::string::npos)
        throw std::runtime_error("Key not found in JSON: " + key);

    auto bracket_open = json.find('[', key_pos);
    auto bracket_close = json.find(']', bracket_open);
    if (bracket_open == std::string::npos || bracket_close == std::string::npos)
        throw std::runtime_error("Malformed JSON array for key: " + key);

    std::string array_str = json.substr(bracket_open + 1, bracket_close - bracket_open - 1);
    std::vector<float> values;
    std::istringstream iss(array_str);
    std::string token;
    while (std::getline(iss, token, ',')) {
        // Strip whitespace and newlines
        token.erase(0, token.find_first_not_of(" \t\n\r"));
        token.erase(token.find_last_not_of(" \t\n\r") + 1);
        if (!token.empty())
            values.push_back(std::stof(token));
    }
    return values;
}

} // namespace detail

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

ScalerParams load_scaler(const std::string& json_path) {
    std::string json = detail::read_file(json_path);
    ScalerParams params;
    params.mean  = detail::parse_float_array(json, "mean");
    params.scale = detail::parse_float_array(json, "scale");

    if (params.mean.size() != N_FLAT_FEATURES || params.scale.size() != N_FLAT_FEATURES) {
        throw std::runtime_error(
            "scaler.json has wrong number of features. Expected " +
            std::to_string(N_FLAT_FEATURES) + ", got mean=" +
            std::to_string(params.mean.size()) + " scale=" +
            std::to_string(params.scale.size())
        );
    }
    return params;
}

std::vector<float> normalize(
    const std::vector<float>& flat_input,
    const ScalerParams& scaler
) {
    if (static_cast<int>(flat_input.size()) != N_FLAT_FEATURES)
        throw std::runtime_error("normalize: expected 160 features, got " +
                                 std::to_string(flat_input.size()));

    std::vector<float> out(N_FLAT_FEATURES);
    for (int i = 0; i < N_FLAT_FEATURES; ++i) {
        out[i] = (flat_input[i] - scaler.mean[i]) / scaler.scale[i];
    }
    return out;
}

std::vector<std::vector<float>> reshape_temporal(
    const std::vector<float>& normalized
) {
    if (static_cast<int>(normalized.size()) != N_FLAT_FEATURES)
        throw std::runtime_error("reshape_temporal: expected 160 elements");

    std::vector<std::vector<float>> temporal(N_TIMESTEPS,
                                             std::vector<float>(N_FEATURES_PER_STEP));
    // Raw input: h0 (most recent) first, h19 (oldest) last.
    // Reverse to chronological order: index 0 = h19 (oldest), index 19 = h0 (newest).
    // Matches Python pipeline's reshape_temporal() exactly.
    for (int t = 0; t < N_TIMESTEPS; ++t) {
        int src_t = (N_TIMESTEPS - 1) - t;   // h19→index 0, h0→index 19
        for (int f = 0; f < N_FEATURES_PER_STEP; ++f) {
            temporal[t][f] = normalized[src_t * N_FEATURES_PER_STEP + f];
        }
    }
    return temporal;
}

std::vector<std::vector<float>> compute_deltas(
    const std::vector<std::vector<float>>& temporal
) {
    std::vector<std::vector<float>> deltas(N_TIMESTEPS,
                                           std::vector<float>(N_FEATURES_PER_STEP, 0.0f));
    // t=0: zero-padded (no prior state)
    for (int t = 1; t < N_TIMESTEPS; ++t) {
        for (int f = 0; f < N_FEATURES_PER_STEP; ++f) {
            deltas[t][f] = temporal[t][f] - temporal[t - 1][f];
        }
    }
    return deltas;
}

std::vector<float> build_input_tensor(
    const std::vector<float>& flat_raw,
    const ScalerParams& scaler
) {
    auto normalized = normalize(flat_raw, scaler);
    auto temporal   = reshape_temporal(normalized);
    auto deltas     = compute_deltas(temporal);

    // Concatenate [abs, delta] per timestep → (20, 16) flattened to 320
    std::vector<float> out;
    out.reserve(N_OUTPUT_ELEMENTS);
    for (int t = 0; t < N_TIMESTEPS; ++t) {
        for (int f = 0; f < N_FEATURES_PER_STEP; ++f)
            out.push_back(temporal[t][f]);
        for (int f = 0; f < N_FEATURES_PER_STEP; ++f)
            out.push_back(deltas[t][f]);
    }
    assert(static_cast<int>(out.size()) == N_OUTPUT_ELEMENTS);
    return out;
}

} // namespace czm

// ---------------------------------------------------------------------------
// Standalone test executable (compiled by CMake target `preprocessor_test`)
// ---------------------------------------------------------------------------

#ifdef CZM_PREPROCESSOR_MAIN

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: preprocessor_test <path/to/scaler.json>\n";
        return 1;
    }

    try {
        czm::ScalerParams scaler = czm::load_scaler(argv[1]);
        std::cout << "Loaded scaler: " << scaler.mean.size() << " features\n";

        // Dummy input: all zeros (representative of an intact cohesive element)
        std::vector<float> raw_input(czm::N_FLAT_FEATURES, 0.0f);

        auto tensor = czm::build_input_tensor(raw_input, scaler);
        std::cout << "Output tensor shape: (20, 16) — " << tensor.size() << " elements\n";

        // Print first timestep's absolute + delta values
        std::cout << "Timestep 0 absolute: ";
        for (int f = 0; f < czm::N_FEATURES_PER_STEP; ++f)
            std::cout << tensor[f] << " ";
        std::cout << "\nTimestep 0 deltas:   ";
        for (int f = 0; f < czm::N_FEATURES_PER_STEP; ++f)
            std::cout << tensor[czm::N_FEATURES_PER_STEP + f] << " ";
        std::cout << "\n\nPreprocessor test PASSED\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

#endif // CZM_PREPROCESSOR_MAIN
