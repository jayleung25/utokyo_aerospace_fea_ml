#pragma once
/**
 * CZM Surrogate — TensorFlow C API Inference Wrapper
 *
 * Loads a TF SavedModel and exposes a UMAT-compatible prediction interface
 * that can be called from an Abaqus VUMAT/UMAT subroutine.
 *
 * C++ class interface for use in modern code.
 * C-linkage functions for Fortran/C UMAT compatibility.
 *
 * Status: interface is fully defined; TF C API internals are stubbed
 * with TODO markers until a SavedModel is exported from training.
 * The file compiles cleanly before a model exists.
 *
 * Input:  (20 * 16) = 320 float32 values — output of czm::build_input_tensor()
 * Output: single float32 in [0, 1] — predicted damage variable D
 */

#include <string>
#include <vector>
#include <memory>

// Forward-declare TF types so this header compiles without TF installed.
// The .cpp file #includes <tensorflow/c/c_api.h> only when TF_AVAILABLE is defined.
struct TF_Session;
struct TF_Graph;
struct TF_Status;
struct TF_SessionOptions;

namespace czm {

static constexpr int INFERENCE_INPUT_SIZE = 320;  // (20 timesteps) * (16 features)

class CZMSurrogate {
public:
    CZMSurrogate();
    ~CZMSurrogate();

    // Non-copyable (owns TF session resources)
    CZMSurrogate(const CZMSurrogate&) = delete;
    CZMSurrogate& operator=(const CZMSurrogate&) = delete;

    /**
     * Load a TF SavedModel from disk.
     * Call once after construction, before any predict() calls.
     *
     * @param saved_model_path  Directory containing saved_model.pb + variables/
     * @return true on success, false on failure (check stderr for details)
     */
    bool load_model(const std::string& saved_model_path);

    /**
     * Predict damage variable D for a single cohesive element.
     *
     * @param input_tensor  Pointer to 320 float32 values (preprocessed)
     * @param input_size    Must equal INFERENCE_INPUT_SIZE (320)
     * @return              Predicted D ∈ [0, 1]; clamped to this range
     */
    float predict(const float* input_tensor, int input_size);

    /**
     * Batch prediction for n cohesive elements.
     * Inputs laid out row-major: inputs[i * INFERENCE_INPUT_SIZE ... (i+1)*... - 1]
     *
     * @param inputs   n * 320 float32 values
     * @param outputs  Caller-allocated buffer for n float32 D predictions
     * @param n        Number of elements in this batch
     */
    void predict_batch(const float* inputs, float* outputs, int n);

    bool is_loaded() const { return model_loaded_; }

private:
    bool model_loaded_ = false;

    // TF session state — opaque pointers, managed manually via TF C API.
    // Null when TF_AVAILABLE is not defined (stub build).
    TF_Graph*          graph_   = nullptr;
    TF_Session*        session_ = nullptr;
    TF_SessionOptions* opts_    = nullptr;
};

} // namespace czm

// ---------------------------------------------------------------------------
// C-linkage API — compatible with Abaqus UMAT/VUMAT (C or Fortran callers)
// ---------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

/** Create a surrogate instance. Returns opaque handle. */
void* czm_surrogate_create();

/** Destroy a surrogate instance created by czm_surrogate_create(). */
void czm_surrogate_destroy(void* handle);

/**
 * Load the SavedModel. Returns 1 on success, 0 on failure.
 * Must be called before czm_surrogate_predict / czm_surrogate_predict_batch.
 */
int czm_surrogate_load(void* handle, const char* saved_model_path);

/**
 * Predict D for a single element.
 *
 * @param handle      Opaque surrogate handle
 * @param input       Pointer to INFERENCE_INPUT_SIZE (320) float32 values
 * @param input_size  Must be 320
 * @return            Predicted D ∈ [0, 1]
 */
float czm_surrogate_predict(void* handle, const float* input, int input_size);

/**
 * Batch prediction. Lays out results in caller-allocated output buffer.
 *
 * @param handle   Opaque surrogate handle
 * @param inputs   n * 320 float32 row-major input array
 * @param outputs  Caller-allocated float32 buffer of length n
 * @param n        Number of elements
 */
void czm_surrogate_predict_batch(void* handle, const float* inputs, float* outputs, int n);

#ifdef __cplusplus
} // extern "C"
#endif
