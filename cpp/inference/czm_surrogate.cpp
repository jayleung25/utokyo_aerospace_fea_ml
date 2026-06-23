/**
 * CZM Surrogate — Inference Wrapper Implementation
 *
 * Two build modes controlled by the CMake option CZM_TF_AVAILABLE:
 *
 *   Stub build (default — no TF installed):
 *     load_model() logs a warning and returns false.
 *     predict() returns 0.0f (intact element).
 *     The C-linkage wrappers compile and link cleanly.
 *     Use this mode during pipeline development before a model is trained.
 *
 *   TF build (-DCZM_TF_AVAILABLE=ON):
 *     Wires up TF_LoadSessionFromSavedModel, TF_SessionRun, etc.
 *     Requires TF_LIB_DIR to be set in CMakeLists.txt.
 *     TODO markers below show exactly where each API call goes.
 */

#include "czm_surrogate.h"

#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cstring>

#ifdef CZM_TF_AVAILABLE
#include <tensorflow/c/c_api.h>
#endif

namespace czm {

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

CZMSurrogate::CZMSurrogate() = default;

CZMSurrogate::~CZMSurrogate() {
#ifdef CZM_TF_AVAILABLE
    // TODO: TF_CloseSession(session_, ...); TF_DeleteSession(...);
    //       TF_DeleteGraph(graph_); TF_DeleteSessionOptions(opts_);
#endif
}

// ---------------------------------------------------------------------------
// load_model
// ---------------------------------------------------------------------------

bool CZMSurrogate::load_model(const std::string& saved_model_path) {
#ifdef CZM_TF_AVAILABLE
    // TODO: Wire TF C API session loading.
    //
    // TF_Status* status = TF_NewStatus();
    // opts_ = TF_NewSessionOptions();
    // graph_ = TF_NewGraph();
    //
    // const char* tags[] = {"serve"};
    // TF_Buffer* run_opts = nullptr;
    // session_ = TF_LoadSessionFromSavedModel(
    //     opts_, run_opts, saved_model_path.c_str(),
    //     tags, 1, graph_, nullptr, status
    // );
    // if (TF_GetCode(status) != TF_OK) {
    //     std::cerr << "[czm_surrogate] load_model failed: "
    //               << TF_Message(status) << "\n";
    //     TF_DeleteStatus(status);
    //     return false;
    // }
    // TF_DeleteStatus(status);
    // model_loaded_ = true;
    // return true;
    (void)saved_model_path;
    std::cerr << "[czm_surrogate] TF C API wiring not yet implemented.\n";
    return false;
#else
    (void)saved_model_path;
    std::cerr << "[czm_surrogate] Stub build — no TensorFlow linked. "
              << "Rebuild with -DCZM_TF_AVAILABLE=ON after exporting SavedModel.\n";
    model_loaded_ = false;
    return false;
#endif
}

// ---------------------------------------------------------------------------
// predict (single element)
// ---------------------------------------------------------------------------

float CZMSurrogate::predict(const float* input_tensor, int input_size) {
    if (input_size != INFERENCE_INPUT_SIZE) {
        throw std::runtime_error(
            "czm_surrogate::predict: expected input_size=" +
            std::to_string(INFERENCE_INPUT_SIZE) + ", got " +
            std::to_string(input_size)
        );
    }

#ifdef CZM_TF_AVAILABLE
    if (!model_loaded_) {
        std::cerr << "[czm_surrogate] predict() called before load_model()\n";
        return 0.0f;
    }
    // TODO: Wire TF_SessionRun for single-sample inference.
    //
    // int64_t dims[] = {1, N_TIMESTEPS, N_OUTPUT_FEATURES};  // {1, 20, 16}
    // TF_Tensor* input_tf = TF_AllocateTensor(TF_FLOAT, dims, 3,
    //     INFERENCE_INPUT_SIZE * sizeof(float));
    // memcpy(TF_TensorData(input_tf), input_tensor, INFERENCE_INPUT_SIZE * sizeof(float));
    //
    // TF_Output input_op  = {TF_GraphOperationByName(graph_, "serving_default_input_1"), 0};
    // TF_Output output_op = {TF_GraphOperationByName(graph_, "StatefulPartitionedCall"), 0};
    // TF_Tensor* output_tf = nullptr;
    // TF_Status* status = TF_NewStatus();
    //
    // TF_SessionRun(session_, nullptr,
    //     &input_op,  &input_tf,  1,
    //     &output_op, &output_tf, 1,
    //     nullptr, 0, nullptr, status);
    //
    // float result = *static_cast<float*>(TF_TensorData(output_tf));
    // TF_DeleteTensor(input_tf);
    // TF_DeleteTensor(output_tf);
    // TF_DeleteStatus(status);
    //
    // return std::clamp(result, 0.0f, 1.0f);
    (void)input_tensor;
    return 0.0f;
#else
    // Stub: return 0.0 (undamaged) for all inputs
    (void)input_tensor;
    return 0.0f;
#endif
}

// ---------------------------------------------------------------------------
// predict_batch
// ---------------------------------------------------------------------------

void CZMSurrogate::predict_batch(const float* inputs, float* outputs, int n) {
#ifdef CZM_TF_AVAILABLE
    if (!model_loaded_) {
        std::cerr << "[czm_surrogate] predict_batch() called before load_model()\n";
        std::fill(outputs, outputs + n, 0.0f);
        return;
    }
    // TODO: Wire batched TF_SessionRun.
    //   int64_t dims[] = {n, N_TIMESTEPS, N_OUTPUT_FEATURES};
    //   ... (similar to predict() above but with batch dimension n)
    (void)inputs;
    std::fill(outputs, outputs + n, 0.0f);
#else
    (void)inputs;
    std::fill(outputs, outputs + n, 0.0f);
#endif
}

} // namespace czm

// ---------------------------------------------------------------------------
// C-linkage wrappers (thin delegates to the C++ class)
// ---------------------------------------------------------------------------

extern "C" {

void* czm_surrogate_create() {
    return new czm::CZMSurrogate();
}

void czm_surrogate_destroy(void* handle) {
    delete static_cast<czm::CZMSurrogate*>(handle);
}

int czm_surrogate_load(void* handle, const char* saved_model_path) {
    auto* s = static_cast<czm::CZMSurrogate*>(handle);
    return s->load_model(std::string(saved_model_path)) ? 1 : 0;
}

float czm_surrogate_predict(void* handle, const float* input, int input_size) {
    auto* s = static_cast<czm::CZMSurrogate*>(handle);
    return s->predict(input, input_size);
}

void czm_surrogate_predict_batch(void* handle, const float* inputs, float* outputs, int n) {
    auto* s = static_cast<czm::CZMSurrogate*>(handle);
    s->predict_batch(inputs, outputs, n);
}

} // extern "C"
