/// RMSNorm forward and backward helpers for phase 3.

#pragma once

#include "core.h"

namespace rms_norm {

/// Hold the intermediate tensors for one RMSNorm application.
struct Cache {
  std::vector<float> normalized_input;
  std::vector<float> rms_norm_output;
  std::vector<float> inv_rms;
};

/// Run one RMSNorm over the embedding dimension.
void forward(const std::vector<float> &inputs, const Param &gain, Cache &cache);

/// Backpropagate through one RMSNorm application.
void backward(const std::vector<float> &d_output, const std::vector<float> &inputs,
              const Cache &cache, Param &gain, std::vector<float> &d_inputs);

} // namespace rms_norm
