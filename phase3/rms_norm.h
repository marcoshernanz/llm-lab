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
Cache forward(const std::vector<float> &inputs, const Param &gain);

/// Backpropagate through one RMSNorm application.
std::vector<float> backward(const std::vector<float> &d_output, const std::vector<float> &inputs,
                            const Cache &cache, Param &gain);

} // namespace rms_norm
