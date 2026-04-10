/// LayerNorm forward and backward helpers for phase 3.

#pragma once

#include "core.h"

namespace layer_norm {

/// Hold the intermediate tensors for one LayerNorm application.
struct Cache {
  std::vector<float> normalized_input;
  std::vector<float> layer_norm_output;
  std::vector<float> inv_std;
};

/// Run one LayerNorm over the embedding dimension.
Cache forward(const std::vector<float> &inputs, const Param &scale, const Param &shift);

/// Backpropagate through one LayerNorm application.
std::vector<float> backward(const std::vector<float> &d_output, const Cache &cache, Param &scale,
                            Param &shift);

} // namespace layer_norm
