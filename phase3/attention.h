/// Single-head attention forward and backward helpers for phase 3.

#pragma once

#include "core.h"

namespace attention {

/// Hold the intermediate tensors for the attention sublayer.
struct Cache {
  std::vector<float> queries;
  std::vector<float> keys;
  std::vector<float> values;
  std::vector<float> weights;
  std::vector<float> head;
  std::vector<float> projected;
  std::vector<float> residual;
};

/// Run the full single-head attention sublayer with its skip connection.
Cache forward(const std::vector<float> &inputs, const Param &query_weights,
              const Param &key_weights, const Param &value_weights, const Param &output_weights);

/// Backpropagate through the full single-head attention sublayer.
std::vector<float> backward(const std::vector<float> &inputs, const Cache &cache,
                            const std::vector<float> &d_residual, Param &query_weights,
                            Param &key_weights, Param &value_weights, Param &output_weights);

} // namespace attention
