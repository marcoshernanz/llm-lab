/// Feedforward forward and backward helpers for phase 3.

#pragma once

#include "core.h"

namespace feed_forward {

/// Hold the intermediate tensors for the feedforward sublayer.
struct Cache {
  std::vector<float> hidden_pre;
  std::vector<float> hidden;
  std::vector<float> projected_output;
  std::vector<float> d_hidden;
};

/// Run the feedforward sublayer without the skip connection.
void forward(const std::vector<float> &inputs, const Param &hidden_weights,
             const Param &hidden_bias, const Param &output_projection_weights,
             const Param &output_projection_bias, Cache &cache);

/// Backpropagate through the feedforward sublayer and its skip path.
void backward(const std::vector<float> &inputs, Cache &cache,
              const std::vector<float> &d_projected_output, Param &hidden_weights,
              Param &hidden_bias, Param &output_projection_weights,
              Param &output_projection_bias, std::vector<float> &d_inputs);

} // namespace feed_forward
