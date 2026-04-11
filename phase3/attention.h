/// Multi-head attention forward and backward helpers for phase 3.

#pragma once

#include "core.h"

namespace attention {

/// Hold the intermediate tensors for the attention sublayer.
struct Cache {
  std::vector<float> queries;
  std::vector<float> keys;
  std::vector<float> values;
  std::vector<float> attention_weights;
  std::vector<float> attended_values;
  std::vector<float> projected_output;
  std::vector<float> d_attended_values;
  std::vector<float> d_attention_weights;
  std::vector<float> d_values;
  std::vector<float> d_scores;
  std::vector<float> d_queries;
  std::vector<float> d_keys;
};

/// Run the full multi-head attention sublayer without the skip connection.
void forward(const std::vector<float> &inputs, const Param &query_weights,
             const Param &key_weights, const Param &value_weights,
             const Param &output_projection_weights, Cache &cache);

/// Backpropagate through the full multi-head attention sublayer.
void backward(const std::vector<float> &inputs, Cache &cache,
              const std::vector<float> &d_projected_output, Param &query_weights,
              Param &key_weights, Param &value_weights,
              Param &output_projection_weights, std::vector<float> &d_inputs);

} // namespace attention
