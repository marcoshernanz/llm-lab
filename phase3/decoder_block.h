/// Decoder block helpers for phase 3.

#pragma once

#include "attention.h"
#include "feed_forward.h"
#include "layer_norm.h"

namespace decoder_block {

/// Hold the intermediate tensors for one decoder block.
struct Cache {
  std::vector<float> block_input;
  attention::Cache attention;
  layer_norm::Cache attention_layer_norm;
  feed_forward::Cache feed_forward;
  layer_norm::Cache feed_forward_layer_norm;
};

/// Hold one trainable decoder block.
class Block {
public:
  Param attention_query_weights;
  Param attention_key_weights;
  Param attention_value_weights;
  Param attention_output_projection_weights;
  Param attention_norm_gain;
  Param attention_norm_bias;
  Param feed_forward_in_weights;
  Param feed_forward_in_bias;
  Param feed_forward_out_weights;
  Param feed_forward_out_bias;
  Param feed_forward_norm_gain;
  Param feed_forward_norm_bias;

  /// Construct one decoder block with correctly sized parameter tensors.
  Block();

  /// Initialize one decoder block with scaled weights and simple biases.
  void init();

  /// Reset every decoder-block gradient buffer to zero.
  void zero_grad();

  /// Scale every decoder-block gradient buffer by one constant.
  void scale_grads(float scale);

  /// Apply one optimizer step to every decoder-block parameter tensor.
  void update();

  /// Run one full decoder-block forward pass.
  Cache forward(const std::vector<float> &block_input) const;

  /// Backpropagate through one full decoder block.
  std::vector<float> backward(const Cache &cache, const std::vector<float> &d_block_output);
};

} // namespace decoder_block
