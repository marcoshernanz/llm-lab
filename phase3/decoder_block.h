/// Decoder block helpers for phase 3.

#pragma once

#include "attention.h"
#include "feed_forward.h"
#include "rms_norm.h"

namespace decoder_block {

/// Hold the intermediate tensors for one decoder block.
struct Cache {
  std::vector<float> block_input;
  rms_norm::Cache attention_rms_norm;
  attention::Cache attention;
  std::vector<float> attention_residual;
  rms_norm::Cache feed_forward_rms_norm;
  feed_forward::Cache feed_forward;
  std::vector<float> block_output;
};

/// Hold one trainable decoder block.
class Block {
public:
  Param attention_query_weights;
  Param attention_key_weights;
  Param attention_value_weights;
  Param attention_output_projection_weights;
  Param attention_rms_gain;
  Param feed_forward_in_weights;
  Param feed_forward_in_bias;
  Param feed_forward_out_weights;
  Param feed_forward_out_bias;
  Param feed_forward_rms_gain;

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
  void forward(const std::vector<float> &block_input, Cache &cache) const;

  /// Backpropagate through one full decoder block.
  void backward(Cache &cache, const std::vector<float> &d_block_output,
                std::vector<float> &d_block_input);

private:
  /// Reuse block-local backward buffers across training steps.
  mutable std::vector<float> d_feed_forward_norm_output;
  mutable std::vector<float> d_attention_residual;
  mutable std::vector<float> d_attention_norm_output;
  mutable std::vector<float> d_block_input_from_norm;
};

} // namespace decoder_block
