/// Stacked decoder helpers for phase 3.

#pragma once

#include "decoder_block.h"

namespace decoder {

/// Hold the intermediate tensors for the full decoder stack.
struct Cache {
  std::vector<decoder_block::Cache> blocks;
  std::vector<float> decoder_output;
};

/// Hold one stacked decoder made of several decoder blocks.
class Stack {
public:
  std::vector<decoder_block::Block> blocks;

  /// Construct one stacked decoder with the configured number of blocks.
  Stack();

  /// Initialize every block in the decoder.
  void init();

  /// Reset every decoder-block gradient buffer to zero.
  void zero_grad();

  /// Scale every decoder-block gradient buffer by one constant.
  void scale_grads(float scale);

  /// Apply one optimizer step to every decoder block.
  void update();

  /// Run one full decoder-stack forward pass.
  void forward(const std::vector<float> &decoder_input, Cache &cache) const;

  /// Backpropagate through the full decoder stack.
  void backward(Cache &cache, const std::vector<float> &d_decoder_output,
                std::vector<float> &d_decoder_input);

private:
  /// Reuse the alternating decoder backward buffers across training steps.
  mutable std::vector<float> backward_buffer_a;
  mutable std::vector<float> backward_buffer_b;
};

} // namespace decoder
