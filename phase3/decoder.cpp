/// Stacked decoder helpers for phase 3.

#include "decoder.h"
#include "profiler.h"

namespace decoder {

/// Construct one stacked decoder with the configured number of blocks.
Stack::Stack() : blocks(num_decoder_blocks) {}

/// Initialize every block in the decoder.
void Stack::init() {
  for (decoder_block::Block &block : blocks) {
    block.init();
  }
}

/// Reset every decoder-block gradient buffer to zero.
void Stack::zero_grad() {
  for (decoder_block::Block &block : blocks) {
    block.zero_grad();
  }
}

/// Scale every decoder-block gradient buffer by one constant.
void Stack::scale_grads(float scale) {
  for (decoder_block::Block &block : blocks) {
    block.scale_grads(scale);
  }
}

/// Apply one optimizer step to every decoder block.
void Stack::update() {
  for (decoder_block::Block &block : blocks) {
    block.update();
  }
}

/// Run one full decoder-stack forward pass.
void Stack::forward(const std::vector<float> &decoder_input, Cache &cache) const {
  const profiler::Scope scope("decoder.forward");
  cache.blocks.resize(blocks.size());
  for (size_t i = 0; i < blocks.size(); ++i) {
    const std::vector<float> &block_input = i == 0 ? decoder_input : cache.blocks[i - 1].block_output;
    blocks[i].forward(block_input, cache.blocks[i]);
  }
  if (blocks.empty()) {
    copy_into(cache.decoder_output, decoder_input);
    return;
  }
  copy_into(cache.decoder_output, cache.blocks.back().block_output);
}

/// Backpropagate through the full decoder stack.
void Stack::backward(Cache &cache, const std::vector<float> &d_decoder_output,
                     std::vector<float> &d_decoder_input) {
  const profiler::Scope scope("decoder.backward");
  const std::vector<float> *current_input = &d_decoder_output;
  std::vector<float> *current_output = &backward_buffer_a;

  for (size_t i = blocks.size(); i-- > 0;) {
    if (i == 0) {
      current_output = &d_decoder_input;
    }
    blocks[i].backward(cache.blocks[i], *current_input, *current_output);
    current_input = current_output;
    current_output = current_output == &backward_buffer_a ? &backward_buffer_b : &backward_buffer_a;
  }
}

} // namespace decoder
