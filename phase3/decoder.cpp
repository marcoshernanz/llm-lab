/// Stacked decoder helpers for phase 3.

#include "decoder.h"

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
Cache Stack::forward(const std::vector<float> &decoder_input) const {
  Cache cache;
  cache.blocks.reserve(blocks.size());

  std::vector<float> block_input = decoder_input;
  for (const decoder_block::Block &block : blocks) {
    cache.blocks.push_back(block.forward(block_input));
    block_input = cache.blocks.back().block_output;
  }

  cache.decoder_output = block_input;
  return cache;
}

/// Backpropagate through the full decoder stack.
std::vector<float> Stack::backward(const Cache &cache, const std::vector<float> &d_decoder_output) {
  std::vector<float> d_block_input = d_decoder_output;
  for (size_t i = blocks.size(); i-- > 0;) {
    d_block_input = blocks[i].backward(cache.blocks[i], d_block_input);
  }
  return d_block_input;
}

} // namespace decoder
