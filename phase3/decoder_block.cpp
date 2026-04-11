/// Decoder block helpers for phase 3.

#include "decoder_block.h"
#include "profiler.h"

namespace decoder_block {

/// Construct one decoder block with correctly sized parameter tensors.
Block::Block()
    : attention_query_weights(embedding_dim * attention_dim),
      attention_key_weights(embedding_dim * attention_dim),
      attention_value_weights(embedding_dim * attention_dim),
      attention_output_projection_weights(attention_dim * embedding_dim), attention_rms_gain(embedding_dim),
      feed_forward_in_weights(embedding_dim * feed_forward_dim),
      feed_forward_in_bias(feed_forward_dim),
      feed_forward_out_weights(feed_forward_dim * embedding_dim),
      feed_forward_out_bias(embedding_dim), feed_forward_rms_gain(embedding_dim) {}

/// Initialize one decoder block with scaled weights and simple biases.
void Block::init() {
  attention_query_weights.init_normal(fan_in_stddev(embedding_dim));
  attention_key_weights.init_normal(fan_in_stddev(embedding_dim));
  attention_value_weights.init_normal(fan_in_stddev(embedding_dim));
  attention_output_projection_weights.init_normal(fan_in_stddev(attention_dim));
  attention_rms_gain.init_ones();
  feed_forward_in_weights.init_normal(fan_in_stddev(embedding_dim));
  feed_forward_in_bias.init_zeros();
  feed_forward_out_weights.init_normal(fan_in_stddev(feed_forward_dim));
  feed_forward_out_bias.init_zeros();
  feed_forward_rms_gain.init_ones();
}

/// Reset every decoder-block gradient buffer to zero.
void Block::zero_grad() {
  attention_query_weights.zero_grad();
  attention_key_weights.zero_grad();
  attention_value_weights.zero_grad();
  attention_output_projection_weights.zero_grad();
  attention_rms_gain.zero_grad();
  feed_forward_in_weights.zero_grad();
  feed_forward_in_bias.zero_grad();
  feed_forward_out_weights.zero_grad();
  feed_forward_out_bias.zero_grad();
  feed_forward_rms_gain.zero_grad();
}

/// Scale every decoder-block gradient buffer by one constant.
void Block::scale_grads(float scale) {
  attention_query_weights.scale_grad(scale);
  attention_key_weights.scale_grad(scale);
  attention_value_weights.scale_grad(scale);
  attention_output_projection_weights.scale_grad(scale);
  attention_rms_gain.scale_grad(scale);
  feed_forward_in_weights.scale_grad(scale);
  feed_forward_in_bias.scale_grad(scale);
  feed_forward_out_weights.scale_grad(scale);
  feed_forward_out_bias.scale_grad(scale);
  feed_forward_rms_gain.scale_grad(scale);
}

/// Apply one optimizer step to every decoder-block parameter tensor.
void Block::update() {
  attention_query_weights.update();
  attention_key_weights.update();
  attention_value_weights.update();
  attention_output_projection_weights.update();
  attention_rms_gain.update();
  feed_forward_in_weights.update();
  feed_forward_in_bias.update();
  feed_forward_out_weights.update();
  feed_forward_out_bias.update();
  feed_forward_rms_gain.update();
}

/// Run one full decoder-block forward pass.
void Block::forward(const std::vector<float> &block_input, Cache &cache) const {
  const profiler::Scope scope("decoder_block.forward");
  copy_into(cache.block_input, block_input);
  rms_norm::forward(block_input, attention_rms_gain, cache.attention_rms_norm);
  attention::forward(cache.attention_rms_norm.rms_norm_output, attention_query_weights,
                     attention_key_weights, attention_value_weights,
                     attention_output_projection_weights, cache.attention);
  cache.attention_residual.resize(batch_size * context_len * embedding_dim);
  for (size_t i = 0; i < cache.attention_residual.size(); ++i) {
    cache.attention_residual[i] = block_input[i] + cache.attention.projected_output[i];
  }
  rms_norm::forward(cache.attention_residual, feed_forward_rms_gain, cache.feed_forward_rms_norm);
  feed_forward::forward(cache.feed_forward_rms_norm.rms_norm_output, feed_forward_in_weights,
                        feed_forward_in_bias, feed_forward_out_weights, feed_forward_out_bias,
                        cache.feed_forward);
  cache.block_output.resize(batch_size * context_len * embedding_dim);
  for (size_t i = 0; i < cache.block_output.size(); ++i) {
    cache.block_output[i] = cache.attention_residual[i] + cache.feed_forward.projected_output[i];
  }
}

/// Backpropagate through one full decoder block.
void Block::backward(Cache &cache, const std::vector<float> &d_block_output,
                     std::vector<float> &d_block_input) {
  const profiler::Scope scope("decoder_block.backward");
  copy_into(d_attention_residual, d_block_output);
  feed_forward::backward(cache.feed_forward_rms_norm.rms_norm_output, cache.feed_forward,
                         d_block_output, feed_forward_in_weights, feed_forward_in_bias,
                         feed_forward_out_weights, feed_forward_out_bias,
                         d_feed_forward_norm_output);
  rms_norm::backward(d_feed_forward_norm_output, cache.attention_residual,
                     cache.feed_forward_rms_norm, feed_forward_rms_gain,
                     d_block_input_from_norm);
  for (size_t i = 0; i < d_attention_residual.size(); ++i) {
    d_attention_residual[i] += d_block_input_from_norm[i];
  }

  copy_into(d_block_input, d_attention_residual);
  attention::backward(cache.attention_rms_norm.rms_norm_output, cache.attention,
                      d_attention_residual,
                      attention_query_weights, attention_key_weights, attention_value_weights,
                      attention_output_projection_weights, d_attention_norm_output);
  rms_norm::backward(d_attention_norm_output, cache.block_input, cache.attention_rms_norm,
                     attention_rms_gain, d_block_input_from_norm);
  for (size_t i = 0; i < d_block_input.size(); ++i) {
    d_block_input[i] += d_block_input_from_norm[i];
  }
}

} // namespace decoder_block
