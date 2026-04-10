/// Decoder block helpers for phase 3.

#include "decoder_block.h"

namespace decoder_block {

/// Construct one decoder block with correctly sized parameter tensors.
Block::Block()
    : attention_query_weights(embedding_dim * attention_dim),
      attention_key_weights(embedding_dim * attention_dim),
      attention_value_weights(embedding_dim * attention_dim),
      attention_output_projection_weights(attention_dim * embedding_dim),
      attention_norm_gain(embedding_dim), attention_norm_bias(embedding_dim),
      feed_forward_in_weights(embedding_dim * feed_forward_dim),
      feed_forward_in_bias(feed_forward_dim),
      feed_forward_out_weights(feed_forward_dim * embedding_dim),
      feed_forward_out_bias(embedding_dim), feed_forward_norm_gain(embedding_dim),
      feed_forward_norm_bias(embedding_dim) {}

/// Initialize one decoder block with scaled weights and simple biases.
void Block::init() {
  attention_query_weights.init_normal(fan_in_stddev(embedding_dim));
  attention_key_weights.init_normal(fan_in_stddev(embedding_dim));
  attention_value_weights.init_normal(fan_in_stddev(embedding_dim));
  attention_output_projection_weights.init_normal(fan_in_stddev(attention_dim));
  attention_norm_gain.init_ones();
  attention_norm_bias.init_zeros();
  feed_forward_in_weights.init_normal(fan_in_stddev(embedding_dim));
  feed_forward_in_bias.init_zeros();
  feed_forward_out_weights.init_normal(fan_in_stddev(feed_forward_dim));
  feed_forward_out_bias.init_zeros();
  feed_forward_norm_gain.init_ones();
  feed_forward_norm_bias.init_zeros();
}

/// Reset every decoder-block gradient buffer to zero.
void Block::zero_grad() {
  attention_query_weights.zero_grad();
  attention_key_weights.zero_grad();
  attention_value_weights.zero_grad();
  attention_output_projection_weights.zero_grad();
  attention_norm_gain.zero_grad();
  attention_norm_bias.zero_grad();
  feed_forward_in_weights.zero_grad();
  feed_forward_in_bias.zero_grad();
  feed_forward_out_weights.zero_grad();
  feed_forward_out_bias.zero_grad();
  feed_forward_norm_gain.zero_grad();
  feed_forward_norm_bias.zero_grad();
}

/// Scale every decoder-block gradient buffer by one constant.
void Block::scale_grads(float scale) {
  attention_query_weights.scale_grad(scale);
  attention_key_weights.scale_grad(scale);
  attention_value_weights.scale_grad(scale);
  attention_output_projection_weights.scale_grad(scale);
  attention_norm_gain.scale_grad(scale);
  attention_norm_bias.scale_grad(scale);
  feed_forward_in_weights.scale_grad(scale);
  feed_forward_in_bias.scale_grad(scale);
  feed_forward_out_weights.scale_grad(scale);
  feed_forward_out_bias.scale_grad(scale);
  feed_forward_norm_gain.scale_grad(scale);
  feed_forward_norm_bias.scale_grad(scale);
}

/// Apply one optimizer step to every decoder-block parameter tensor.
void Block::update() {
  attention_query_weights.update();
  attention_key_weights.update();
  attention_value_weights.update();
  attention_output_projection_weights.update();
  attention_norm_gain.update();
  attention_norm_bias.update();
  feed_forward_in_weights.update();
  feed_forward_in_bias.update();
  feed_forward_out_weights.update();
  feed_forward_out_bias.update();
  feed_forward_norm_gain.update();
  feed_forward_norm_bias.update();
}

/// Run one full decoder-block forward pass.
Cache Block::forward(const std::vector<float> &block_input) const {
  Cache cache{.block_input = block_input,
              .attention_layer_norm = layer_norm::forward(block_input, attention_norm_gain,
                                                          attention_norm_bias),
              .attention = {},
              .attention_residual = std::vector<float>(batch_size * context_len * embedding_dim),
              .feed_forward_layer_norm = {},
              .feed_forward = {},
              .block_output = std::vector<float>(batch_size * context_len * embedding_dim)};
  cache.attention = attention::forward(cache.attention_layer_norm.layer_norm_output,
                                       attention_query_weights, attention_key_weights,
                                       attention_value_weights,
                                       attention_output_projection_weights);
  for (size_t i = 0; i < cache.attention_residual.size(); ++i) {
    cache.attention_residual[i] = block_input[i] + cache.attention.projected_output[i];
  }
  cache.feed_forward_layer_norm =
      layer_norm::forward(cache.attention_residual, feed_forward_norm_gain, feed_forward_norm_bias);
  cache.feed_forward = feed_forward::forward(cache.feed_forward_layer_norm.layer_norm_output,
                                             feed_forward_in_weights, feed_forward_in_bias,
                                             feed_forward_out_weights, feed_forward_out_bias);
  for (size_t i = 0; i < cache.block_output.size(); ++i) {
    cache.block_output[i] = cache.attention_residual[i] + cache.feed_forward.projected_output[i];
  }
  return cache;
}

/// Backpropagate through one full decoder block.
std::vector<float> Block::backward(const Cache &cache, const std::vector<float> &d_block_output) {
  std::vector<float> d_attention_residual = d_block_output;
  const std::vector<float> d_feed_forward_norm_output = feed_forward::backward(
      cache.feed_forward_layer_norm.layer_norm_output, cache.feed_forward, d_block_output,
      feed_forward_in_weights, feed_forward_in_bias, feed_forward_out_weights,
      feed_forward_out_bias);
  const std::vector<float> d_attention_residual_from_norm =
      layer_norm::backward(d_feed_forward_norm_output, cache.feed_forward_layer_norm,
                           feed_forward_norm_gain, feed_forward_norm_bias);
  for (size_t i = 0; i < d_attention_residual.size(); ++i) {
    d_attention_residual[i] += d_attention_residual_from_norm[i];
  }

  std::vector<float> d_block_input = d_attention_residual;
  const std::vector<float> d_attention_norm_output =
      attention::backward(cache.attention_layer_norm.layer_norm_output, cache.attention,
                          d_attention_residual, attention_query_weights, attention_key_weights,
                          attention_value_weights, attention_output_projection_weights);
  const std::vector<float> d_block_input_from_norm =
      layer_norm::backward(d_attention_norm_output, cache.attention_layer_norm,
                           attention_norm_gain, attention_norm_bias);
  for (size_t i = 0; i < d_block_input.size(); ++i) {
    d_block_input[i] += d_block_input_from_norm[i];
  }

  return d_block_input;
}

} // namespace decoder_block
