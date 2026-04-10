/// LayerNorm forward and backward helpers for phase 3.

#include "layer_norm.h"

namespace layer_norm {

/// Run one LayerNorm over the embedding dimension.
Cache forward(const std::vector<float> &inputs, const Param &scale, const Param &shift) {
  Cache cache;
  cache.normalized_input.resize(batch_size * context_len * embedding_dim);
  cache.layer_norm_output.resize(batch_size * context_len * embedding_dim);
  cache.inv_std.resize(batch_size * context_len);

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t c = 0; c < context_len; ++c) {
      const size_t row_base = b * context_len * embedding_dim + c * embedding_dim;
      const size_t norm_index = b * context_len + c;

      float mean = 0.0f;
      for (size_t i = 0; i < embedding_dim; ++i) {
        mean += inputs[row_base + i];
      }
      mean /= static_cast<float>(embedding_dim);

      float variance = 0.0f;
      for (size_t i = 0; i < embedding_dim; ++i) {
        const float centered = inputs[row_base + i] - mean;
        variance += centered * centered;
      }
      variance /= static_cast<float>(embedding_dim);

      const float inv_std = 1.0f / std::sqrt(variance + layer_norm_eps);
      cache.inv_std[norm_index] = inv_std;

      for (size_t i = 0; i < embedding_dim; ++i) {
        const float normalized = (inputs[row_base + i] - mean) * inv_std;
        cache.normalized_input[row_base + i] = normalized;
        cache.layer_norm_output[row_base + i] = scale.val[i] * normalized + shift.val[i];
      }
    }
  }

  return cache;
}

/// Backpropagate through one LayerNorm application.
std::vector<float> backward(const std::vector<float> &d_output, const Cache &cache, Param &scale,
                            Param &shift) {
  std::vector<float> d_inputs(batch_size * context_len * embedding_dim, 0.0f);

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t c = 0; c < context_len; ++c) {
      const size_t row_base = b * context_len * embedding_dim + c * embedding_dim;
      const size_t norm_index = b * context_len + c;
      const float inv_std = cache.inv_std[norm_index];

      float sum_dxhat = 0.0f;
      float sum_dxhat_xhat = 0.0f;
      for (size_t i = 0; i < embedding_dim; ++i) {
        const float dxhat = d_output[row_base + i] * scale.val[i];
        sum_dxhat += dxhat;
        sum_dxhat_xhat += dxhat * cache.normalized_input[row_base + i];
        scale.grad[i] += d_output[row_base + i] * cache.normalized_input[row_base + i];
        shift.grad[i] += d_output[row_base + i];
      }

      for (size_t i = 0; i < embedding_dim; ++i) {
        const float dxhat = d_output[row_base + i] * scale.val[i];
        d_inputs[row_base + i] = inv_std *
                                 (static_cast<float>(embedding_dim) * dxhat - sum_dxhat -
                                  cache.normalized_input[row_base + i] * sum_dxhat_xhat) /
                                 static_cast<float>(embedding_dim);
      }
    }
  }

  return d_inputs;
}

} // namespace layer_norm
