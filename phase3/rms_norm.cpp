/// RMSNorm forward and backward helpers for phase 3.

#include "rms_norm.h"

namespace rms_norm {

/// Run one RMSNorm over the embedding dimension.
Cache forward(const std::vector<float> &inputs, const Param &gain) {
  Cache cache;
  cache.normalized_input.resize(batch_size * context_len * embedding_dim);
  cache.rms_norm_output.resize(batch_size * context_len * embedding_dim);
  cache.inv_rms.resize(batch_size * context_len);

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t c = 0; c < context_len; ++c) {
      const size_t row_base = b * context_len * embedding_dim + c * embedding_dim;
      const size_t norm_index = b * context_len + c;

      float mean_square = 0.0f;
      for (size_t i = 0; i < embedding_dim; ++i) {
        mean_square += inputs[row_base + i] * inputs[row_base + i];
      }
      mean_square /= static_cast<float>(embedding_dim);

      const float inv_rms_value = 1.0f / std::sqrt(mean_square + rms_norm_eps);
      cache.inv_rms[norm_index] = inv_rms_value;

      for (size_t i = 0; i < embedding_dim; ++i) {
        const float normalized = inputs[row_base + i] * inv_rms_value;
        cache.normalized_input[row_base + i] = normalized;
        cache.rms_norm_output[row_base + i] = gain.val[i] * normalized;
      }
    }
  }

  return cache;
}

/// Backpropagate through one RMSNorm application.
std::vector<float> backward(const std::vector<float> &d_output, const std::vector<float> &inputs,
                            const Cache &cache, Param &gain) {
  std::vector<float> d_inputs(batch_size * context_len * embedding_dim, 0.0f);

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t c = 0; c < context_len; ++c) {
      const size_t row_base = b * context_len * embedding_dim + c * embedding_dim;
      const size_t norm_index = b * context_len + c;
      const float inv_rms_value = cache.inv_rms[norm_index];

      float dot = 0.0f;
      for (size_t i = 0; i < embedding_dim; ++i) {
        const float d_normalized = d_output[row_base + i] * gain.val[i];
        gain.grad[i] += d_output[row_base + i] * cache.normalized_input[row_base + i];
        dot += d_normalized * inputs[row_base + i];
      }

      const float correction = dot * inv_rms_value * inv_rms_value * inv_rms_value /
                               static_cast<float>(embedding_dim);
      for (size_t i = 0; i < embedding_dim; ++i) {
        const float d_normalized = d_output[row_base + i] * gain.val[i];
        d_inputs[row_base + i] = inv_rms_value * d_normalized - inputs[row_base + i] * correction;
      }
    }
  }

  return d_inputs;
}

} // namespace rms_norm
