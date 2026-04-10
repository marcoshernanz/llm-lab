/// Single-head attention forward and backward helpers for phase 3.

#include "attention.h"

#include <limits>

namespace attention {

/// Run the full single-head attention sublayer with its skip connection.
Cache forward(const std::vector<float> &inputs, const Param &query_weights,
              const Param &key_weights, const Param &value_weights, const Param &output_weights) {
  Cache cache;
  cache.queries.resize(batch_size * context_len * head_dim);
  cache.keys.resize(batch_size * context_len * head_dim);
  cache.values.resize(batch_size * context_len * head_dim);
  cache.weights.resize(batch_size * context_len * context_len);
  cache.head.assign(batch_size * context_len * head_dim, 0.0f);
  cache.projected.assign(batch_size * context_len * embedding_dim, 0.0f);
  cache.residual.resize(batch_size * context_len * embedding_dim);

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t c = 0; c < context_len; ++c) {
      const size_t in_base = b * context_len * embedding_dim + c * embedding_dim;
      const size_t out_base = b * context_len * head_dim + c * head_dim;

      for (size_t i = 0; i < head_dim; ++i) {
        float q = 0.0f;
        float k = 0.0f;
        float v = 0.0f;

        for (size_t j = 0; j < embedding_dim; ++j) {
          const float x = inputs[in_base + j];
          q += x * query_weights.val[j * head_dim + i];
          k += x * key_weights.val[j * head_dim + i];
          v += x * value_weights.val[j * head_dim + i];
        }

        cache.queries[out_base + i] = q;
        cache.keys[out_base + i] = k;
        cache.values[out_base + i] = v;
      }
    }
  }

  const float inv_sqrt_head_dim = 1.0f / std::sqrt(static_cast<float>(head_dim));
  for (size_t b = 0; b < batch_size; ++b) {
    const size_t qk_base = b * context_len * head_dim;
    const size_t w_base = b * context_len * context_len;

    for (size_t i = 0; i < context_len; ++i) {
      const size_t q_base = qk_base + i * head_dim;
      const size_t row_base = w_base + i * context_len;

      for (size_t j = 0; j < context_len; ++j) {
        if (j > i) {
          cache.weights[row_base + j] = -std::numeric_limits<float>::infinity();
          continue;
        }

        const size_t k_base = qk_base + j * head_dim;
        float score = 0.0f;
        for (size_t h = 0; h < head_dim; ++h) {
          score += cache.queries[q_base + h] * cache.keys[k_base + h];
        }
        cache.weights[row_base + j] = score * inv_sqrt_head_dim;
      }

      float max_score = cache.weights[row_base];
      for (size_t j = 1; j < context_len; ++j) {
        max_score = std::max(max_score, cache.weights[row_base + j]);
      }

      double sum_exp = 0.0;
      for (size_t j = 0; j < context_len; ++j) {
        sum_exp += std::exp(static_cast<double>(cache.weights[row_base + j] - max_score));
      }

      for (size_t j = 0; j < context_len; ++j) {
        cache.weights[row_base + j] = static_cast<float>(
            std::exp(static_cast<double>(cache.weights[row_base + j] - max_score)) / sum_exp);
      }
    }
  }

  for (size_t b = 0; b < batch_size; ++b) {
    const size_t w_base = b * context_len * context_len;
    const size_t v_base = b * context_len * head_dim;
    const size_t h_base = b * context_len * head_dim;

    for (size_t i = 0; i < context_len; ++i) {
      const size_t row_base = w_base + i * context_len;
      const size_t head_base = h_base + i * head_dim;

      for (size_t h = 0; h < head_dim; ++h) {
        for (size_t j = 0; j < context_len; ++j) {
          cache.head[head_base + h] +=
              cache.weights[row_base + j] * cache.values[v_base + j * head_dim + h];
        }
      }
    }
  }

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t c = 0; c < context_len; ++c) {
      const size_t head_base = b * context_len * head_dim + c * head_dim;
      const size_t out_base = b * context_len * embedding_dim + c * embedding_dim;

      for (size_t i = 0; i < embedding_dim; ++i) {
        for (size_t j = 0; j < head_dim; ++j) {
          cache.projected[out_base + i] +=
              cache.head[head_base + j] * output_weights.val[j * embedding_dim + i];
        }
        cache.residual[out_base + i] = inputs[out_base + i] + cache.projected[out_base + i];
      }
    }
  }

  return cache;
}

/// Backpropagate through the full single-head attention sublayer.
std::vector<float> backward(const std::vector<float> &inputs, const Cache &cache,
                            const std::vector<float> &d_residual, Param &query_weights,
                            Param &key_weights, Param &value_weights, Param &output_weights) {
  std::vector<float> d_inputs = d_residual;
  std::vector<float> d_head(batch_size * context_len * head_dim, 0.0f);

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t c = 0; c < context_len; ++c) {
      const size_t head_base = b * context_len * head_dim + c * head_dim;
      const size_t out_base = b * context_len * embedding_dim + c * embedding_dim;

      for (size_t i = 0; i < embedding_dim; ++i) {
        const float grad = d_residual[out_base + i];
        for (size_t j = 0; j < head_dim; ++j) {
          output_weights.grad[j * embedding_dim + i] += cache.head[head_base + j] * grad;
          d_head[head_base + j] += grad * output_weights.val[j * embedding_dim + i];
        }
      }
    }
  }

  std::vector<float> d_weights(batch_size * context_len * context_len, 0.0f);
  std::vector<float> d_values(batch_size * context_len * head_dim, 0.0f);
  for (size_t b = 0; b < batch_size; ++b) {
    const size_t w_base = b * context_len * context_len;
    const size_t v_base = b * context_len * head_dim;

    for (size_t i = 0; i < context_len; ++i) {
      const size_t row_base = w_base + i * context_len;
      const size_t head_base = b * context_len * head_dim + i * head_dim;

      for (size_t h = 0; h < head_dim; ++h) {
        const float grad = d_head[head_base + h];
        for (size_t j = 0; j < context_len; ++j) {
          d_weights[row_base + j] += grad * cache.values[v_base + j * head_dim + h];
          d_values[v_base + j * head_dim + h] += cache.weights[row_base + j] * grad;
        }
      }
    }
  }

  std::vector<float> d_scores(batch_size * context_len * context_len, 0.0f);
  for (size_t b = 0; b < batch_size; ++b) {
    const size_t w_base = b * context_len * context_len;

    for (size_t i = 0; i < context_len; ++i) {
      const size_t row_base = w_base + i * context_len;
      float dot = 0.0f;
      for (size_t j = 0; j < context_len; ++j) {
        dot += d_weights[row_base + j] * cache.weights[row_base + j];
      }
      for (size_t j = 0; j < context_len; ++j) {
        d_scores[row_base + j] = cache.weights[row_base + j] * (d_weights[row_base + j] - dot);
      }
    }
  }

  std::vector<float> d_queries(batch_size * context_len * head_dim, 0.0f);
  std::vector<float> d_keys(batch_size * context_len * head_dim, 0.0f);
  const float inv_sqrt_head_dim = 1.0f / std::sqrt(static_cast<float>(head_dim));
  for (size_t b = 0; b < batch_size; ++b) {
    const size_t qk_base = b * context_len * head_dim;
    const size_t s_base = b * context_len * context_len;

    for (size_t i = 0; i < context_len; ++i) {
      const size_t q_base = qk_base + i * head_dim;
      const size_t row_base = s_base + i * context_len;

      for (size_t j = 0; j <= i; ++j) {
        const float grad = d_scores[row_base + j] * inv_sqrt_head_dim;
        const size_t k_base = qk_base + j * head_dim;

        for (size_t h = 0; h < head_dim; ++h) {
          d_queries[q_base + h] += grad * cache.keys[k_base + h];
          d_keys[k_base + h] += grad * cache.queries[q_base + h];
        }
      }
    }
  }

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t c = 0; c < context_len; ++c) {
      const size_t in_base = b * context_len * embedding_dim + c * embedding_dim;
      const size_t qkv_base = b * context_len * head_dim + c * head_dim;

      for (size_t i = 0; i < embedding_dim; ++i) {
        const float x = inputs[in_base + i];
        for (size_t j = 0; j < head_dim; ++j) {
          const float dq = d_queries[qkv_base + j];
          const float dk = d_keys[qkv_base + j];
          const float dv = d_values[qkv_base + j];

          query_weights.grad[i * head_dim + j] += x * dq;
          key_weights.grad[i * head_dim + j] += x * dk;
          value_weights.grad[i * head_dim + j] += x * dv;

          d_inputs[in_base + i] += dq * query_weights.val[i * head_dim + j];
          d_inputs[in_base + i] += dk * key_weights.val[i * head_dim + j];
          d_inputs[in_base + i] += dv * value_weights.val[i * head_dim + j];
        }
      }
    }
  }

  return d_inputs;
}

} // namespace attention
