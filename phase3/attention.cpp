/// Multi-head attention forward and backward helpers for phase 3.

#include "attention.h"

#include <limits>

namespace attention {

/// Return the flat index for one Q/K/V element.
size_t qkv_index(size_t batch_index, size_t token_index, size_t head_index,
                 size_t channel_index) {
  return (((batch_index * context_len + token_index) * num_heads + head_index) * head_dim) +
         channel_index;
}

/// Return the flat index for one attention weight.
size_t weight_index(size_t batch_index, size_t head_index, size_t query_index,
                    size_t key_index) {
  return (((batch_index * num_heads + head_index) * context_len + query_index) * context_len) +
         key_index;
}

/// Return the flat index for one concatenated attended-value element.
size_t attended_index(size_t batch_index, size_t token_index, size_t head_index,
                      size_t channel_index) {
  return qkv_index(batch_index, token_index, head_index, channel_index);
}

/// Run the full multi-head attention sublayer with its skip connection.
Cache forward(const std::vector<float> &inputs, const Param &query_weights,
              const Param &key_weights, const Param &value_weights,
              const Param &output_projection_weights) {
  Cache cache;
  cache.queries.resize(batch_size * context_len * attention_dim);
  cache.keys.resize(batch_size * context_len * attention_dim);
  cache.values.resize(batch_size * context_len * attention_dim);
  cache.attention_weights.resize(batch_size * num_heads * context_len * context_len);
  cache.attended_values.assign(batch_size * context_len * attention_dim, 0.0f);
  cache.projected_output.assign(batch_size * context_len * embedding_dim, 0.0f);
  cache.residual_output.resize(batch_size * context_len * embedding_dim);

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t t = 0; t < context_len; ++t) {
      const size_t input_base = b * context_len * embedding_dim + t * embedding_dim;
      const size_t qkv_base = b * context_len * attention_dim + t * attention_dim;

      for (size_t i = 0; i < attention_dim; ++i) {
        float q = 0.0f;
        float k = 0.0f;
        float v = 0.0f;

        for (size_t j = 0; j < embedding_dim; ++j) {
          const float x = inputs[input_base + j];
          q += x * query_weights.val[j * attention_dim + i];
          k += x * key_weights.val[j * attention_dim + i];
          v += x * value_weights.val[j * attention_dim + i];
        }

        cache.queries[qkv_base + i] = q;
        cache.keys[qkv_base + i] = k;
        cache.values[qkv_base + i] = v;
      }
    }
  }

  const float inv_sqrt_head_dim = 1.0f / std::sqrt(static_cast<float>(head_dim));
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t h = 0; h < num_heads; ++h) {
      for (size_t q = 0; q < context_len; ++q) {
        const size_t row_base = weight_index(b, h, q, 0);

        for (size_t k = 0; k < context_len; ++k) {
          if (k > q) {
            cache.attention_weights[row_base + k] = -std::numeric_limits<float>::infinity();
            continue;
          }

          float score = 0.0f;
          for (size_t c = 0; c < head_dim; ++c) {
            score += cache.queries[qkv_index(b, q, h, c)] * cache.keys[qkv_index(b, k, h, c)];
          }
          cache.attention_weights[row_base + k] = score * inv_sqrt_head_dim;
        }

        float max_score = cache.attention_weights[row_base];
        for (size_t k = 1; k < context_len; ++k) {
          max_score = std::max(max_score, cache.attention_weights[row_base + k]);
        }

        double sum_exp = 0.0;
        for (size_t k = 0; k < context_len; ++k) {
          sum_exp +=
              std::exp(static_cast<double>(cache.attention_weights[row_base + k] - max_score));
        }

        for (size_t k = 0; k < context_len; ++k) {
          cache.attention_weights[row_base + k] = static_cast<float>(
              std::exp(static_cast<double>(cache.attention_weights[row_base + k] - max_score)) /
              sum_exp);
        }
      }
    }
  }

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t h = 0; h < num_heads; ++h) {
      for (size_t q = 0; q < context_len; ++q) {
        for (size_t c = 0; c < head_dim; ++c) {
          for (size_t k = 0; k < context_len; ++k) {
            cache.attended_values[attended_index(b, q, h, c)] +=
                cache.attention_weights[weight_index(b, h, q, k)] * cache.values[qkv_index(b, k, h, c)];
          }
        }
      }
    }
  }

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t t = 0; t < context_len; ++t) {
      const size_t attended_base = b * context_len * attention_dim + t * attention_dim;
      const size_t output_base = b * context_len * embedding_dim + t * embedding_dim;

      for (size_t i = 0; i < embedding_dim; ++i) {
        for (size_t j = 0; j < attention_dim; ++j) {
          cache.projected_output[output_base + i] +=
              cache.attended_values[attended_base + j] * output_projection_weights.val[j * embedding_dim + i];
        }
        cache.residual_output[output_base + i] =
            inputs[output_base + i] + cache.projected_output[output_base + i];
      }
    }
  }

  return cache;
}

/// Backpropagate through the full multi-head attention sublayer.
std::vector<float> backward(const std::vector<float> &inputs, const Cache &cache,
                            const std::vector<float> &d_residual, Param &query_weights,
                            Param &key_weights, Param &value_weights,
                            Param &output_projection_weights) {
  std::vector<float> d_inputs = d_residual;
  std::vector<float> d_attended_values(batch_size * context_len * attention_dim, 0.0f);

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t t = 0; t < context_len; ++t) {
      const size_t attended_base = b * context_len * attention_dim + t * attention_dim;
      const size_t output_base = b * context_len * embedding_dim + t * embedding_dim;

      for (size_t i = 0; i < embedding_dim; ++i) {
        const float grad = d_residual[output_base + i];
        for (size_t j = 0; j < attention_dim; ++j) {
          output_projection_weights.grad[j * embedding_dim + i] +=
              cache.attended_values[attended_base + j] * grad;
          d_attended_values[attended_base + j] +=
              grad * output_projection_weights.val[j * embedding_dim + i];
        }
      }
    }
  }

  std::vector<float> d_attention_weights(batch_size * num_heads * context_len * context_len, 0.0f);
  std::vector<float> d_values(batch_size * context_len * attention_dim, 0.0f);
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t h = 0; h < num_heads; ++h) {
      for (size_t q = 0; q < context_len; ++q) {
        for (size_t c = 0; c < head_dim; ++c) {
          const float grad = d_attended_values[attended_index(b, q, h, c)];
          for (size_t k = 0; k < context_len; ++k) {
            d_attention_weights[weight_index(b, h, q, k)] +=
                grad * cache.values[qkv_index(b, k, h, c)];
            d_values[qkv_index(b, k, h, c)] +=
                cache.attention_weights[weight_index(b, h, q, k)] * grad;
          }
        }
      }
    }
  }

  std::vector<float> d_scores(batch_size * num_heads * context_len * context_len, 0.0f);
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t h = 0; h < num_heads; ++h) {
      for (size_t q = 0; q < context_len; ++q) {
        float dot = 0.0f;
        for (size_t k = 0; k < context_len; ++k) {
          dot += d_attention_weights[weight_index(b, h, q, k)] *
                 cache.attention_weights[weight_index(b, h, q, k)];
        }

        for (size_t k = 0; k < context_len; ++k) {
          d_scores[weight_index(b, h, q, k)] =
              cache.attention_weights[weight_index(b, h, q, k)] *
              (d_attention_weights[weight_index(b, h, q, k)] - dot);
        }
      }
    }
  }

  std::vector<float> d_queries(batch_size * context_len * attention_dim, 0.0f);
  std::vector<float> d_keys(batch_size * context_len * attention_dim, 0.0f);
  const float inv_sqrt_head_dim = 1.0f / std::sqrt(static_cast<float>(head_dim));
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t h = 0; h < num_heads; ++h) {
      for (size_t q = 0; q < context_len; ++q) {
        for (size_t k = 0; k <= q; ++k) {
          const float grad = d_scores[weight_index(b, h, q, k)] * inv_sqrt_head_dim;
          for (size_t c = 0; c < head_dim; ++c) {
            d_queries[qkv_index(b, q, h, c)] += grad * cache.keys[qkv_index(b, k, h, c)];
            d_keys[qkv_index(b, k, h, c)] += grad * cache.queries[qkv_index(b, q, h, c)];
          }
        }
      }
    }
  }

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t t = 0; t < context_len; ++t) {
      const size_t input_base = b * context_len * embedding_dim + t * embedding_dim;
      const size_t qkv_base = b * context_len * attention_dim + t * attention_dim;

      for (size_t i = 0; i < embedding_dim; ++i) {
        const float x = inputs[input_base + i];
        for (size_t j = 0; j < attention_dim; ++j) {
          const float dq = d_queries[qkv_base + j];
          const float dk = d_keys[qkv_base + j];
          const float dv = d_values[qkv_base + j];

          query_weights.grad[i * attention_dim + j] += x * dq;
          key_weights.grad[i * attention_dim + j] += x * dk;
          value_weights.grad[i * attention_dim + j] += x * dv;

          d_inputs[input_base + i] += dq * query_weights.val[i * attention_dim + j];
          d_inputs[input_base + i] += dk * key_weights.val[i * attention_dim + j];
          d_inputs[input_base + i] += dv * value_weights.val[i * attention_dim + j];
        }
      }
    }
  }

  return d_inputs;
}

} // namespace attention
