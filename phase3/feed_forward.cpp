/// Feedforward forward and backward helpers for phase 3.

#include "feed_forward.h"
#include "profiler.h"

namespace feed_forward {

/// Run the feedforward sublayer without the skip connection.
Cache forward(const std::vector<float> &inputs, const Param &hidden_weights,
              const Param &hidden_bias, const Param &output_projection_weights,
              const Param &output_projection_bias) {
  const profiler::Scope scope("feed_forward.forward");
  Cache cache;
  cache.hidden_pre.resize(batch_size * context_len * feed_forward_dim);
  cache.hidden.resize(batch_size * context_len * feed_forward_dim);
  cache.projected_output.assign(batch_size * context_len * embedding_dim, 0.0f);

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t c = 0; c < context_len; ++c) {
      const size_t in_base = b * context_len * embedding_dim + c * embedding_dim;
      const size_t hidden_base = b * context_len * feed_forward_dim + c * feed_forward_dim;

      for (size_t i = 0; i < feed_forward_dim; ++i) {
        float hidden = hidden_bias.val[i];
        for (size_t j = 0; j < embedding_dim; ++j) {
          hidden += inputs[in_base + j] * hidden_weights.val[j * feed_forward_dim + i];
        }
        cache.hidden_pre[hidden_base + i] = hidden;
        cache.hidden[hidden_base + i] = std::tanh(hidden);
      }
    }
  }

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t c = 0; c < context_len; ++c) {
      const size_t hidden_base = b * context_len * feed_forward_dim + c * feed_forward_dim;
      const size_t out_base = b * context_len * embedding_dim + c * embedding_dim;

      for (size_t i = 0; i < embedding_dim; ++i) {
        float output = output_projection_bias.val[i];
        for (size_t j = 0; j < feed_forward_dim; ++j) {
          output +=
              cache.hidden[hidden_base + j] * output_projection_weights.val[j * embedding_dim + i];
        }
        cache.projected_output[out_base + i] = output;
      }
    }
  }

  return cache;
}

/// Backpropagate through the feedforward sublayer and its skip path.
std::vector<float> backward(const std::vector<float> &inputs, const Cache &cache,
                            const std::vector<float> &d_projected_output, Param &hidden_weights,
                            Param &hidden_bias, Param &output_projection_weights,
                            Param &output_projection_bias) {
  const profiler::Scope scope("feed_forward.backward");
  std::vector<float> d_inputs(batch_size * context_len * embedding_dim, 0.0f);
  std::vector<float> d_hidden(batch_size * context_len * feed_forward_dim, 0.0f);

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t c = 0; c < context_len; ++c) {
      const size_t hidden_base = b * context_len * feed_forward_dim + c * feed_forward_dim;
      const size_t out_base = b * context_len * embedding_dim + c * embedding_dim;

      for (size_t i = 0; i < embedding_dim; ++i) {
        const float grad = d_projected_output[out_base + i];
        output_projection_bias.grad[i] += grad;
        for (size_t j = 0; j < feed_forward_dim; ++j) {
          output_projection_weights.grad[j * embedding_dim + i] +=
              cache.hidden[hidden_base + j] * grad;
          d_hidden[hidden_base + j] +=
              grad * output_projection_weights.val[j * embedding_dim + i];
        }
      }
    }
  }

  for (size_t i = 0; i < d_hidden.size(); ++i) {
    d_hidden[i] *= (1.0f - cache.hidden[i] * cache.hidden[i]);
  }

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t c = 0; c < context_len; ++c) {
      const size_t hidden_base = b * context_len * feed_forward_dim + c * feed_forward_dim;
      const size_t in_base = b * context_len * embedding_dim + c * embedding_dim;

      for (size_t i = 0; i < feed_forward_dim; ++i) {
        const float grad = d_hidden[hidden_base + i];
        hidden_bias.grad[i] += grad;
        for (size_t j = 0; j < embedding_dim; ++j) {
          hidden_weights.grad[j * feed_forward_dim + i] += inputs[in_base + j] * grad;
          d_inputs[in_base + j] += grad * hidden_weights.val[j * feed_forward_dim + i];
        }
      }
    }
  }

  return d_inputs;
}

} // namespace feed_forward
