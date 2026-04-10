/// Minimal phase-3 script for learning manual language-model gradients.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

const std::string corpus_path = "../datasets/tinyshakespeare.txt";
const int vocab_size = 128;
const int context_len = 4;
const int embedding_dim = 32;
const int head_dim = 64;
const int feed_forward_dim = 128;

const int steps = 10000;
const int steps_per_chunk = 100;
const int batch_size = 32;
const float inv_token_count = 1.0f / static_cast<float>(batch_size * context_len);
const float validation_split = 0.1f;

const float learning_rate = 0.01f;
const float beta1 = 0.9f;
const float beta2 = 0.999f;
const float eps = 1e-8f;
const float layer_norm_eps = 1e-5f;

std::unordered_map<char, int> char_to_id;

/// Return the shared random generator for reproducible experiments.
std::mt19937 &rng() {
  static std::mt19937 gen(0);
  return gen;
}

/// Sample one normal random value for parameter initialization.
float randn() {
  std::normal_distribution<float> dist(0.0f, 1.0f);
  return dist(rng());
}

/// Sample one integer in the half-open range [min, max).
int randint(int min, int max) {
  std::uniform_int_distribution<int> dist(min, max - 1);
  return dist(rng());
}

/// Load the training text from disk.
std::string load_corpus() {
  std::ifstream file(corpus_path);
  if (!file) {
    throw std::runtime_error("could not open corpus file");
  }
  return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

/// Build a tiny character vocabulary and return the encoded corpus.
std::vector<int> prepare_vocab(const std::string &corpus) {
  char_to_id.clear();
  char_to_id.reserve(vocab_size);
  std::vector<int> token_ids(corpus.size());
  for (size_t i = 0; i < corpus.size(); ++i) {
    const char c = corpus[i];
    const auto it = char_to_id.find(c);
    if (it != char_to_id.end()) {
      token_ids[i] = it->second;
      continue;
    }
    if (char_to_id.size() >= vocab_size) {
      throw std::runtime_error("vocab_size too small");
    }
    const int id = static_cast<int>(char_to_id.size());
    char_to_id[c] = id;
    token_ids[i] = id;
  }

  return token_ids;
}


/// Hold the intermediate tensors from one forward pass.
struct ForwardCache {
  std::vector<float> embeddings;
  std::vector<float> queries;
  std::vector<float> keys;
  std::vector<float> values;
  std::vector<float> attention;
  std::vector<float> head;
  std::vector<float> projected_attention;
  std::vector<float> residual;
  std::vector<float> normalized_residual;
  std::vector<float> layer_norm_output;
  std::vector<float> layer_norm_inv_std;
  std::vector<float> feed_forward_hidden_pre;
  std::vector<float> feed_forward_hidden;
  std::vector<float> feed_forward_output;
  std::vector<float> transformer;
  std::vector<float> normalized_transformer;
  std::vector<float> transformer_output;
  std::vector<float> transformer_inv_std;
  std::vector<float> logits;
  std::vector<float> probs;
  float avg_loss;
};

/// Hold Adam state for one parameter tensor.
class Adam {
public:
  float beta1_pow = 1.0f;
  float beta2_pow = 1.0f;
  std::vector<float> first_moment;
  std::vector<float> second_moment;

  /// Construct one Adam state object with zero moments.
  explicit Adam(size_t size) : first_moment(size, 0.0f), second_moment(size, 0.0f) {}

  /// Apply one Adam update to a parameter tensor.
  void update(std::vector<float> &values, const std::vector<float> &gradients) {
    beta1_pow *= beta1;
    beta2_pow *= beta2;
    const float beta1_correction = 1.0f - beta1_pow;
    const float beta2_correction = 1.0f - beta2_pow;

    for (size_t i = 0; i < values.size(); ++i) {
      first_moment[i] = beta1 * first_moment[i] + (1.0f - beta1) * gradients[i];
      second_moment[i] = beta2 * second_moment[i] + (1.0f - beta2) * gradients[i] * gradients[i];

      const float corrected_first = first_moment[i] / beta1_correction;
      const float corrected_second = second_moment[i] / beta2_correction;
      values[i] -= learning_rate * corrected_first / (std::sqrt(corrected_second) + eps);
    }
  }
};

/// Hold one trainable tensor, its gradient, and its optimizer state.
class Param {
public:
  std::vector<float> val;
  std::vector<float> grad;
  Adam optimizer;

  /// Construct one parameter tensor with matching gradient and optimizer state.
  explicit Param(size_t size) : val(size), grad(size, 0.0f), optimizer(size) {}

  /// Fill the parameter tensor with random values.
  void init_randn() {
    for (float &x : val) {
      x = randn();
    }
  }

  /// Fill the parameter tensor with zeros.
  void init_zeros() { std::fill(val.begin(), val.end(), 0.0f); }

  /// Fill the parameter tensor with ones.
  void init_ones() { std::fill(val.begin(), val.end(), 1.0f); }

  /// Reset the gradient buffer to zero.
  void zero_grad() { std::fill(grad.begin(), grad.end(), 0.0f); }

  /// Scale the gradient buffer by one constant.
  void scale_grad(float scale) {
    for (float &x : grad) {
      x *= scale;
    }
  }

  /// Apply one optimizer step using the current gradient buffer.
  void update() { optimizer.update(val, grad); }
};

/// Hold the trainable tensors for the tiny language model.
class Model {
public:
  Param token_embeddings;
  Param position_embeddings;
  Param query_weights;
  Param key_weights;
  Param value_weights;
  Param attention_output_weights;
  Param layer_norm_scale;
  Param layer_norm_shift;
  Param feed_forward_hidden_weights;
  Param feed_forward_hidden_bias;
  Param feed_forward_output_weights;
  Param feed_forward_output_bias;
  Param feed_forward_norm_scale;
  Param feed_forward_norm_shift;
  Param logit_weights;
  Param output_bias;

  /// Construct one model with correctly sized parameter tensors.
  Model()
      : token_embeddings(vocab_size * embedding_dim),
        position_embeddings(context_len * embedding_dim), query_weights(embedding_dim * head_dim),
        key_weights(embedding_dim * head_dim), value_weights(embedding_dim * head_dim),
        attention_output_weights(head_dim * embedding_dim), layer_norm_scale(embedding_dim),
        layer_norm_shift(embedding_dim), feed_forward_hidden_weights(embedding_dim * feed_forward_dim),
        feed_forward_hidden_bias(feed_forward_dim),
        feed_forward_output_weights(feed_forward_dim * embedding_dim),
        feed_forward_output_bias(embedding_dim), feed_forward_norm_scale(embedding_dim),
        feed_forward_norm_shift(embedding_dim), logit_weights(embedding_dim * vocab_size),
        output_bias(vocab_size) {}

  /// Initialize one model with random weights and zero biases.
  static Model init() {
    Model model;
    model.token_embeddings.init_randn();
    model.position_embeddings.init_randn();
    model.query_weights.init_randn();
    model.key_weights.init_randn();
    model.value_weights.init_randn();
    model.attention_output_weights.init_randn();
    model.layer_norm_scale.init_ones();
    model.layer_norm_shift.init_zeros();
    model.feed_forward_hidden_weights.init_randn();
    model.feed_forward_hidden_bias.init_zeros();
    model.feed_forward_output_weights.init_randn();
    model.feed_forward_output_bias.init_zeros();
    model.feed_forward_norm_scale.init_ones();
    model.feed_forward_norm_shift.init_zeros();
    model.logit_weights.init_randn();
    model.output_bias.init_zeros();
    return model;
  }

  /// Reset every parameter gradient buffer to zero.
  void zero_grad() {
    token_embeddings.zero_grad();
    position_embeddings.zero_grad();
    query_weights.zero_grad();
    key_weights.zero_grad();
    value_weights.zero_grad();
    attention_output_weights.zero_grad();
    layer_norm_scale.zero_grad();
    layer_norm_shift.zero_grad();
    feed_forward_hidden_weights.zero_grad();
    feed_forward_hidden_bias.zero_grad();
    feed_forward_output_weights.zero_grad();
    feed_forward_output_bias.zero_grad();
    feed_forward_norm_scale.zero_grad();
    feed_forward_norm_shift.zero_grad();
    logit_weights.zero_grad();
    output_bias.zero_grad();
  }

  /// Scale every parameter gradient buffer by one constant.
  void scale_grads(float scale) {
    token_embeddings.scale_grad(scale);
    position_embeddings.scale_grad(scale);
    query_weights.scale_grad(scale);
    key_weights.scale_grad(scale);
    value_weights.scale_grad(scale);
    attention_output_weights.scale_grad(scale);
    layer_norm_scale.scale_grad(scale);
    layer_norm_shift.scale_grad(scale);
    feed_forward_hidden_weights.scale_grad(scale);
    feed_forward_hidden_bias.scale_grad(scale);
    feed_forward_output_weights.scale_grad(scale);
    feed_forward_output_bias.scale_grad(scale);
    feed_forward_norm_scale.scale_grad(scale);
    feed_forward_norm_shift.scale_grad(scale);
    logit_weights.scale_grad(scale);
    output_bias.scale_grad(scale);
  }

  /// Run one full forward pass and keep the tensors needed for backprop.
  ForwardCache forward(const std::vector<int> &ids, const std::vector<int> &targets) const {
    ForwardCache cache{
        .embeddings = std::vector<float>(batch_size * context_len * embedding_dim),
        .queries = std::vector<float>(batch_size * context_len * head_dim),
        .keys = std::vector<float>(batch_size * context_len * head_dim),
        .values = std::vector<float>(batch_size * context_len * head_dim),
        .attention = std::vector<float>(batch_size * context_len * context_len),
        .head = std::vector<float>(batch_size * context_len * head_dim, 0.0f),
        .projected_attention = std::vector<float>(batch_size * context_len * embedding_dim, 0.0f),
        .residual = std::vector<float>(batch_size * context_len * embedding_dim),
        .normalized_residual = std::vector<float>(batch_size * context_len * embedding_dim),
        .layer_norm_output = std::vector<float>(batch_size * context_len * embedding_dim),
        .layer_norm_inv_std = std::vector<float>(batch_size * context_len),
        .feed_forward_hidden_pre =
            std::vector<float>(batch_size * context_len * feed_forward_dim),
        .feed_forward_hidden = std::vector<float>(batch_size * context_len * feed_forward_dim),
        .feed_forward_output = std::vector<float>(batch_size * context_len * embedding_dim),
        .transformer = std::vector<float>(batch_size * context_len * embedding_dim),
        .normalized_transformer = std::vector<float>(batch_size * context_len * embedding_dim),
        .transformer_output = std::vector<float>(batch_size * context_len * embedding_dim),
        .transformer_inv_std = std::vector<float>(batch_size * context_len),
        .logits = std::vector<float>(batch_size * context_len * vocab_size),
        .probs = std::vector<float>(batch_size * context_len * vocab_size),
        .avg_loss = 0.0f,
    };

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t out_base = b * context_len * embedding_dim + c * embedding_dim;
        const size_t token_id = ids[b * context_len + c];
        const size_t tok_base = token_id * embedding_dim;
        const size_t pos_base = c * embedding_dim;

        for (size_t i = 0; i < embedding_dim; ++i) {
          cache.embeddings[out_base + i] =
              token_embeddings.val[tok_base + i] + position_embeddings.val[pos_base + i];
        }
      }
    }

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t emb_base = b * context_len * embedding_dim + c * embedding_dim;
        const size_t out_base = b * context_len * head_dim + c * head_dim;

        for (size_t i = 0; i < head_dim; ++i) {
          float q = 0.0f;
          float k = 0.0f;
          float v = 0.0f;

          for (size_t j = 0; j < embedding_dim; ++j) {
            const float x = cache.embeddings[emb_base + j];
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
      const size_t a_base = b * context_len * context_len;

      for (size_t i = 0; i < context_len; ++i) {
        const size_t q_base = qk_base + i * head_dim;
        const size_t row_base = a_base + i * context_len;

        for (size_t j = 0; j < context_len; ++j) {
          if (j > i) {
            cache.attention[row_base + j] = -std::numeric_limits<float>::infinity();
            continue;
          }

          const size_t k_base = qk_base + j * head_dim;
          float score = 0.0f;
          for (size_t h = 0; h < head_dim; ++h) {
            score += cache.queries[q_base + h] * cache.keys[k_base + h];
          }
          cache.attention[row_base + j] = score * inv_sqrt_head_dim;
        }

        float max_score = cache.attention[row_base];
        for (size_t j = 1; j < context_len; ++j) {
          max_score = std::max(max_score, cache.attention[row_base + j]);
        }

        double sum_exp = 0.0;
        for (size_t j = 0; j < context_len; ++j) {
          sum_exp += std::exp(static_cast<double>(cache.attention[row_base + j] - max_score));
        }

        for (size_t j = 0; j < context_len; ++j) {
          cache.attention[row_base + j] = static_cast<float>(
              std::exp(static_cast<double>(cache.attention[row_base + j] - max_score)) / sum_exp);
        }
      }
    }

    for (size_t b = 0; b < batch_size; ++b) {
      const size_t a_base = b * context_len * context_len;
      const size_t v_base = b * context_len * head_dim;
      const size_t h_base = b * context_len * head_dim;

      for (size_t i = 0; i < context_len; ++i) {
        const size_t att_row = a_base + i * context_len;
        const size_t head_row = h_base + i * head_dim;

        for (size_t h = 0; h < head_dim; ++h) {
          for (size_t j = 0; j < context_len; ++j) {
            cache.head[head_row + h] +=
                cache.attention[att_row + j] * cache.values[v_base + j * head_dim + h];
          }
        }
      }
    }

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t head_base = b * context_len * head_dim + c * head_dim;
        const size_t emb_base = b * context_len * embedding_dim + c * embedding_dim;

        for (size_t i = 0; i < embedding_dim; ++i) {
          float projected = 0.0f;
          for (size_t j = 0; j < head_dim; ++j) {
            projected += cache.head[head_base + j] * attention_output_weights.val[j * embedding_dim + i];
          }
          cache.projected_attention[emb_base + i] = projected;
          cache.residual[emb_base + i] = cache.embeddings[emb_base + i] + projected;
        }
      }
    }

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t residual_base = b * context_len * embedding_dim + c * embedding_dim;
        const size_t norm_index = b * context_len + c;

        float mean = 0.0f;
        for (size_t i = 0; i < embedding_dim; ++i) {
          mean += cache.residual[residual_base + i];
        }
        mean /= static_cast<float>(embedding_dim);

        float variance = 0.0f;
        for (size_t i = 0; i < embedding_dim; ++i) {
          const float centered = cache.residual[residual_base + i] - mean;
          variance += centered * centered;
        }
        variance /= static_cast<float>(embedding_dim);

        const float inv_std = 1.0f / std::sqrt(variance + layer_norm_eps);
        cache.layer_norm_inv_std[norm_index] = inv_std;

        for (size_t i = 0; i < embedding_dim; ++i) {
          const float normalized = (cache.residual[residual_base + i] - mean) * inv_std;
          cache.normalized_residual[residual_base + i] = normalized;
          cache.layer_norm_output[residual_base + i] =
              layer_norm_scale.val[i] * normalized + layer_norm_shift.val[i];
        }
      }
    }

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t emb_base = b * context_len * embedding_dim + c * embedding_dim;
        const size_t hidden_base = b * context_len * feed_forward_dim + c * feed_forward_dim;

        for (size_t i = 0; i < feed_forward_dim; ++i) {
          float hidden = feed_forward_hidden_bias.val[i];
          for (size_t j = 0; j < embedding_dim; ++j) {
            hidden += cache.layer_norm_output[emb_base + j] *
                      feed_forward_hidden_weights.val[j * feed_forward_dim + i];
          }
          cache.feed_forward_hidden_pre[hidden_base + i] = hidden;
          cache.feed_forward_hidden[hidden_base + i] = std::tanh(hidden);
        }
      }
    }

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t hidden_base = b * context_len * feed_forward_dim + c * feed_forward_dim;
        const size_t emb_base = b * context_len * embedding_dim + c * embedding_dim;

        for (size_t i = 0; i < embedding_dim; ++i) {
          float output = feed_forward_output_bias.val[i];
          for (size_t j = 0; j < feed_forward_dim; ++j) {
            output += cache.feed_forward_hidden[hidden_base + j] *
                      feed_forward_output_weights.val[j * embedding_dim + i];
          }
          cache.feed_forward_output[emb_base + i] = output;
          cache.transformer[emb_base + i] = cache.layer_norm_output[emb_base + i] + output;
        }
      }
    }

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t transformer_base = b * context_len * embedding_dim + c * embedding_dim;
        const size_t norm_index = b * context_len + c;

        float mean = 0.0f;
        for (size_t i = 0; i < embedding_dim; ++i) {
          mean += cache.transformer[transformer_base + i];
        }
        mean /= static_cast<float>(embedding_dim);

        float variance = 0.0f;
        for (size_t i = 0; i < embedding_dim; ++i) {
          const float centered = cache.transformer[transformer_base + i] - mean;
          variance += centered * centered;
        }
        variance /= static_cast<float>(embedding_dim);

        const float inv_std = 1.0f / std::sqrt(variance + layer_norm_eps);
        cache.transformer_inv_std[norm_index] = inv_std;

        for (size_t i = 0; i < embedding_dim; ++i) {
          const float normalized = (cache.transformer[transformer_base + i] - mean) * inv_std;
          cache.normalized_transformer[transformer_base + i] = normalized;
          cache.transformer_output[transformer_base + i] =
              feed_forward_norm_scale.val[i] * normalized + feed_forward_norm_shift.val[i];
        }
      }
    }

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t residual_base = b * context_len * embedding_dim + c * embedding_dim;
        const size_t out_base = b * context_len * vocab_size + c * vocab_size;

        for (size_t i = 0; i < vocab_size; ++i) {
          float logit = output_bias.val[i];
          for (size_t j = 0; j < embedding_dim; ++j) {
            logit += cache.transformer_output[residual_base + j] * logit_weights.val[j * vocab_size + i];
          }
          cache.logits[out_base + i] = logit;
        }
      }
    }

    float loss_sum = 0.0f;
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t row_base = b * context_len * vocab_size + c * vocab_size;
        const size_t target = targets[b * context_len + c];

        float max_logit = cache.logits[row_base];
        for (size_t i = 1; i < vocab_size; ++i) {
          max_logit = std::max(max_logit, cache.logits[row_base + i]);
        }

        double sum_exp = 0.0;
        for (size_t i = 0; i < vocab_size; ++i) {
          sum_exp += std::exp(static_cast<double>(cache.logits[row_base + i] - max_logit));
        }

        for (size_t i = 0; i < vocab_size; ++i) {
          cache.probs[row_base + i] = static_cast<float>(
              std::exp(static_cast<double>(cache.logits[row_base + i] - max_logit)) / sum_exp);
        }

        loss_sum += static_cast<float>(max_logit + std::log(sum_exp) - cache.logits[row_base + target]);
      }
    }

    cache.avg_loss = loss_sum * inv_token_count;
    return cache;
  }

  /// Run one full forward and backward pass for one batch.
  float forward_backward(const std::vector<int> &ids, const std::vector<int> &targets) {
    zero_grad();

    const ForwardCache cache = forward(ids, targets);

    std::vector<float> d_logits = cache.probs;
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        d_logits[b * context_len * vocab_size + c * vocab_size + targets[b * context_len + c]] -= 1.0f;
      }
    }

    std::vector<float> d_transformer_output(batch_size * context_len * embedding_dim, 0.0f);
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t residual_base = b * context_len * embedding_dim + c * embedding_dim;
        const size_t logit_base = b * context_len * vocab_size + c * vocab_size;

        for (size_t i = 0; i < vocab_size; ++i) {
          const float grad = d_logits[logit_base + i];
          output_bias.grad[i] += grad;
          for (size_t j = 0; j < embedding_dim; ++j) {
            logit_weights.grad[j * vocab_size + i] += cache.transformer_output[residual_base + j] * grad;
            d_transformer_output[residual_base + j] += grad * logit_weights.val[j * vocab_size + i];
          }
        }
      }
    }

    std::vector<float> d_transformer(batch_size * context_len * embedding_dim, 0.0f);
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t transformer_base = b * context_len * embedding_dim + c * embedding_dim;
        const size_t norm_index = b * context_len + c;
        const float inv_std = cache.transformer_inv_std[norm_index];

        float sum_dxhat = 0.0f;
        float sum_dxhat_xhat = 0.0f;
        for (size_t i = 0; i < embedding_dim; ++i) {
          const float dxhat = d_transformer_output[transformer_base + i] * feed_forward_norm_scale.val[i];
          sum_dxhat += dxhat;
          sum_dxhat_xhat += dxhat * cache.normalized_transformer[transformer_base + i];
          feed_forward_norm_scale.grad[i] +=
              d_transformer_output[transformer_base + i] * cache.normalized_transformer[transformer_base + i];
          feed_forward_norm_shift.grad[i] += d_transformer_output[transformer_base + i];
        }

        for (size_t i = 0; i < embedding_dim; ++i) {
          const float dxhat = d_transformer_output[transformer_base + i] * feed_forward_norm_scale.val[i];
          d_transformer[transformer_base + i] =
              inv_std *
              (static_cast<float>(embedding_dim) * dxhat - sum_dxhat -
               cache.normalized_transformer[transformer_base + i] * sum_dxhat_xhat) /
              static_cast<float>(embedding_dim);
        }
      }
    }

    std::vector<float> d_layer_norm_output(batch_size * context_len * embedding_dim, 0.0f);
    for (size_t i = 0; i < d_layer_norm_output.size(); ++i) {
      d_layer_norm_output[i] = d_transformer[i];
    }

    std::vector<float> d_feed_forward_output(batch_size * context_len * embedding_dim, 0.0f);
    for (size_t i = 0; i < d_feed_forward_output.size(); ++i) {
      d_feed_forward_output[i] = d_transformer[i];
    }

    std::vector<float> d_feed_forward_hidden(batch_size * context_len * feed_forward_dim, 0.0f);
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t hidden_base = b * context_len * feed_forward_dim + c * feed_forward_dim;
        const size_t emb_base = b * context_len * embedding_dim + c * embedding_dim;

        for (size_t i = 0; i < embedding_dim; ++i) {
          const float grad = d_feed_forward_output[emb_base + i];
          feed_forward_output_bias.grad[i] += grad;
          for (size_t j = 0; j < feed_forward_dim; ++j) {
            feed_forward_output_weights.grad[j * embedding_dim + i] +=
                cache.feed_forward_hidden[hidden_base + j] * grad;
            d_feed_forward_hidden[hidden_base + j] +=
                grad * feed_forward_output_weights.val[j * embedding_dim + i];
          }
        }
      }
    }

    std::vector<float> d_feed_forward_hidden_pre(batch_size * context_len * feed_forward_dim, 0.0f);
    for (size_t i = 0; i < d_feed_forward_hidden_pre.size(); ++i) {
      d_feed_forward_hidden_pre[i] = d_feed_forward_hidden[i] *
                                     (1.0f - cache.feed_forward_hidden[i] * cache.feed_forward_hidden[i]);
    }

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t hidden_base = b * context_len * feed_forward_dim + c * feed_forward_dim;
        const size_t emb_base = b * context_len * embedding_dim + c * embedding_dim;

        for (size_t i = 0; i < feed_forward_dim; ++i) {
          const float grad = d_feed_forward_hidden_pre[hidden_base + i];
          feed_forward_hidden_bias.grad[i] += grad;
          for (size_t j = 0; j < embedding_dim; ++j) {
            feed_forward_hidden_weights.grad[j * feed_forward_dim + i] +=
                cache.layer_norm_output[emb_base + j] * grad;
            d_layer_norm_output[emb_base + j] +=
                grad * feed_forward_hidden_weights.val[j * feed_forward_dim + i];
          }
        }
      }
    }

    std::vector<float> d_residual(batch_size * context_len * embedding_dim, 0.0f);
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t residual_base = b * context_len * embedding_dim + c * embedding_dim;
        const size_t norm_index = b * context_len + c;
        const float inv_std = cache.layer_norm_inv_std[norm_index];

        float sum_dxhat = 0.0f;
        float sum_dxhat_xhat = 0.0f;
        for (size_t i = 0; i < embedding_dim; ++i) {
          const float dxhat = d_layer_norm_output[residual_base + i] * layer_norm_scale.val[i];
          sum_dxhat += dxhat;
          sum_dxhat_xhat += dxhat * cache.normalized_residual[residual_base + i];
          layer_norm_scale.grad[i] +=
              d_layer_norm_output[residual_base + i] * cache.normalized_residual[residual_base + i];
          layer_norm_shift.grad[i] += d_layer_norm_output[residual_base + i];
        }

        for (size_t i = 0; i < embedding_dim; ++i) {
          const float dxhat = d_layer_norm_output[residual_base + i] * layer_norm_scale.val[i];
          d_residual[residual_base + i] =
              inv_std *
              (static_cast<float>(embedding_dim) * dxhat - sum_dxhat -
               cache.normalized_residual[residual_base + i] * sum_dxhat_xhat) /
              static_cast<float>(embedding_dim);
        }
      }
    }

    std::vector<float> d_head(batch_size * context_len * head_dim, 0.0f);
    std::vector<float> d_embeddings(batch_size * context_len * embedding_dim, 0.0f);
    for (size_t i = 0; i < d_embeddings.size(); ++i) {
      d_embeddings[i] = d_residual[i];
    }

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t head_base = b * context_len * head_dim + c * head_dim;
        const size_t residual_base = b * context_len * embedding_dim + c * embedding_dim;

        for (size_t i = 0; i < embedding_dim; ++i) {
          const float grad = d_residual[residual_base + i];
          for (size_t j = 0; j < head_dim; ++j) {
            attention_output_weights.grad[j * embedding_dim + i] += cache.head[head_base + j] * grad;
            d_head[head_base + j] += grad * attention_output_weights.val[j * embedding_dim + i];
          }
        }
      }
    }

    std::vector<float> d_attention(batch_size * context_len * context_len, 0.0f);
    std::vector<float> d_values(batch_size * context_len * head_dim, 0.0f);
    for (size_t b = 0; b < batch_size; ++b) {
      const size_t a_base = b * context_len * context_len;
      const size_t v_base = b * context_len * head_dim;

      for (size_t i = 0; i < context_len; ++i) {
        const size_t att_row = a_base + i * context_len;
        const size_t head_row = b * context_len * head_dim + i * head_dim;

        for (size_t h = 0; h < head_dim; ++h) {
          const float grad = d_head[head_row + h];
          for (size_t j = 0; j < context_len; ++j) {
            d_attention[att_row + j] += grad * cache.values[v_base + j * head_dim + h];
            d_values[v_base + j * head_dim + h] += cache.attention[att_row + j] * grad;
          }
        }
      }
    }

    std::vector<float> d_scores(batch_size * context_len * context_len, 0.0f);
    for (size_t b = 0; b < batch_size; ++b) {
      const size_t a_base = b * context_len * context_len;

      for (size_t i = 0; i < context_len; ++i) {
        const size_t row_base = a_base + i * context_len;
        float dot = 0.0f;
        for (size_t j = 0; j < context_len; ++j) {
          dot += d_attention[row_base + j] * cache.attention[row_base + j];
        }
        for (size_t j = 0; j < context_len; ++j) {
          d_scores[row_base + j] =
              cache.attention[row_base + j] * (d_attention[row_base + j] - dot);
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
        const size_t emb_base = b * context_len * embedding_dim + c * embedding_dim;
        const size_t qkv_base = b * context_len * head_dim + c * head_dim;

        for (size_t i = 0; i < embedding_dim; ++i) {
          const float x = cache.embeddings[emb_base + i];
          float grad = 0.0f;

          for (size_t j = 0; j < head_dim; ++j) {
            const float dq = d_queries[qkv_base + j];
            const float dk = d_keys[qkv_base + j];
            const float dv = d_values[qkv_base + j];

            query_weights.grad[i * head_dim + j] += x * dq;
            key_weights.grad[i * head_dim + j] += x * dk;
            value_weights.grad[i * head_dim + j] += x * dv;

            grad += dq * query_weights.val[i * head_dim + j];
            grad += dk * key_weights.val[i * head_dim + j];
            grad += dv * value_weights.val[i * head_dim + j];
          }

          d_embeddings[emb_base + i] += grad;
        }
      }
    }

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t emb_base = b * context_len * embedding_dim + c * embedding_dim;
        const size_t token_id = ids[b * context_len + c];
        const size_t tok_base = token_id * embedding_dim;
        const size_t pos_base = c * embedding_dim;

        for (size_t i = 0; i < embedding_dim; ++i) {
          token_embeddings.grad[tok_base + i] += d_embeddings[emb_base + i];
          position_embeddings.grad[pos_base + i] += d_embeddings[emb_base + i];
        }
      }
    }

    scale_grads(inv_token_count);
    return cache.avg_loss;
  }

  /// Compute the average loss for one batch without building gradients.
  float forward_loss(const std::vector<int> &ids, const std::vector<int> &targets) const {
    return forward(ids, targets).avg_loss;
  }

  /// Apply one optimizer step to every parameter tensor.
  void update() {
    token_embeddings.update();
    position_embeddings.update();
    query_weights.update();
    key_weights.update();
    value_weights.update();
    attention_output_weights.update();
    layer_norm_scale.update();
    layer_norm_shift.update();
    feed_forward_hidden_weights.update();
    feed_forward_hidden_bias.update();
    feed_forward_output_weights.update();
    feed_forward_output_bias.update();
    feed_forward_norm_scale.update();
    feed_forward_norm_shift.update();
    logit_weights.update();
    output_bias.update();
  }
};

/// Sample one batch of context windows and next-token targets.
void generate_batch(int min, int max, std::vector<int> &ids, std::vector<int> &targets,
                    const std::vector<int> &token_ids) {
  for (size_t b = 0; b < batch_size; ++b) {
    const int index = randint(min, max);
    for (size_t j = 0; j < context_len; ++j) {
      ids[b * context_len + j] = token_ids[index + j];
      targets[b * context_len + j] = token_ids[index + j + 1];
    }
  }
}

/// Run the current single-file training loop.
void run_training(Model &model, const std::vector<int> &token_ids) {
  const int split_index =
      static_cast<int>(std::floor(token_ids.size() * (1.0f - validation_split)));

  std::vector<int> ids(batch_size * context_len);
  std::vector<int> targets(batch_size * context_len);

  for (int start_step = 0; start_step < steps; start_step += steps_per_chunk) {
    const int chunk_steps = std::min(steps_per_chunk, steps - start_step);
    float train_loss = 0.0f;
    float val_loss = 0.0f;

    for (int step = 0; step < chunk_steps; ++step) {
      generate_batch(0, split_index - context_len, ids, targets, token_ids);
      train_loss += model.forward_backward(ids, targets);

      generate_batch(split_index, static_cast<int>(token_ids.size()) - context_len, ids, targets,
                     token_ids);
      val_loss += model.forward_loss(ids, targets);

      model.update();
    }

    std::cout << "step=" << start_step << " train_loss=" << train_loss / chunk_steps
              << " val_loss=" << val_loss / chunk_steps << "\n";
  }
}

/// Initialize the toy model and train it.
int main() {
  Model model = Model::init();

  const std::string corpus = load_corpus();
  const std::vector<int> token_ids = prepare_vocab(corpus);
  run_training(model, token_ids);
}
