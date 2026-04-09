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

const int steps = 10000;
const int steps_per_chunk = 100;
const int batch_size = 32;
const float inv_batch_size = 1.0f / static_cast<float>(batch_size);
const float validation_split = 0.1f;

const float learning_rate = 0.01f;
const float beta1 = 0.9f;
const float beta2 = 0.999f;
const float eps = 1e-8f;

std::unordered_map<char, int> char_to_id;

/// Hold the numerically stable loss terms for one batch.
struct LossStats {
  std::vector<float> max_logits;
  std::vector<double> sums_exp;
  float avg_loss;
};

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

/// Compute the stable softmax-loss terms for one batch of logits.
LossStats compute_loss_stats(const std::vector<float> &logits, const std::vector<int> &targets) {
  std::vector<float> max_logits(batch_size);
  for (size_t b = 0; b < batch_size; ++b) {
    max_logits[b] = logits[b * vocab_size];
    for (size_t i = 0; i < vocab_size; ++i) {
      max_logits[b] = std::max(max_logits[b], logits[b * vocab_size + i]);
    }
  }

  std::vector<double> sums_exp(batch_size, 0.0);
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t i = 0; i < vocab_size; ++i) {
      sums_exp[b] += std::exp(static_cast<double>(logits[b * vocab_size + i] - max_logits[b]));
    }
  }

  float loss_sum = 0.0f;
  for (size_t b = 0; b < batch_size; ++b) {
    loss_sum += static_cast<float>(max_logits[b] + std::log(sums_exp[b]) -
                                   logits[b * vocab_size + targets[b]]);
  }

  return LossStats{
      .max_logits = max_logits,
      .sums_exp = sums_exp,
      .avg_loss = loss_sum * inv_batch_size,
  };
}

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
  Param output_weights;
  Param output_bias;

  /// Construct one model with correctly sized parameter tensors.
  Model()
      : token_embeddings(vocab_size * embedding_dim),
        position_embeddings(context_len * embedding_dim), query_weights(embedding_dim * head_dim),
        key_weights(embedding_dim * head_dim), value_weights(embedding_dim * head_dim),
        output_weights(head_dim * vocab_size), output_bias(vocab_size) {}

  /// Initialize one model with random weights and zero biases.
  static Model init() {
    Model model;
    model.token_embeddings.init_randn();
    model.position_embeddings.init_randn();
    model.query_weights.init_randn();
    model.key_weights.init_randn();
    model.value_weights.init_randn();
    model.output_weights.init_randn();
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
    output_weights.zero_grad();
    output_bias.zero_grad();
  }

  /// Scale every parameter gradient buffer by one constant.
  void scale_grads(float scale) {
    token_embeddings.scale_grad(scale);
    position_embeddings.scale_grad(scale);
    query_weights.scale_grad(scale);
    key_weights.scale_grad(scale);
    value_weights.scale_grad(scale);
    output_weights.scale_grad(scale);
    output_bias.scale_grad(scale);
  }

  /// Run one full forward and backward pass for one batch.
  float forward_backward(const std::vector<int> &ids, const std::vector<int> &targets) {
    zero_grad();

    std::vector<float> embeddings(batch_size * context_len * embedding_dim, 0.0f);

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t out_base = b * context_len * embedding_dim + c * embedding_dim;
        const size_t token_id = ids[b * context_len + c];
        const size_t tok_base = token_id * embedding_dim;
        const size_t pos_base = c * embedding_dim;

        for (size_t i = 0; i < embedding_dim; ++i) {
          embeddings[out_base + i] =
              token_embeddings.val[tok_base + i] + position_embeddings.val[pos_base + i];
        }
      }
    }

    std::vector<float> queries(batch_size * context_len * head_dim);
    std::vector<float> keys(batch_size * context_len * head_dim);
    std::vector<float> values(batch_size * context_len * head_dim);

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t emb_base = b * context_len * embedding_dim + c * embedding_dim;
        const size_t out_base = b * context_len * head_dim + c * head_dim;

        for (size_t i = 0; i < head_dim; ++i) {
          float q = 0.0f;
          float k = 0.0f;
          float v = 0.0f;

          for (size_t j = 0; j < embedding_dim; ++j) {
            const float x = embeddings[emb_base + j];
            q += x * query_weights.val[j * head_dim + i];
            k += x * key_weights.val[j * head_dim + i];
            v += x * value_weights.val[j * head_dim + i];
          }

          queries[out_base + i] = q;
          keys[out_base + i] = k;
          values[out_base + i] = v;
        }
      }
    }

    std::vector<float> attention(batch_size * context_len * context_len);
    const float sqrt_hidden_dim = std::sqrt(static_cast<float>(head_dim));

    for (size_t b = 0; b < batch_size; ++b) {
      const size_t qk_base = b * context_len * head_dim;
      const size_t a_base = b * context_len * context_len;

      for (size_t i = 0; i < context_len; ++i) {
        const size_t q_base = qk_base + i * head_dim;

        for (size_t j = 0; j < context_len; ++j) {
          const size_t k_base = qk_base + j * head_dim;

          float score = 0.0f;
          for (size_t h = 0; h < head_dim; ++h) {
            score += queries[q_base + h] * keys[k_base + h];
          }

          if (j <= i) {
            attention[a_base + i * context_len + j] = score / sqrt_hidden_dim;
          } else {
            attention[a_base + i * context_len + j] = -std::numeric_limits<float>::infinity();
          }
        }
      }
    }

    std::vector<float> max_logits(batch_size * context_len);

    for (size_t b = 0; b < batch_size; ++b) {
      const size_t a_base = b * context_len * context_len;
      const size_t m_base = b * context_len;

      for (size_t i = 0; i < context_len; ++i) {
        const size_t row_base = a_base + i * context_len;
        float max_val = attention[row_base];
        for (size_t j = 1; j < context_len; ++j) {
          max_val = std::max(max_val, attention[row_base + j]);
        }
        max_logits[m_base + i] = max_val;
      }
    }

    std::vector<double> sums_exp(batch_size * context_len, 0.0);

    for (size_t b = 0; b < batch_size; ++b) {
      const size_t a_base = b * context_len * context_len;
      const size_t m_base = b * context_len;

      for (size_t i = 0; i < context_len; ++i) {
        const size_t row_base = a_base + i * context_len;
        const float max_val = max_logits[m_base + i];

        for (size_t j = 0; j < context_len; ++j) {
          sums_exp[m_base + i] += std::exp(static_cast<double>(attention[row_base + j] - max_val));
        }
      }
    }

    for (size_t b = 0; b < batch_size; ++b) {
      const size_t a_base = b * context_len * context_len;
      const size_t m_base = b * context_len;

      for (size_t i = 0; i < context_len; ++i) {
        const size_t row_base = a_base + i * context_len;
        const float max_val = max_logits[m_base + i];
        const double denom = sums_exp[m_base + i];

        for (size_t j = 0; j < context_len; ++j) {
          attention[row_base + j] = static_cast<float>(
              std::exp(static_cast<double>(attention[row_base + j] - max_val)) / denom);
        }
      }
    }

    std::vector<float> head(batch_size * context_len * head_dim);

    // head[b,i,h] = sum_j a[b,i,j] * v[b,j,h]
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t i = 0; i < context_len; ++i) {
        for (size_t h = 0; h < head_dim; ++h) {
          float sum = 0.0f;

          for (size_t j = 0; j < context_len; ++j) {
            sum += attention[b * context_len * context_len + i * context_len + j] *
                   values[b * context_len * head_dim + j * head_dim + h];
          }

          head[b * context_len * head_dim + i * head_dim + h] = sum;
        }
      }
    }

    // TODO

    std::vector<float> hidden(batch_size * head_dim);
    for (size_t b = 0; b < batch_size; ++b) {
      const size_t hidden_offset = b * head_dim;
      const size_t ids_offset = b * context_len;
      for (size_t i = 0; i < head_dim; ++i) {
        hidden[hidden_offset + i] = hidden_bias.val[i];
      }

      for (size_t c = 0; c < context_len; ++c) {
        for (size_t i = 0; i < head_dim; ++i) {
          for (size_t j = 0; j < embedding_dim; ++j) {
            hidden[hidden_offset + i] +=
                embeddings.val[ids[ids_offset + c] * embedding_dim + j] *
                hidden_weights.val[c * embedding_dim * hidden_dim + j * hidden_dim + i];
          }
        }
      }
    }

    for (float &x : hidden) {
      x = std::tanh(x);
    }

    std::vector<float> logits(batch_size * vocab_size);
    for (size_t b = 0; b < batch_size; ++b) {
      const size_t hidden_offset = b * head_dim;
      const size_t logits_offset = b * vocab_size;
      for (size_t i = 0; i < vocab_size; ++i) {
        logits[logits_offset + i] = output_bias.val[i];
        for (size_t j = 0; j < head_dim; ++j) {
          logits[logits_offset + i] +=
              hidden[hidden_offset + j] * output_weights.val[j * vocab_size + i];
        }
      }
    }

    const LossStats loss_stats = compute_loss_stats(logits, targets);

    std::vector<float> d_logits(batch_size * vocab_size);
    for (size_t b = 0; b < batch_size; ++b) {
      const size_t logits_offset = b * vocab_size;
      for (size_t i = 0; i < vocab_size; ++i) {
        d_logits[logits_offset + i] = static_cast<float>(
            std::exp(static_cast<double>(logits[logits_offset + i] - loss_stats.max_logits[b])) /
            loss_stats.sums_exp[b]);
      }
      d_logits[logits_offset + targets[b]] -= 1.0f;
    }

    for (size_t b = 0; b < batch_size; ++b) {
      const size_t logits_offset = b * vocab_size;
      for (size_t i = 0; i < vocab_size; ++i) {
        output_bias.grad[i] += d_logits[logits_offset + i];
      }
    }

    std::vector<float> d_hidden(batch_size * head_dim, 0.0f);
    for (size_t b = 0; b < batch_size; ++b) {
      const size_t hidden_offset = b * head_dim;
      const size_t logits_offset = b * vocab_size;
      for (size_t i = 0; i < vocab_size; ++i) {
        for (size_t j = 0; j < head_dim; ++j) {
          output_weights.grad[j * vocab_size + i] +=
              d_logits[logits_offset + i] * hidden[hidden_offset + j];
          d_hidden[hidden_offset + j] +=
              d_logits[logits_offset + i] * output_weights.val[j * vocab_size + i];
        }
      }
    }

    std::vector<float> d_hidden_pre(batch_size * head_dim);
    for (size_t b = 0; b < batch_size; ++b) {
      const size_t hidden_offset = b * head_dim;
      for (size_t i = 0; i < head_dim; ++i) {
        d_hidden_pre[hidden_offset + i] =
            d_hidden[hidden_offset + i] *
            (1.0f - hidden[hidden_offset + i] * hidden[hidden_offset + i]);
      }
    }

    for (size_t b = 0; b < batch_size; ++b) {
      const size_t hidden_offset = b * head_dim;
      for (size_t i = 0; i < head_dim; ++i) {
        head_dim.grad[i] += d_hidden_pre[hidden_offset + i];
      }
    }

    for (size_t b = 0; b < batch_size; ++b) {
      const size_t hidden_offset = b * head_dim;
      const size_t ids_offset = b * context_len;
      for (size_t c = 0; c < context_len; ++c) {
        for (size_t i = 0; i < head_dim; ++i) {
          for (size_t j = 0; j < embedding_dim; ++j) {
            const int token_id = ids[ids_offset + c];
            embeddings.grad[token_id * embedding_dim + j] +=
                d_hidden_pre[hidden_offset + i] *
                hidden_weights.val[c * embedding_dim * hidden_dim + j * hidden_dim + i];
            hidden_weights.grad[c * embedding_dim * hidden_dim + j * hidden_dim + i] +=
                d_hidden_pre[hidden_offset + i] * embeddings.val[token_id * embedding_dim + j];
          }
        }
      }
    }

    scale_grads(inv_batch_size);
    return loss_stats.avg_loss;
  }

  /// Compute the average loss for one batch without building gradients.
  float forward_loss(const std::vector<int> &ids, const std::vector<int> &targets) const {
    // TODO
  }

  /// Apply one optimizer step to every parameter tensor.
  void update() {
    embeddings.update();
    hidden_weights.update();
    head_dim.update();
    output_weights.update();
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
    }
    targets[b] = token_ids[index + context_len];
  }
}

/// Run the current single-file training loop.
void run_training(Model &model, const std::vector<int> &token_ids) {
  const int split_index =
      static_cast<int>(std::floor(token_ids.size() * (1.0f - validation_split)));

  std::vector<int> ids(batch_size * context_len);
  std::vector<int> targets(batch_size);

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
