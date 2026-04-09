/// Minimal phase-3 script for learning manual language-model gradients.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
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
const int hidden_dim = 64;

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

class Adam {
public:
  int step = 1;
  size_t size;
  std::vector<float> first_moment;
  std::vector<float> second_moment;

  Adam(size_t size) : size(size), first_moment(size), second_moment(size) {}

  void update(std::vector<float> &param, const std::vector<float> &grad) {
    for (size_t i = 0; i < size; i++) {
      first_moment[i] = beta1 * first_moment[i] + (1.0f - beta1) * grad[i];
      second_moment[i] = beta2 * second_moment[i] + (1.0f - beta2) * grad[i] * grad[i];

      float corrected_first_moment = first_moment[i] / (1.0f - std::pow(beta1, step));
      float corrected_second_moment = second_moment[i] / (1.0f - std::pow(beta2, step));

      param[i] = param[i] - learning_rate * corrected_first_moment /
                                (std::sqrt(corrected_second_moment) + eps);
    }

    step++;
  }
};

class Param {
public:
  size_t size;
  std::vector<float> val;

  Param(size_t size) : size(size), val(size) {}

  /// Create one random parameter tensor with standard-normal entries.
  void init_randn() {
    for (auto &x : val) {
      x = randn();
    }
  }

  /// Create one zero-initialized tensor with the requested size.
  void init_zeros() { std::fill(val.begin(), val.end(), 0.0f); }
};

/// Apply one SGD update to a parameter tensor.
void update_parameter(std::vector<float> &param, const std::vector<float> &grad) {
  for (size_t i = 0; i < param.size(); ++i) {
    param[i] -= learning_rate * grad[i];
  }
}

/// Hold the trainable tensors for the tiny language model.
class Model {
public:
  Param embeddings;
  Param hidden_weights;
  Param hidden_bias;
  Param output_weights;
  Param output_bias;

  Model()
      : embeddings(vocab_size * embedding_dim),
        hidden_weights(context_len * embedding_dim * hidden_dim), hidden_bias(hidden_dim),
        output_weights(hidden_dim * vocab_size), output_bias(vocab_size) {}

  /// Initialize one model with random weights and zero biases.
  static Model init() {
    Model model = Model();
    init_randn(model.embeddings);
    init_randn(model.hidden_weights);
    init_zeros(model.hidden_bias);
    init_randn(model.output_weights);
    init_zeros(model.output_bias);
    return model;
  }

  /// Compute one batch of hidden activations from token ids.
  std::vector<float> compute_hidden(const std::vector<int> &ids) const {
    std::vector<float> hidden(batch_size * hidden_dim);
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t i = 0; i < hidden_dim; ++i) {
        hidden[b * hidden_dim + i] = hidden_bias[i];
      }
    }

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        for (size_t i = 0; i < hidden_dim; ++i) {
          for (size_t j = 0; j < embedding_dim; ++j) {
            hidden[b * hidden_dim + i] +=
                embeddings[ids[b * context_len + c] * embedding_dim + j] *
                hidden_weights[c * embedding_dim * hidden_dim + j * hidden_dim + i];
          }
        }
      }
    }

    for (float &x : hidden) {
      x = std::tanh(x);
    }

    return hidden;
  }

  /// Compute one batch of logits from hidden activations.
  std::vector<float> compute_logits(const std::vector<float> &hidden) const {
    std::vector<float> logits(batch_size * vocab_size);
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t i = 0; i < vocab_size; ++i) {
        logits[b * vocab_size + i] = output_bias[i];
      }
    }

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t i = 0; i < vocab_size; ++i) {
        for (size_t j = 0; j < hidden_dim; ++j) {
          logits[b * vocab_size + i] +=
              hidden[b * hidden_dim + j] * output_weights[j * vocab_size + i];
        }
      }
    }

    return logits;
  }

  /// Run one full forward and backward pass for one training example.
  std::pair<float, Model> forward_backward(const std::vector<int> &ids,
                                           const std::vector<int> &targets) const {
    const std::vector<float> hidden = compute_hidden(ids);
    const std::vector<float> logits = compute_logits(hidden);

    std::vector<float> max_logits(batch_size);
    for (size_t b = 0; b < batch_size; ++b) {
      max_logits[b] = logits[b * vocab_size];
    }

    for (size_t b = 0; b < batch_size; ++b) {
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

    const float avg_loss = loss_sum / static_cast<float>(batch_size);

    std::vector<float> d_logits(batch_size * vocab_size);
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t i = 0; i < vocab_size; ++i) {
        d_logits[b * vocab_size + i] = static_cast<float>(
            std::exp(static_cast<double>(logits[b * vocab_size + i] - max_logits[b])) /
            sums_exp[b]);
      }
      d_logits[b * vocab_size + targets[b]] -= 1.0f;
    }

    std::vector<float> d_output_bias(vocab_size, 0.0f);
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t i = 0; i < vocab_size; ++i) {
        d_output_bias[i] += d_logits[b * vocab_size + i];
      }
    }

    std::vector<float> d_output_weights(hidden_dim * vocab_size, 0.0f);
    std::vector<float> d_hidden(batch_size * hidden_dim, 0.0f);
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t i = 0; i < vocab_size; ++i) {
        for (size_t j = 0; j < hidden_dim; ++j) {
          d_output_weights[j * vocab_size + i] +=
              d_logits[b * vocab_size + i] * hidden[b * hidden_dim + j];
          d_hidden[b * hidden_dim + j] +=
              d_logits[b * vocab_size + i] * output_weights[j * vocab_size + i];
        }
      }
    }

    std::vector<float> d_z(batch_size * hidden_dim);
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t i = 0; i < hidden_dim; ++i) {
        d_z[b * hidden_dim + i] = d_hidden[b * hidden_dim + i] *
                                  (1.0f - hidden[b * hidden_dim + i] * hidden[b * hidden_dim + i]);
      }
    }

    std::vector<float> d_hidden_bias(hidden_dim, 0.0f);
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t i = 0; i < hidden_dim; ++i) {
        d_hidden_bias[i] += d_z[b * hidden_dim + i];
      }
    }

    std::vector<float> d_hidden_weights(context_len * embedding_dim * hidden_dim, 0.0f);
    std::vector<float> d_embeddings(vocab_size * embedding_dim, 0.0f);
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        for (size_t i = 0; i < hidden_dim; ++i) {
          for (size_t j = 0; j < embedding_dim; ++j) {
            d_embeddings[ids[b * context_len + c] * embedding_dim + j] +=
                d_z[b * hidden_dim + i] *
                hidden_weights[c * embedding_dim * hidden_dim + j * hidden_dim + i];
            d_hidden_weights[c * embedding_dim * hidden_dim + j * hidden_dim + i] +=
                d_z[b * hidden_dim + i] * embeddings[ids[b * context_len + c] * embedding_dim + j];
          }
        }
      }
    }

    for (float &x : d_embeddings) {
      x *= inv_batch_size;
    }
    for (float &x : d_hidden_weights) {
      x *= inv_batch_size;
    }
    for (float &x : d_hidden_bias) {
      x *= inv_batch_size;
    }
    for (float &x : d_output_weights) {
      x *= inv_batch_size;
    }
    for (float &x : d_output_bias) {
      x *= inv_batch_size;
    }

    Model gradient;
    gradient.embeddings = d_embeddings;
    gradient.hidden_weights = d_hidden_weights;
    gradient.hidden_bias = d_hidden_bias;
    gradient.output_weights = d_output_weights;
    gradient.output_bias = d_output_bias;

    return {avg_loss, gradient};
  }

  /// Compute the average loss for one batch without building gradients.
  float forward_loss(const std::vector<int> &ids, const std::vector<int> &targets) const {
    const std::vector<float> hidden = compute_hidden(ids);
    const std::vector<float> logits = compute_logits(hidden);

    std::vector<float> max_logits(batch_size);
    for (size_t b = 0; b < batch_size; ++b) {
      max_logits[b] = logits[b * vocab_size];
    }

    for (size_t b = 0; b < batch_size; ++b) {
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

    return loss_sum * inv_batch_size;
  }

  /// Apply one SGD update from one gradient container.
  void update(const Model &gradient) {
    update_parameter(embeddings, gradient.embeddings);
    update_parameter(hidden_weights, gradient.hidden_weights);
    update_parameter(hidden_bias, gradient.hidden_bias);
    update_parameter(output_weights, gradient.output_weights);
    update_parameter(output_bias, gradient.output_bias);
  }
};

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

  for (int start_step = 0; start_step < steps; start_step += steps_per_chunk) {
    const int chunk_steps = std::min(steps_per_chunk, steps - start_step);
    float train_loss = 0.0f;
    float val_loss = 0.0f;

    std::vector<int> ids(batch_size * context_len);
    std::vector<int> targets(batch_size);
    for (int step = 0; step < chunk_steps; ++step) {
      generate_batch(0, split_index - context_len, ids, targets, token_ids);

      const auto [loss, gradient] = model.forward_backward(ids, targets);
      train_loss += loss;

      generate_batch(split_index, static_cast<int>(token_ids.size()) - context_len, ids, targets,
                     token_ids);
      val_loss += model.forward_loss(ids, targets);

      model.update(gradient);
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
