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
const float learning_rate = 0.01f;
const float validation_split = 0.1f;

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

std::vector<float> tensor_randn(int numel) {
  std::vector<float> tensor(numel);
  for (auto &x : tensor) {
    x = randn();
  }
  return tensor;
}

std::vector<float> tensor_zeros(int numel) {
  std::vector<float> tensor(numel);
  for (auto &x : tensor) {
    x = randn();
  }
  return tensor;
}

void update_parameter(std::vector<float> param, std::vector<float> grad) {
  for (size_t i = 0; i < param.size(); ++i) {
    param[i] -= learning_rate * grad[i];
  }
}

class Model {
public:
  std::vector<float> embeddings;
  std::vector<float> hidden_weights;
  std::vector<float> hidden_bias;
  std::vector<float> output_weights;
  std::vector<float> output_bias;

  Model() {
    this->embeddings = tensor_randn(vocab_size * embedding_dim);
    this->hidden_weights = tensor_randn(context_len * embedding_dim * hidden_dim);
    this->hidden_bias = tensor_zeros(hidden_dim);
    this->output_weights = tensor_randn(hidden_dim * vocab_size);
    this->output_bias = tensor_zeros(vocab_size);
  }

  std::pair<float, Model> forward_backward(const std::vector<int> &ids, const int target) {
    std::vector<float> hidden = hidden_bias;
    for (size_t c = 0; c < context_len; ++c) {
      for (size_t i = 0; i < hidden_dim; ++i) {
        for (size_t j = 0; j < embedding_dim; ++j) {
          hidden[i] += embeddings[ids[c] * embedding_dim + j] *
                       hidden_weights[c * embedding_dim * hidden_dim + j * hidden_dim + i];
        }
      }
    }
    for (float &x : hidden) {
      x = std::tanh(x);
    }

    std::vector<float> logits = output_bias;
    for (size_t i = 0; i < vocab_size; ++i) {
      for (size_t j = 0; j < hidden_dim; ++j) {
        logits[i] += hidden[j] * output_weights[j * vocab_size + i];
      }
    }

    float max_logit = logits[0];
    for (const float logit : logits) {
      max_logit = std::max(max_logit, logit);
    }

    double sum_exp = 0.0;
    for (const float logit : logits) {
      sum_exp += std::exp(static_cast<double>(logit - max_logit));
    }

    const float loss = static_cast<float>(max_logit + std::log(sum_exp) - logits[target]);

    std::vector<float> d_logits(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
      d_logits[i] =
          static_cast<float>(std::exp(static_cast<double>(logits[i] - max_logit)) / sum_exp);
    }
    d_logits[target] -= 1.0f;

    std::vector<float> d_output_bias = d_logits;
    std::vector<float> d_output_weights(hidden_dim * vocab_size, 0.0f);
    std::vector<float> d_hidden(hidden_dim, 0.0f);
    for (size_t i = 0; i < vocab_size; i++) {
      for (size_t j = 0; j < hidden_dim; j++) {
        d_output_weights[j * vocab_size + i] += d_logits[i] * hidden[j];
        d_hidden[j] += d_logits[i] * output_weights[j * vocab_size + i];
      }
    }

    std::vector<float> d_z(hidden_dim);
    for (size_t i = 0; i < hidden_dim; i++) {
      d_z[i] = d_hidden[i] * (1.0f - hidden[i] * hidden[i]);
    }

    std::vector<float> d_hidden_bias = d_z;
    std::vector<float> d_hidden_weights(context_len * embedding_dim * hidden_dim, 0.0f);
    std::vector<float> d_embeddings(vocab_size * embedding_dim, 0.0f);

    for (size_t c = 0; c < context_len; ++c) {
      for (size_t i = 0; i < hidden_dim; ++i) {
        for (size_t j = 0; j < embedding_dim; ++j) {
          d_embeddings[ids[c] * embedding_dim + j] +=
              d_z[i] * hidden_weights[c * embedding_dim * hidden_dim + j * hidden_dim + i];
          d_hidden_weights[c * embedding_dim * hidden_dim + j * hidden_dim + i] +=
              d_z[i] * embeddings[ids[c] * embedding_dim + j];
        }
      }
    }

    Model gradient;
    gradient.embeddings = d_embeddings;
    gradient.hidden_weights = d_hidden_weights;
    gradient.hidden_bias = d_hidden_bias;
    gradient.output_weights = d_output_weights;
    gradient.output_bias = d_output_bias;

    return {loss, gradient};
  }

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

/// Copy one context window from the token stream into the working buffer.
void fill_context(std::vector<int> &ids, const std::vector<int> &token_ids, int start) {
  for (size_t i = 0; i < context_len; ++i) {
    ids[i] = token_ids[start + i];
  }
}

/// Run the current single-file training loop.
void run_training(Model model, const std::vector<int> &token_ids) {
  const int split_index =
      static_cast<int>(std::floor(token_ids.size() * (1.0f - validation_split)));

  for (int start_step = 0; start_step < steps; start_step += steps_per_chunk) {
    const int chunk_steps = std::min(steps_per_chunk, steps - start_step);
    float train_loss = 0.0f;
    float val_loss = 0.0f;

    std::vector<int> ids(context_len);
    for (int step = 0; step < chunk_steps; ++step) {
      const int train_index = randint(0, split_index - context_len);
      fill_context(ids, token_ids, train_index);
      const int target = token_ids[train_index + context_len];

      auto [loss, gradient] = model.forward_backward(ids, target);
      train_loss += loss;

      const int val_index = randint(split_index, static_cast<int>(token_ids.size()) - context_len);
      fill_context(ids, token_ids, val_index);
      const int val_target = token_ids[val_index + context_len];
      val_loss += model.forward_backward(ids, val_target).first;

      model.update(gradient);
    }

    std::cout << "step=" << start_step << " train_loss=" << train_loss / chunk_steps
              << " val_loss=" << val_loss / chunk_steps << "\n";
  }
}

/// Initialize the toy model and train it.
int main() {
  Model model = Model();

  const std::string corpus = load_corpus();
  const std::vector<int> token_ids = prepare_vocab(corpus);
  run_training(model, token_ids);
}
