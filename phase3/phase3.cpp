/// Minimal phase-3 script for learning manual language-model gradients.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

const std::string corpus_path = "../datasets/tinyshakespeare.txt";
const int vocab_size = 128;
const int context_len = 4;
const int embedding_dim = 32;
const int hidden_dim = 64;
const int steps = 10000;
const int steps_per_chunk = 100;
const float learning_rate = 0.01f;

std::unordered_map<char, int> char_to_id;

struct ForwardBackwardResult {
  float loss;
  std::vector<float> d_biases;
  std::vector<float> d_weights;
  std::vector<float> d_embeddings;
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

/// Sample one integer in the half-open range [0, max).
int randint(int max) {
  std::uniform_int_distribution<int> dist(0, max - 1);
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

/// Run one full forward and backward pass for a single training example.
ForwardBackwardResult forward_backward(const std::vector<float> &embeddings,
                                       const std::vector<float> &w, const std::vector<float> &b,
                                       const std::vector<float> &w_out,
                                       const std::vector<float> &b_out, const std::vector<int> &ids,
                                       int target) {

  std::vector<float> h(hidden_dim, 0.0f);
  for (size_t i = 0; i < vocab_size; ++i) {
    h[i] = b[i];
  }

  for (size_t c = 0; c < context_len; ++c) {
    for (size_t i = 0; i < hidden_dim; ++i) {
      for (size_t j = 0; j < embedding_dim; ++j) {
        h[i] += embeddings[ids[c] * embedding_dim + j] *
                w[c * embedding_dim * hidden_dim + j * hidden_dim + i];
      }
    }
  }

  for (float &x : h) {
    x = std::tanh(x);
  }

  std::vector<float> logits(vocab_size, 0.0f);
  for (size_t i = 0; i < vocab_size; ++i) {
    logits[i] = b_out[i];
  }

  for (size_t i = 0; i < vocab_size; ++i) {
    for (size_t j = 0; j < hidden_dim; ++j) {
      logits[i] += h[j] * w_out[j * vocab_size + i];
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

  std::vector<float> d_b_out = d_logits;
  std::vector<float> d_w_out(hidden_dim * vocab_size, 0.0f);
  std::vector<float> d_h(hidden_dim, 0.0f);
  for (size_t i = 0; i < vocab_size; i++) {
    for (size_t j = 0; j < hidden_dim; j++) {
      d_w_out[j * vocab_size + i] = d_logits[i] * h[j];
      d_h[j] = d_logits[i] * w_out[j * vocab_size + i];
    }
  }

  std::vector<float> d_biases = d_logits;
  std::vector<float> d_weights(context_len * embedding_dim * vocab_size, 0.0f);
  std::vector<float> d_embeddings(vocab_size * embedding_dim, 0.0f);

  for (size_t c = 0; c < context_len; ++c) {
    for (size_t i = 0; i < vocab_size; ++i) {
      for (size_t j = 0; j < embedding_dim; ++j) {
        d_weights[c * embedding_dim * vocab_size + j * vocab_size + i] =
            embeddings[ids[c] * embedding_dim + j] * d_logits[i];
        d_embeddings[ids[c] * embedding_dim + j] +=
            weights[c * embedding_dim * vocab_size + j * vocab_size + i] * d_logits[i];
      }
    }
  }

  return ForwardBackwardResult{
      .loss = loss,
      .d_biases = d_biases,
      .d_weights = d_weights,
      .d_embeddings = d_embeddings,
  };
}

/// Apply one SGD update using the current single-example gradients.
void apply_gradients(std::vector<float> &embeddings, std::vector<float> &w, std::vector<float> &b,
                     std::vector<float> &w_out, std::vector<float> &b_out,
                     const ForwardBackwardResult &result) {
  for (size_t i = 0; i < biases.size(); ++i) {
    biases[i] -= learning_rate * result.d_biases[i];
  }

  for (size_t i = 0; i < weights.size(); ++i) {
    weights[i] -= learning_rate * result.d_weights[i];
  }
  for (size_t i = 0; i < embeddings.size(); ++i) {
    embeddings[i] -= learning_rate * result.d_embeddings[i];
  }
}

/// Run the current single-file training loop.
void run_training(std::vector<float> &embeddings, std::vector<float> &w, std::vector<float> &b,
                  std::vector<float> &w_out, std::vector<float> &b_out,
                  const std::vector<int> &token_ids) {
  for (int start_step = 0; start_step < steps; start_step += steps_per_chunk) {
    const int chunk_steps = std::min(steps_per_chunk, steps - start_step);
    float loss = 0.0f;
    std::vector<int> ids(context_len);
    for (int step = 0; step < chunk_steps; ++step) {
      const int index = randint(static_cast<int>(token_ids.size()) - context_len);
      for (size_t i = 0; i < context_len; ++i) {
        ids[i] = token_ids[index + i];
      }
      const int target = token_ids[index + context_len];

      const ForwardBackwardResult result =
          forward_backward(embeddings, w, b, w_out, b_out, ids, target);
      apply_gradients(embeddings, w, b, w_out, b_out, result);
      loss += result.loss;
    }

    std::cout << "step=" << start_step << " loss=" << loss / chunk_steps << "\n";
  }
}

/// Initialize the toy model and train it.
int main() {
  std::vector<float> embeddings(vocab_size * embedding_dim);
  std::vector<float> w(context_len * embedding_dim * hidden_dim);
  std::vector<float> b(hidden_dim, 0.0f);
  std::vector<float> w_out(hidden_dim * vocab_size);
  std::vector<float> b_out(vocab_size, 0.0f);

  for (float &x : embeddings) {
    x = randn();
  }
  for (float &x : w) {
    x = randn();
  }
  for (float &x : w_out) {
    x = randn();
  }

  const std::string corpus = load_corpus();
  const std::vector<int> token_ids = prepare_vocab(corpus);
  run_training(embeddings, w, b, w_out, b_out, token_ids);
}
