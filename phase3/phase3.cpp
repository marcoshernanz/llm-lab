/// Minimal phase-3 script for learning manual language-model gradients.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

const std::string corpus = "This is a test string for my first MLP model";
const int vocab_size = 64;
const int embedding_dim = 32;
const int steps = 1000;
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

/// Build a tiny character vocabulary and return the encoded corpus.
std::vector<int> prepare_vocab() {
  std::vector<int> token_ids(corpus.size());
  for (size_t i = 0; i < corpus.size(); ++i) {
    const char c = corpus[i];
    if (const auto it = char_to_id.find(c); it != char_to_id.end()) {
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
                                       const std::vector<float> &weights,
                                       const std::vector<float> &biases, int id, int target) {
  std::vector<float> logits(vocab_size, 0.0f);
  for (size_t i = 0; i < vocab_size; ++i) {
    logits[i] = biases[i];
    for (size_t j = 0; j < embedding_dim; ++j) {
      logits[i] += embeddings[id * embedding_dim + j] * weights[j * vocab_size + i];
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

  std::vector<float> d_biases = d_logits;
  std::vector<float> d_weights(embedding_dim * vocab_size, 0.0f);
  std::vector<float> d_embeddings(vocab_size * embedding_dim, 0.0f);
  for (size_t i = 0; i < vocab_size; ++i) {
    for (size_t j = 0; j < embedding_dim; ++j) {
      d_weights[j * vocab_size + i] = embeddings[id * embedding_dim + j] * d_logits[i];
      d_embeddings[id * embedding_dim + j] += weights[j * vocab_size + i] * d_logits[i];
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
void apply_gradients(std::vector<float> &embeddings, std::vector<float> &weights,
                     std::vector<float> &biases, const ForwardBackwardResult &result) {
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
void run_training(std::vector<float> &embeddings, std::vector<float> &weights,
                  std::vector<float> &biases, const std::vector<int> &token_ids) {
  for (size_t step = 0; step < steps; ++step) {
    const int index = randint(static_cast<int>(token_ids.size()) - 1);
    const int id = token_ids[index];
    const int target = token_ids[index + 1];

    const ForwardBackwardResult result = forward_backward(embeddings, weights, biases, id, target);
    apply_gradients(embeddings, weights, biases, result);

    std::cout << "step=" << step << " loss=" << result.loss << "\n";
  }
}

/// Initialize the toy model and train it.
int main() {
  std::vector<float> embeddings(vocab_size * embedding_dim);
  std::vector<float> weights(embedding_dim * vocab_size);
  std::vector<float> biases(vocab_size, 0.0f);

  for (float &x : embeddings) {
    x = randn();
  }
  for (float &x : weights) {
    x = randn();
  }

  const std::vector<int> token_ids = prepare_vocab();
  run_training(embeddings, weights, biases, token_ids);
}
