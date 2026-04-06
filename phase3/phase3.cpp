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
const int steps = 10;

std::unordered_map<char, int> char_to_id;
std::vector<char> id_to_char(vocab_size);

std::mt19937 &rng() {
  static std::mt19937 gen(0);
  return gen;
}

float randn() {
  std::normal_distribution<float> dist(0.0f, 1.0f);
  return dist(rng());
}

int randint(int max) {
  std::uniform_int_distribution<int> dist(0, max - 1);
  return dist(rng());
}

std::vector<int> prepare_vocab() {
  std::vector<int> token_ids(corpus.size());
  for (size_t i = 0; i < corpus.size(); ++i) {
    char c = corpus[i];
    if (char_to_id.find(c) != char_to_id.end()) {
      token_ids[i] = char_to_id[c];
    } else if (char_to_id.size() < vocab_size) {
      id_to_char[char_to_id.size()] = c;
      char_to_id[c] = char_to_id.size();
      token_ids[i] = char_to_id[c];
    } else {
      throw std::runtime_error("vocab_size too small");
    }
  }

  return token_ids;
}

int main() {
  std::vector<float> embeddings(vocab_size * embedding_dim);
  std::vector<float> weights(embedding_dim * vocab_size);
  std::vector<float> biases(vocab_size, 0.0f);
  // for auto or for i in? x good name?
  for (float &x : embeddings) {
    x = randn();
  }
  for (float &x : weights) {
    x = randn();
  }

  std::vector<int> token_ids = prepare_vocab();

  for (int step = 0; step < steps; ++step) {
    int index = randint(token_ids.size() - 1);
    int id = token_ids[index];
    int target = token_ids[index + 1];

    std::vector<float> out(vocab_size, 0.0f);
    for (size_t i = 0; i < vocab_size; ++i) {
      for (size_t j = 0; j < embedding_dim; ++j) {
        out[i] += embeddings[id * embedding_dim + j] * weights[j * vocab_size + i];
      }
      out[i] += biases[i];
    }

    float max_logit = out[0];
    for (float &x : out) {
      max_logit = std::max(max_logit, x);
    }

    double sum_exp = 0.0;
    for (float &x : out) {
      sum_exp += std::exp(static_cast<double>(x - max_logit));
    }

    float loss = static_cast<float>(max_logit + std::log(sum_exp) - out[target]);

    std::vector<float> d_logits(vocab_size);
    for (size_t i = 0; i < vocab_size; i++) {
      d_logits[i] = std::exp(out[i] - max_logit) / sum_exp;
    }
    d_logits[target] -= 1.0f;

    std::cout << "step=" << step << " loss=" << loss << "\n";
  }
}
