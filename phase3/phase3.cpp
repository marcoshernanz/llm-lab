#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

const std::string corpus = "This is a test string for my first MLP model";
const int vocab_size = 64;
const int embedding_dim = 32;

// Is this a good idea? to do this globally?
std::mt19937 gen(0);
std::normal_distribution<float> dist(0.0f, 1.0f);

std::unordered_map<char, int> char_to_id;
std::vector<char> id_to_char(vocab_size);

float randn() { return dist(gen); }

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
      std::cout << "Skipped char: " << c << '\n';
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

  for (int &id : token_ids) {
    std::vector<float> out(1, vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
      for (size_t j = 0; j < embedding_dim; ++j) {
        out[i] += embeddings[id * vocab_size + j] * weights[i * embedding_dim + i];
      }
      out[i] += biases[i];
    }
    for (auto x : out) {
      std::cout << x << " ";
    }
    std::cout << "\n";
  }
}
