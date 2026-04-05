#include <array>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

const std::string corpus = "This is a test string for my first MLP model";
const int vocab_size = 64;
const int embedding_dim = 32;

// Is this a good idea? to do this globally?
std::random_device rd;
std::mt19937 gen(rd());
std::normal_distribution<float> dist(0.0f, 1.0f);

std::unordered_map<char, int> char_to_id;

float randn() { return dist(gen); }

std::vector<int> prepare_vocab() {
  std::vector<int> token_ids(corpus.size());
  for (int i = 0; i < corpus.size(); ++i) {
    char c = corpus[i];
    if (char_to_id.find(c) != char_to_id.end()) {
      token_ids[i] = char_to_id[c];
    } else if (char_to_id.size() < vocab_size) {
      char_to_id[c] = char_to_id.size();
      token_ids[i] = char_to_id[c];
    } else {
      std::cout << "Skipped char: " << c << '\n';
    }
  }

  return token_ids;
}

int main() {
  std::array<std::array<float, embedding_dim>, vocab_size> embeddings;

  // for auto or for i in? row & col good names?
  for (auto &row : embeddings) {
    for (auto &col : row) {
      col = randn();
    }
  }
}
