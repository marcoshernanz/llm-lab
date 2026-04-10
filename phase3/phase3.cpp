/// Minimal phase-3 script for learning manual language-model gradients.

#include "attention.h"
#include "core.h"
#include "feed_forward.h"
#include "layer_norm.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

const std::string corpus_path = "../datasets/tinyshakespeare.txt";

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

/// Hold the intermediate tensors from one full block forward pass.
struct ForwardCache {
  std::vector<float> embeddings;
  attention::Cache attention;
  layer_norm::Cache attention_norm;
  feed_forward::Cache feed_forward;
  layer_norm::Cache feed_forward_norm;
  std::vector<float> logits;
  std::vector<float> probs;
  float avg_loss = 0.0f;
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
  Param attention_norm_scale;
  Param attention_norm_shift;
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
        attention_output_weights(head_dim * embedding_dim), attention_norm_scale(embedding_dim),
        attention_norm_shift(embedding_dim),
        feed_forward_hidden_weights(embedding_dim * feed_forward_dim),
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
    model.attention_norm_scale.init_ones();
    model.attention_norm_shift.init_zeros();
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
    attention_norm_scale.zero_grad();
    attention_norm_shift.zero_grad();
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
    attention_norm_scale.scale_grad(scale);
    attention_norm_shift.scale_grad(scale);
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
    ForwardCache cache;
    cache.embeddings = compute_embeddings(ids);
    cache.attention = attention::forward(cache.embeddings, query_weights, key_weights,
                                         value_weights, attention_output_weights);
    cache.attention_norm =
        layer_norm::forward(cache.attention.residual, attention_norm_scale, attention_norm_shift);
    cache.feed_forward = feed_forward::forward(
        cache.attention_norm.output, feed_forward_hidden_weights, feed_forward_hidden_bias,
        feed_forward_output_weights, feed_forward_output_bias);
    cache.feed_forward_norm = layer_norm::forward(cache.feed_forward.residual,
                                                  feed_forward_norm_scale, feed_forward_norm_shift);
    compute_logits_and_loss(cache.feed_forward_norm.output, targets, cache.logits, cache.probs,
                            cache.avg_loss);
    return cache;
  }

  /// Run one full forward and backward pass for one batch.
  float forward_backward(const std::vector<int> &ids, const std::vector<int> &targets) {
    zero_grad();

    const ForwardCache cache = forward(ids, targets);
    const std::vector<float> d_block_output =
        backward_logits(cache.feed_forward_norm.output, targets, cache.probs);
    const std::vector<float> d_feed_forward_residual = layer_norm::backward(
        d_block_output, cache.feed_forward_norm, feed_forward_norm_scale, feed_forward_norm_shift);
    const std::vector<float> d_attention_norm_output = feed_forward::backward(
        cache.attention_norm.output, cache.feed_forward, d_feed_forward_residual,
        feed_forward_hidden_weights, feed_forward_hidden_bias, feed_forward_output_weights,
        feed_forward_output_bias);
    const std::vector<float> d_attention_residual = layer_norm::backward(
        d_attention_norm_output, cache.attention_norm, attention_norm_scale, attention_norm_shift);
    const std::vector<float> d_embeddings =
        attention::backward(cache.embeddings, cache.attention, d_attention_residual, query_weights,
                            key_weights, value_weights, attention_output_weights);
    accumulate_embedding_grads(ids, d_embeddings);

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
    attention_norm_scale.update();
    attention_norm_shift.update();
    feed_forward_hidden_weights.update();
    feed_forward_hidden_bias.update();
    feed_forward_output_weights.update();
    feed_forward_output_bias.update();
    feed_forward_norm_scale.update();
    feed_forward_norm_shift.update();
    logit_weights.update();
    output_bias.update();
  }

private:
  /// Build token-plus-position embeddings for one batch.
  std::vector<float> compute_embeddings(const std::vector<int> &ids) const {
    std::vector<float> embeddings(batch_size * context_len * embedding_dim);
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
    return embeddings;
  }

  /// Compute logits, probabilities, and loss from the final block output.
  void compute_logits_and_loss(const std::vector<float> &inputs, const std::vector<int> &targets,
                               std::vector<float> &logits, std::vector<float> &probs,
                               float &avg_loss) const {
    logits.resize(batch_size * context_len * vocab_size);
    probs.resize(batch_size * context_len * vocab_size);
    float loss_sum = 0.0f;

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t in_base = b * context_len * embedding_dim + c * embedding_dim;
        const size_t row_base = b * context_len * vocab_size + c * vocab_size;

        for (size_t i = 0; i < vocab_size; ++i) {
          float logit = output_bias.val[i];
          for (size_t j = 0; j < embedding_dim; ++j) {
            logit += inputs[in_base + j] * logit_weights.val[j * vocab_size + i];
          }
          logits[row_base + i] = logit;
        }

        float max_logit = logits[row_base];
        for (size_t i = 1; i < vocab_size; ++i) {
          max_logit = std::max(max_logit, logits[row_base + i]);
        }

        double sum_exp = 0.0;
        for (size_t i = 0; i < vocab_size; ++i) {
          sum_exp += std::exp(static_cast<double>(logits[row_base + i] - max_logit));
        }

        for (size_t i = 0; i < vocab_size; ++i) {
          probs[row_base + i] = static_cast<float>(
              std::exp(static_cast<double>(logits[row_base + i] - max_logit)) / sum_exp);
        }

        loss_sum += static_cast<float>(max_logit + std::log(sum_exp) -
                                       logits[row_base + targets[b * context_len + c]]);
      }
    }

    avg_loss = loss_sum * inv_token_count;
  }

  /// Backpropagate through the final logits projection and softmax loss.
  std::vector<float> backward_logits(const std::vector<float> &inputs,
                                     const std::vector<int> &targets,
                                     const std::vector<float> &probs) {
    std::vector<float> d_logits = probs;
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        d_logits[b * context_len * vocab_size + c * vocab_size + targets[b * context_len + c]] -=
            1.0f;
      }
    }

    std::vector<float> d_inputs(batch_size * context_len * embedding_dim, 0.0f);
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t in_base = b * context_len * embedding_dim + c * embedding_dim;
        const size_t row_base = b * context_len * vocab_size + c * vocab_size;

        for (size_t i = 0; i < vocab_size; ++i) {
          const float grad = d_logits[row_base + i];
          output_bias.grad[i] += grad;
          for (size_t j = 0; j < embedding_dim; ++j) {
            logit_weights.grad[j * vocab_size + i] += inputs[in_base + j] * grad;
            d_inputs[in_base + j] += grad * logit_weights.val[j * vocab_size + i];
          }
        }
      }
    }

    return d_inputs;
  }

  /// Accumulate token and position embedding gradients from one input gradient tensor.
  void accumulate_embedding_grads(const std::vector<int> &ids,
                                  const std::vector<float> &d_embeddings) {
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t grad_base = b * context_len * embedding_dim + c * embedding_dim;
        const size_t token_id = ids[b * context_len + c];
        const size_t tok_base = token_id * embedding_dim;
        const size_t pos_base = c * embedding_dim;

        for (size_t i = 0; i < embedding_dim; ++i) {
          token_embeddings.grad[tok_base + i] += d_embeddings[grad_base + i];
          position_embeddings.grad[pos_base + i] += d_embeddings[grad_base + i];
        }
      }
    }
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
