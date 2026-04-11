/// Minimal phase-3 script for learning manual language-model gradients.

#include "artifact_logging.h"
#include "core.h"
#include "decoder.h"
#include "profiler.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
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
  std::vector<float> input_embeddings;
  decoder::Cache decoder_cache;
  std::vector<float> logits;
  std::vector<float> probs;
  float avg_loss = 0.0f;
};

/// Hold the trainable tensors for the tiny language model.
class Model {
public:
  Param token_embedding_table;
  Param position_embedding_table;
  decoder::Stack decoder_stack;
  Param lm_head_bias;

  /// Construct one model with correctly sized parameter tensors.
  Model()
      : token_embedding_table(vocab_size * embedding_dim),
        position_embedding_table(context_len * embedding_dim), decoder_stack(), lm_head_bias(vocab_size) {}

  /// Initialize one model with random weights and zero biases.
  static Model init() {
    Model model;
    model.token_embedding_table.init_normal(0.1f);
    model.position_embedding_table.init_normal(0.1f);
    model.decoder_stack.init();
    model.lm_head_bias.init_zeros();
    return model;
  }

  /// Reset every parameter gradient buffer to zero.
  void zero_grad() {
    token_embedding_table.zero_grad();
    position_embedding_table.zero_grad();
    decoder_stack.zero_grad();
    lm_head_bias.zero_grad();
  }

  /// Scale every parameter gradient buffer by one constant.
  void scale_grads(float scale) {
    token_embedding_table.scale_grad(scale);
    position_embedding_table.scale_grad(scale);
    decoder_stack.scale_grads(scale);
    lm_head_bias.scale_grad(scale);
  }

  /// Run one full forward pass and keep the tensors needed for backprop.
  void forward(const std::vector<int> &ids, const std::vector<int> &targets,
               ForwardCache &cache) const {
    const profiler::Scope scope("model.forward");
    compute_input_embeddings(ids, cache.input_embeddings);
    decoder_stack.forward(cache.input_embeddings, cache.decoder_cache);
    compute_logits_and_loss(cache.decoder_cache.decoder_output, targets, cache.logits, cache.probs,
                            cache.avg_loss);
  }

  /// Run one full forward and backward pass for one batch.
  float forward_backward(const std::vector<int> &ids, const std::vector<int> &targets) {
    const profiler::Scope scope("model.forward_backward");
    zero_grad();

    forward(ids, targets, forward_cache);
    backward_logits(forward_cache.decoder_cache.decoder_output, targets, forward_cache.probs,
                    d_block_output);
    decoder_stack.backward(forward_cache.decoder_cache, d_block_output, d_embeddings);
    accumulate_embedding_grads(ids, d_embeddings);

    scale_grads(inv_token_count);
    return forward_cache.avg_loss;
  }

  /// Compute the average loss for one batch without building gradients.
  float forward_loss(const std::vector<int> &ids, const std::vector<int> &targets) const {
    const profiler::Scope scope("model.forward_loss");
    forward(ids, targets, forward_cache);
    return forward_cache.avg_loss;
  }

  /// Apply one optimizer step to every parameter tensor.
  void update() {
    const profiler::Scope scope("model.update");
    token_embedding_table.update();
    position_embedding_table.update();
    decoder_stack.update();
    lm_head_bias.update();
  }

  /// Return the total number of trainable scalar parameters in the model.
  size_t parameter_count() const {
    size_t total = token_embedding_table.val.size() + position_embedding_table.val.size() +
                   lm_head_bias.val.size();
    for (const decoder_block::Block &block : decoder_stack.blocks) {
      total += block.attention_query_weights.val.size();
      total += block.attention_key_weights.val.size();
      total += block.attention_value_weights.val.size();
      total += block.attention_output_projection_weights.val.size();
      total += block.attention_rms_gain.val.size();
      total += block.feed_forward_in_weights.val.size();
      total += block.feed_forward_in_bias.val.size();
      total += block.feed_forward_out_weights.val.size();
      total += block.feed_forward_out_bias.val.size();
      total += block.feed_forward_rms_gain.val.size();
    }
    return total;
  }

private:
  /// Build token-plus-position embeddings for one batch.
  void compute_input_embeddings(const std::vector<int> &ids, std::vector<float> &embeddings) const {
    const profiler::Scope scope("model.compute_input_embeddings");
    embeddings.resize(batch_size * context_len * embedding_dim);
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t out_base = b * context_len * embedding_dim + c * embedding_dim;
        const size_t token_id = ids[b * context_len + c];
        const size_t tok_base = token_id * embedding_dim;
        const size_t pos_base = c * embedding_dim;

        for (size_t i = 0; i < embedding_dim; ++i) {
          embeddings[out_base + i] =
              token_embedding_table.val[tok_base + i] + position_embedding_table.val[pos_base + i];
        }
      }
    }
  }

  /// Compute logits, probabilities, and loss from the final block output.
  void compute_logits_and_loss(const std::vector<float> &inputs, const std::vector<int> &targets,
                               std::vector<float> &logits, std::vector<float> &probs,
                               float &avg_loss) const {
    const profiler::Scope scope("model.compute_logits_and_loss");
    logits.resize(batch_size * context_len * vocab_size);
    probs.resize(batch_size * context_len * vocab_size);
    float loss_sum = 0.0f;

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t in_base = b * context_len * embedding_dim + c * embedding_dim;
        const size_t row_base = b * context_len * vocab_size + c * vocab_size;

        for (size_t i = 0; i < vocab_size; ++i) {
          float logit = lm_head_bias.val[i];
          for (size_t j = 0; j < embedding_dim; ++j) {
            logit += inputs[in_base + j] * token_embedding_table.val[i * embedding_dim + j];
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
  void backward_logits(const std::vector<float> &inputs, const std::vector<int> &targets,
                       const std::vector<float> &probs, std::vector<float> &d_inputs) {
    const profiler::Scope scope("model.backward_logits");
    copy_into(d_logits, probs);
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        d_logits[b * context_len * vocab_size + c * vocab_size + targets[b * context_len + c]] -=
            1.0f;
      }
    }

    resize_and_zero(d_inputs, batch_size * context_len * embedding_dim);
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t in_base = b * context_len * embedding_dim + c * embedding_dim;
        const size_t row_base = b * context_len * vocab_size + c * vocab_size;

        for (size_t i = 0; i < vocab_size; ++i) {
          const float grad = d_logits[row_base + i];
          lm_head_bias.grad[i] += grad;
          for (size_t j = 0; j < embedding_dim; ++j) {
            token_embedding_table.grad[i * embedding_dim + j] += inputs[in_base + j] * grad;
            d_inputs[in_base + j] += grad * token_embedding_table.val[i * embedding_dim + j];
          }
        }
      }
    }
  }

  /// Accumulate token and position embedding gradients from one input gradient tensor.
  void accumulate_embedding_grads(const std::vector<int> &ids,
                                  const std::vector<float> &d_embeddings) {
    const profiler::Scope scope("model.accumulate_embedding_grads");
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < context_len; ++c) {
        const size_t grad_base = b * context_len * embedding_dim + c * embedding_dim;
        const size_t token_id = ids[b * context_len + c];
        const size_t tok_base = token_id * embedding_dim;
        const size_t pos_base = c * embedding_dim;

        for (size_t i = 0; i < embedding_dim; ++i) {
          token_embedding_table.grad[tok_base + i] += d_embeddings[grad_base + i];
          position_embedding_table.grad[pos_base + i] += d_embeddings[grad_base + i];
        }
      }
    }
  }

  /// Reuse one forward cache and gradient buffers across training steps.
  mutable ForwardCache forward_cache;
  std::vector<float> d_block_output;
  std::vector<float> d_embeddings;
  std::vector<float> d_logits;
};

/// Sample one batch of context windows and next-token targets.
void generate_batch(int min, int max, std::vector<int> &ids, std::vector<int> &targets,
                    const std::vector<int> &token_ids) {
  const profiler::Scope scope("train.generate_batch");
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
  const profiler::Scope scope("train.run");
  const int split_index =
      static_cast<int>(std::floor(token_ids.size() * (1.0f - validation_split)));
  ArtifactLogger artifact_logger(corpus_path, model.parameter_count());
  profiler::reset();
  const auto training_start = std::chrono::steady_clock::now();

  std::vector<int> ids(batch_size * context_len);
  std::vector<int> targets(batch_size * context_len);

  std::cout << "run_dir=" << artifact_logger.paths.run_dir.string() << "\n";
  std::cout << "metrics_csv=" << artifact_logger.paths.metrics_csv.string() << "\n";
  std::cout << "metadata_json=" << artifact_logger.paths.metadata_json.string() << "\n";
  std::cout << "profile_summary_csv=" << artifact_logger.paths.profile_summary_csv.string() << "\n";
  std::cout << "parameter_count=" << model.parameter_count() << "\n";

  for (int start_step = 0; start_step < steps; start_step += steps_per_chunk) {
    const int chunk_steps = std::min(steps_per_chunk, steps - start_step);
    float train_loss = 0.0f;
    float val_loss = 0.0f;
    const auto chunk_start = std::chrono::steady_clock::now();

    for (int step = 0; step < chunk_steps; ++step) {
      {
        const profiler::Scope training_step_scope("train.forward_backward_step");
        generate_batch(0, split_index - context_len, ids, targets, token_ids);
        train_loss += model.forward_backward(ids, targets);
      }

      {
        const profiler::Scope validation_step_scope("train.validation_step");
        generate_batch(split_index, static_cast<int>(token_ids.size()) - context_len, ids, targets,
                       token_ids);
        val_loss += model.forward_loss(ids, targets);
      }

      {
        const profiler::Scope optimizer_step_scope("train.optimizer_step");
        model.update();
      }
    }

    const double chunk_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - chunk_start).count();
    const double step_time_ms = chunk_seconds * 1000.0 / static_cast<double>(chunk_steps);
    const double tokens_per_second =
        chunk_seconds > 0.0
            ? static_cast<double>(chunk_steps) * static_cast<double>(batch_size * context_len) /
                  chunk_seconds
            : 0.0;
    const float avg_train_loss = train_loss / chunk_steps;
    const float avg_val_loss = val_loss / chunk_steps;

    artifact_logger.log_chunk(start_step, chunk_steps, avg_train_loss, avg_val_loss, chunk_seconds);
    std::cout << std::fixed << std::setprecision(6) << "step=" << start_step
              << " train_loss=" << avg_train_loss << " val_loss=" << avg_val_loss
              << " step_time_ms=" << step_time_ms << " tokens_per_second=" << tokens_per_second
              << "\n";
  }

  const double training_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - training_start).count();
  if (profiler::has_samples()) {
    profiler::write_summary_csv(artifact_logger.paths.profile_summary_csv, training_seconds);
    profiler::print_summary(std::cout, training_seconds, 12);
  }
}

/// Initialize the toy model and train it.
int main() {
  Model model = Model::init();

  const std::string corpus = load_corpus();
  const std::vector<int> token_ids = prepare_vocab(corpus);
  run_training(model, token_ids);
}
