/// Shared phase-3 constants and parameter helpers.

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

inline constexpr int vocab_size = 128;
inline constexpr int context_len = 4;
inline constexpr int embedding_dim = 32;
inline constexpr int num_heads = 4;
inline constexpr int head_dim = 16;
inline constexpr int attention_dim = num_heads * head_dim;
inline constexpr int num_decoder_blocks = 4;
inline constexpr int feed_forward_dim = 128;

inline constexpr int steps = 10000;
inline constexpr int steps_per_chunk = 100;
inline constexpr int batch_size = 32;
inline constexpr float inv_token_count = 1.0f / static_cast<float>(batch_size * context_len);
inline constexpr float validation_split = 0.1f;

inline constexpr float learning_rate = 0.01f;
inline constexpr float beta1 = 0.9f;
inline constexpr float beta2 = 0.999f;
inline constexpr float eps = 1e-8f;
inline constexpr float weight_decay = 0.01f;
inline constexpr float rms_norm_eps = 1e-5f;

/// Sample one normal random value for parameter initialization.
float randn();

/// Return the standard deviation used for fan-in scaled weights.
inline float fan_in_stddev(int fan_in) {
  return 1.0f / std::sqrt(static_cast<float>(fan_in));
}

/// Hold AdamW state for one parameter tensor.
class AdamW {
public:
  float beta1_pow = 1.0f;
  float beta2_pow = 1.0f;
  std::vector<float> first_moment;
  std::vector<float> second_moment;

  /// Construct one AdamW state object with zero moments.
  explicit AdamW(size_t size) : first_moment(size, 0.0f), second_moment(size, 0.0f) {}

  /// Apply one AdamW update to a parameter tensor.
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
      values[i] =
          (1.0f - learning_rate * weight_decay) * values[i] -
          learning_rate * corrected_first / (std::sqrt(corrected_second) + eps);
    }
  }
};

/// Hold one trainable tensor, its gradient, and its optimizer state.
class Param {
public:
  std::vector<float> val;
  std::vector<float> grad;
  AdamW optimizer;

  /// Construct one parameter tensor with matching gradient and optimizer state.
  explicit Param(size_t size) : val(size), grad(size, 0.0f), optimizer(size) {}

  /// Fill the parameter tensor with normal values of one chosen scale.
  void init_normal(float stddev) {
    for (float &x : val) {
      x = randn() * stddev;
    }
  }

  /// Fill the parameter tensor with zeros.
  void init_zeros() { std::fill(val.begin(), val.end(), 0.0f); }

  /// Fill the parameter tensor with ones.
  void init_ones() { std::fill(val.begin(), val.end(), 1.0f); }

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
