/// Minimal artifact logging helpers for the fixed phase-3 trainer.

#include "artifact_logging.h"

#include "core.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>

const std::filesystem::path artifacts_root = "../artifacts/phase3/cpu_reference";

/// Return one timestamp-based run id for a new artifact directory.
std::string build_run_id() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  const std::tm local_time = *std::localtime(&now_time);
  const auto milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;

  std::ostringstream run_id;
  run_id << std::put_time(&local_time, "%Y%m%d_%H%M%S") << "_" << std::setw(3) << std::setfill('0')
         << milliseconds;
  return run_id.str();
}

/// Create one fresh run directory for the fixed phase-3 trainer.
ArtifactPaths create_artifact_paths() {
  ArtifactPaths paths;
  paths.run_id = build_run_id();
  paths.run_dir = artifacts_root / paths.run_id;
  paths.metrics_csv = paths.run_dir / "metrics.csv";
  paths.metadata_json = paths.run_dir / "run_metadata.json";
  paths.profile_summary_csv = paths.run_dir / "profile_summary.csv";

  std::filesystem::create_directories(paths.run_dir);
  return paths;
}

/// Create the run directory, open the metrics CSV, and save fixed metadata.
ArtifactLogger::ArtifactLogger(const std::string &corpus_path, size_t parameter_count)
    : paths(create_artifact_paths()), metrics_file(paths.metrics_csv) {
  if (!metrics_file) {
    throw std::runtime_error("could not open metrics artifact file");
  }
  write_metrics_header();
  write_metadata(corpus_path, parameter_count);
}

/// Append one chunk-level measurement row to the metrics CSV.
void ArtifactLogger::log_chunk(int start_step, int chunk_steps, float train_loss, float val_loss,
                               double chunk_seconds) {
  const double step_time_ms = chunk_steps > 0 ? chunk_seconds * 1000.0 / static_cast<double>(chunk_steps)
                                              : 0.0;
  const double train_tokens =
      static_cast<double>(chunk_steps) * static_cast<double>(batch_size * context_len);
  const double tokens_per_second = chunk_seconds > 0.0 ? train_tokens / chunk_seconds : 0.0;

  metrics_file << start_step << "," << train_loss << "," << val_loss << "," << step_time_ms << ","
               << tokens_per_second << "\n";
  metrics_file.flush();
}

/// Write the fixed CSV header for chunk-level loss and throughput measurements.
void ArtifactLogger::write_metrics_header() {
  metrics_file << "step,train_loss,val_loss,step_time_ms,tokens_per_second\n";
  metrics_file.flush();
}

/// Save the fixed run metadata needed to compare later phase-3 baselines.
void ArtifactLogger::write_metadata(const std::string &corpus_path, size_t parameter_count) const {
  std::ofstream metadata_file(paths.metadata_json);
  if (!metadata_file) {
    throw std::runtime_error("could not open metadata artifact file");
  }

  metadata_file << "{\n";
  metadata_file << "  \"run_id\": \"" << paths.run_id << "\",\n";
  metadata_file << "  \"corpus_path\": \"" << corpus_path << "\",\n";
  metadata_file << "  \"parameter_count\": " << parameter_count << ",\n";
  metadata_file << "  \"steps\": " << steps << ",\n";
  metadata_file << "  \"steps_per_chunk\": " << steps_per_chunk << ",\n";
  metadata_file << "  \"batch_size\": " << batch_size << ",\n";
  metadata_file << "  \"context_len\": " << context_len << ",\n";
  metadata_file << "  \"train_tokens_per_step\": " << batch_size * context_len << ",\n";
  metadata_file << "  \"validation_tokens_per_step\": " << batch_size * context_len << ",\n";
  metadata_file << "  \"vocab_size\": " << vocab_size << ",\n";
  metadata_file << "  \"embedding_dim\": " << embedding_dim << ",\n";
  metadata_file << "  \"num_heads\": " << num_heads << ",\n";
  metadata_file << "  \"head_dim\": " << head_dim << ",\n";
  metadata_file << "  \"num_decoder_blocks\": " << num_decoder_blocks << ",\n";
  metadata_file << "  \"feed_forward_dim\": " << feed_forward_dim << ",\n";
  metadata_file << "  \"learning_rate\": " << learning_rate << ",\n";
  metadata_file << "  \"beta1\": " << beta1 << ",\n";
  metadata_file << "  \"beta2\": " << beta2 << ",\n";
  metadata_file << "  \"eps\": " << eps << ",\n";
  metadata_file << "  \"weight_decay\": " << weight_decay << "\n";
  metadata_file << "}\n";
}
