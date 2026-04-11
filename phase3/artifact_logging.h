/// Minimal artifact logging helpers for the fixed phase-3 trainer.

#pragma once

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>

/// Hold the file paths for one phase-3 artifact run directory.
struct ArtifactPaths {
  std::string run_id;
  std::filesystem::path run_dir;
  std::filesystem::path metrics_csv;
  std::filesystem::path metadata_json;
  std::filesystem::path profile_summary_csv;
};

/// Hold one minimal artifact logger for the fixed phase-3 trainer.
class ArtifactLogger {
public:
  ArtifactPaths paths;

  /// Create the run directory, open the metrics CSV, and save fixed metadata.
  ArtifactLogger(const std::string &corpus_path, size_t parameter_count);

  /// Append one chunk-level measurement row to the metrics CSV.
  void log_chunk(int start_step, int chunk_steps, float train_loss, float val_loss,
                 double chunk_seconds);

private:
  std::ofstream metrics_file;

  /// Write the fixed CSV header for chunk-level loss and throughput measurements.
  void write_metrics_header();

  /// Save the fixed run metadata needed to compare later phase-3 baselines.
  void write_metadata(const std::string &corpus_path, size_t parameter_count) const;
};
