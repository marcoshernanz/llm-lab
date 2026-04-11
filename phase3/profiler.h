/// Minimal inclusive-timing profiler for the fixed phase-3 trainer.

#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <ostream>
#include <string>
#include <vector>

namespace profiler {

/// Hold the aggregated timing statistics for one named code section.
struct SummaryRow {
  std::string section;
  double total_seconds = 0.0;
  double wall_share_percent = 0.0;
  size_t calls = 0;
  double average_milliseconds = 0.0;
};

/// Reset the current profiling session to an empty state.
void reset();

/// Return whether any profiled section has recorded at least one sample.
bool has_samples();

/// Return the current profiling summary sorted by total inclusive time.
std::vector<SummaryRow> summary(double wall_seconds);

/// Write the current profiling summary to one CSV artifact file.
void write_summary_csv(const std::filesystem::path &csv_path, double wall_seconds);

/// Print the slowest profiled sections to one output stream.
void print_summary(std::ostream &stream, double wall_seconds, size_t max_rows);

/// Measure one scope and add its elapsed time to the current profiling session.
class Scope {
public:
  /// Start measuring one named section until this scope object is destroyed.
  explicit Scope(const char *section_name);

  Scope(const Scope &) = delete;
  Scope &operator=(const Scope &) = delete;

  /// Stop measuring and record the elapsed time in the current session.
  ~Scope();

private:
  const char *section_name;
  std::chrono::steady_clock::time_point start_time;
};

} // namespace profiler
