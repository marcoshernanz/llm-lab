/// Minimal inclusive-timing profiler for the fixed phase-3 trainer.

#include "profiler.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <unordered_map>

namespace profiler {

namespace {

/// Hold the running totals for one named code section.
struct SectionStats {
  double total_seconds = 0.0;
  size_t calls = 0;
};

/// Hold the current profiling session across one trainer run.
class Session {
public:
  /// Clear all accumulated timing data from the current session.
  void reset() { sections.clear(); }

  /// Return whether the current session contains any recorded samples.
  bool has_samples() const { return !sections.empty(); }

  /// Add one inclusive timing sample to the named section.
  void add_sample(const char *section_name, double elapsed_seconds) {
    SectionStats &stats = sections[section_name];
    stats.total_seconds += elapsed_seconds;
    stats.calls += 1;
  }

  /// Return the current summary rows sorted from slowest to fastest.
  std::vector<SummaryRow> summary(double wall_seconds) const {
    std::vector<SummaryRow> rows;
    rows.reserve(sections.size());
    for (const auto &[section, stats] : sections) {
      const double wall_share_percent =
          wall_seconds > 0.0 ? 100.0 * stats.total_seconds / wall_seconds : 0.0;
      const double average_milliseconds =
          stats.calls > 0 ? 1000.0 * stats.total_seconds / static_cast<double>(stats.calls) : 0.0;
      rows.push_back(SummaryRow{.section = section,
                                .total_seconds = stats.total_seconds,
                                .wall_share_percent = wall_share_percent,
                                .calls = stats.calls,
                                .average_milliseconds = average_milliseconds});
    }

    std::sort(rows.begin(), rows.end(),
              [](const SummaryRow &left, const SummaryRow &right) {
                return left.total_seconds > right.total_seconds;
              });
    return rows;
  }

private:
  std::unordered_map<std::string, SectionStats> sections;
};

/// Return the process-wide profiling session used by the phase-3 trainer.
Session &current_session() {
  static Session session;
  return session;
}

} // namespace

/// Reset the current profiling session to an empty state.
void reset() { current_session().reset(); }

/// Return whether any profiled section has recorded at least one sample.
bool has_samples() { return current_session().has_samples(); }

/// Return the current profiling summary sorted by total inclusive time.
std::vector<SummaryRow> summary(double wall_seconds) {
  return current_session().summary(wall_seconds);
}

/// Write the current profiling summary to one CSV artifact file.
void write_summary_csv(const std::filesystem::path &csv_path, double wall_seconds) {
  std::ofstream file(csv_path);
  if (!file) {
    throw std::runtime_error("could not open profile summary artifact file");
  }

  file << "section,total_seconds,wall_share_percent,calls,average_milliseconds\n";
  for (const SummaryRow &row : summary(wall_seconds)) {
    file << row.section << "," << row.total_seconds << "," << row.wall_share_percent << ","
         << row.calls << "," << row.average_milliseconds << "\n";
  }
}

/// Print the slowest profiled sections to one output stream.
void print_summary(std::ostream &stream, double wall_seconds, size_t max_rows) {
  const std::vector<SummaryRow> rows = summary(wall_seconds);
  stream << "profile_note=inclusive_timers_nested_sections_can_sum_above_100_percent\n";
  for (size_t i = 0; i < std::min(max_rows, rows.size()); ++i) {
    const SummaryRow &row = rows[i];
    stream << std::fixed << std::setprecision(6) << "profile_section=" << row.section
           << " total_seconds=" << row.total_seconds
           << " wall_share_percent=" << row.wall_share_percent << " calls=" << row.calls
           << " avg_ms=" << row.average_milliseconds << "\n";
  }
}

/// Start measuring one named section until this scope object is destroyed.
Scope::Scope(const char *section_name)
    : section_name(section_name), start_time(std::chrono::steady_clock::now()) {}

/// Stop measuring and record the elapsed time in the current session.
Scope::~Scope() {
  const double elapsed_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
  current_session().add_sample(section_name, elapsed_seconds);
}

} // namespace profiler
