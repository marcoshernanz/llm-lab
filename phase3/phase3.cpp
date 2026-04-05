/// Minimal phase-3 bootstrap executable.
///
/// This file keeps the first C++ step narrow: one executable, one translation
/// unit, and a few inspectable subcommands.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

/// Print the supported subcommands for the phase-3 bootstrap.
void print_usage(std::ostream& out) {
    out << "Usage:\n";
    out << "  ./build/phase3 smoke\n";
    out << "  ./build/phase3 data <text-file>\n";
}

/// Read a whole text file into memory for small data-path experiments.
std::string read_text_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("could not open file: " + path);
    }

    return std::string(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()
    );
}

/// Convert logits into probabilities with the standard max-shift trick.
std::vector<double> softmax(const std::vector<double>& logits) {
    const double max_logit = *std::max_element(logits.begin(), logits.end());

    std::vector<double> shifted_exp;
    shifted_exp.reserve(logits.size());

    double sum = 0.0;
    for (const double logit : logits) {
        const double value = std::exp(logit - max_logit);
        shifted_exp.push_back(value);
        sum += value;
    }

    for (double& value : shifted_exp) {
        value /= sum;
    }

    return shifted_exp;
}

/// Return the negative log-likelihood for one target token index.
double cross_entropy_loss(const std::vector<double>& logits, std::size_t target_index) {
    if (target_index >= logits.size()) {
        throw std::runtime_error("target index is out of range");
    }

    const std::vector<double> probabilities = softmax(logits);
    return -std::log(probabilities[target_index]);
}

/// Run a tiny numerical demo that mirrors later trainer math.
int run_smoke_demo() {
    const std::vector<double> logits = {1.2, 0.1, -0.3, 2.4};
    const std::size_t target_index = 3;

    const std::vector<double> probabilities = softmax(logits);
    const double loss = cross_entropy_loss(logits, target_index);

    std::cout << "logits:";
    for (const double logit : logits) {
        std::cout << ' ' << std::fixed << std::setprecision(4) << logit;
    }
    std::cout << '\n';

    std::cout << "probabilities:";
    for (const double probability : probabilities) {
        std::cout << ' ' << std::fixed << std::setprecision(4) << probability;
    }
    std::cout << '\n';

    std::cout << "target_index: " << target_index << '\n';
    std::cout << "cross_entropy_loss: " << std::fixed << std::setprecision(6) << loss
              << '\n';

    return 0;
}

/// Run a tiny data-path demo that inspects one text file end to end.
int run_data_demo(const std::string& path) {
    const std::string text = read_text_file(path);

    std::cout << "path: " << path << '\n';
    std::cout << "bytes: " << text.size() << '\n';
    std::cout << "preview:\n";
    std::cout << text.substr(0, 120) << '\n';

    return 0;
}

/// Dispatch one of the minimal phase-3 bootstrap commands.
int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            print_usage(std::cerr);
            return 1;
        }

        const std::string_view command = argv[1];

        if (command == "smoke") {
            return run_smoke_demo();
        }

        if (command == "data") {
            if (argc < 3) {
                throw std::runtime_error("data command needs a file path");
            }

            return run_data_demo(argv[2]);
        }

        throw std::runtime_error("unknown command: " + std::string(command));
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        print_usage(std::cerr);
        return 1;
    }
}
