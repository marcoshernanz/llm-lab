# Minimal phase-3 build commands.
#
# This Makefile keeps the C++ workflow to one command while the model now spans
# a small set of translation units.

CXX := c++
CXXFLAGS := -std=c++20 -O0 -g -Wall -Wextra -pedantic
BUILD_DIR := build
TARGET := $(BUILD_DIR)/phase3
SOURCES := phase3.cpp attention.cpp feed_forward.cpp rms_norm.cpp decoder_block.cpp decoder.cpp

.PHONY: run build clean

# Build the current phase-3 executable.
build: $(TARGET)

# Compile and run the current single-file script.
run: $(TARGET)
	./$(TARGET)

# Compile the current phase-3 executable.
$(TARGET): $(SOURCES)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

# Delete generated binaries and debug bundles.
clean:
	rm -rf $(BUILD_DIR)
