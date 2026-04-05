# Minimal phase-3 build commands.
#
# This Makefile keeps the first C++ workflow to one command while the project is
# still intentionally a single-file executable.

CXX := c++
CXXFLAGS := -std=c++20 -O0 -g -Wall -Wextra -pedantic
BUILD_DIR := build
TARGET := $(BUILD_DIR)/phase3
SOURCE := phase3.cpp
DATASET := ../datasets/tinyshakespeare.txt

.PHONY: run smoke data build clean

# Build the single-file phase-3 executable.
build: $(TARGET)

# Compile and run the tiny numerical smoke check.
run: smoke

# Compile and run the softmax and cross-entropy demo.
smoke: $(TARGET)
	./$(TARGET) smoke

# Compile and run the text-loading demo on tiny Shakespeare.
data: $(TARGET)
	./$(TARGET) data $(DATASET)

# Compile the current single-file executable.
$(TARGET): $(SOURCE)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET)

# Delete generated binaries and debug bundles.
clean:
	rm -rf $(BUILD_DIR)
