# Minimal phase-3 build commands.
#
# This Makefile keeps the first C++ workflow to one command while the project is
# still intentionally a single-file executable.

CXX := c++
CXXFLAGS := -std=c++20 -O0 -g -Wall -Wextra -pedantic
BUILD_DIR := build
TARGET := $(BUILD_DIR)/phase3
SOURCE := phase3.cpp

.PHONY: run build clean

# Build the single-file phase-3 executable.
build: $(TARGET)

# Compile and run the current single-file script.
run: $(TARGET)
	./$(TARGET)

# Compile the current single-file executable.
$(TARGET): $(SOURCE)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET)

# Delete generated binaries and debug bundles.
clean:
	rm -rf $(BUILD_DIR)
