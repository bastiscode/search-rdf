#!/bin/bash
# Test runner for model tests that require Python
# Usage: ./test-model.sh [test_path]

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Use first argument as test path, default to model::text::local
TEST_PATH="${1:-model::text}"
shift 2>/dev/null # Remove first argument if present

# Run tests sequentially with single thread
cargo test --lib "$TEST_PATH" -- --ignored --test-threads=1 "$@"
