#!/usr/bin/env bash
set -euo pipefail

# [BUILD] Step 1: Build C++ applications
# This script builds all C++ applications needed for the poisoning experiment

# Load common environment
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

echo "[BUILD] Step 1: Building C++ applications..."

# Clean up build directory
# rm -rf "$BUILD_DIR"

# Build applications
cd "$SCRIPT_DIR"
./build.sh
cd "$BUILD_DIR"

# Verify executables exist
exes=(
    calc_loss
    calc_optimal_poison
    calc_optimal_poison_duplicate_allowed
    calc_upper_bound_binary
    calc_upper_bound_golden
    calc_upper_bound_strict
    gen_real
    gen_sync
    inject_poison
    inject_poison_consecutive_w_endpoints
    inject_poison_consecutive_w_endpoints_duplicate_allowed
    inject_poison_consecutive_w_endpoints_using_relaxed_solution
    inject_poison_duplicate_allowed
)

for exe in "${exes[@]}"; do
    if [ ! -f "$exe" ]; then
        echo "[ERROR] Error: $exe not found in $BUILD_DIR"
        echo "Build may have failed. Please check build output."
        exit 1
    fi
done
