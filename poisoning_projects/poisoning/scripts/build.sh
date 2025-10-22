#!/usr/bin/env bash
set -euo pipefail

# Load common environment
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

echo "[BUILD] Building poisoning C++ tools (minimal dependencies)..."

# Check dependencies
check_dependencies() {
    local missing_deps=()
    
    # Check basic C++ compiler
    if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
        missing_deps+=("C++ compiler (g++ or clang++)")
    fi
    
    if ! command -v cmake &> /dev/null; then
        missing_deps+=("cmake")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo "[ERROR] Missing dependencies:"
        printf "   - %s\n" "${missing_deps[@]}"
        echo ""
        echo "Install with:"
        echo "   Ubuntu/Debian: sudo apt-get install build-essential cmake"
        echo "   CentOS/RHEL:   sudo yum install gcc-c++ cmake"
        echo "   macOS:         brew install cmake"
        exit 1
    fi
}

check_dependencies

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "[DIR] Build directory: $BUILD_DIR"
echo "[CMAKE] Configuring with CMake..."

# Configure with CMake (minimal configuration)
cmake .. -DCMAKE_BUILD_TYPE=Release

echo "[BUILD] Building..."

# Build
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
make -j$NPROC

if [ $? -eq 0 ]; then
    echo "[OK] Build completed successfully!"
    echo ""
    echo "[DIR] Executables in: $BUILD_DIR"
    echo ""
    echo "Available commands:"
    echo "  - calc_loss"
    echo "  - calc_optimal_poison"
    echo "  - calc_optimal_poison_duplicate_allowed"
    echo "  - calc_upper_bound_binary"
    echo "  - calc_upper_bound_golden"
    echo "  - calc_upper_bound_strict"
    echo "  - gen_real"
    echo "  - gen_sync"
    echo "  - inject_poison"
    echo "  - inject_poison_consecutive_w_endpoints"
    echo "  - inject_poison_consecutive_w_endpoints_duplicate_allowed"
    echo "  - inject_poison_consecutive_w_endpoints_using_relaxed_solution"
    echo "  - inject_poison_duplicate_allowed"
    echo ""
else
    echo "[ERROR] Build failed!"
    exit 1
fi
