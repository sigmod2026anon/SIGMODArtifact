#!/usr/bin/env bash
set -euo pipefail

# Step 2: Generate datasets
# This script generates real and synthetic datasets for the poisoning experiment

# Load common environment
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

# Configuration
ALL_MODE=false
QUICK_MODE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            ALL_MODE=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 --all|--quick"
            echo ""
            echo "[DATA] Dataset Generation Tool"
            echo ""
            echo "Options:"
            echo "  --all      Generate large datasets with all parameters"
            echo "  --quick    Generate small test datasets"
            echo "  --help,-h  Show this help message"
            echo ""
            echo "This script uses parameters defined in env.sh and passes them to gen_real and gen_sync programs."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --all|--quick"
            exit 1
            ;;
    esac
done

# Check if either --all or --quick was specified
if [ "$ALL_MODE" = false ] && [ "$QUICK_MODE" = false ]; then
    echo "Error: Either --all or --quick option must be specified"
    echo "Usage: $0 --all|--quick"
    exit 1
fi

echo "[DATA] Step 2: Generating datasets..."
echo "  Data directory: $DATA_DIR"
echo "  Mode: $([ "$ALL_MODE" = true ] && echo "ALL" || ([ "$QUICK_MODE" = true ] && echo "QUICK" || echo "DEFAULT"))"
echo ""

# Create data directory
mkdir -p "$DATA_DIR"

# Change to build directory
cd "$BUILD_DIR"

# Helper function to join array elements with comma
join_by_comma() {
    local IFS=','
    echo "$*"
}

# Set parameters based on mode
if [ "$ALL_MODE" = true ]; then
    # Use all_ prefixed arrays
    real_datasets_param=$(join_by_comma "${all_real_dataset_names[@]}")
    sync_datasets_param=$(join_by_comma "${all_sync_dataset_names[@]}")
    ns_param=$(join_by_comma "${all_ns[@]}")
    seeds_param=$(join_by_comma "${all_seeds[@]}")
    Rs_param=$(join_by_comma "${all_Rs[@]}")
    brute_force_ns_param=$(join_by_comma "${all_brute_force_ns[@]}")
    brute_force_Rs_param=$(join_by_comma "${all_brute_force_Rs[@]}")
else
    # Use quick_ prefixed arrays
    real_datasets_param=$(join_by_comma "${quick_real_dataset_names[@]}")
    sync_datasets_param=$(join_by_comma "${quick_sync_dataset_names[@]}")
    ns_param=$(join_by_comma "${quick_ns[@]}")
    seeds_param=$(join_by_comma "${quick_seeds[@]}")
    Rs_param=$(join_by_comma "${quick_Rs[@]}")
    brute_force_ns_param=$(join_by_comma "${quick_brute_force_ns[@]}")
    brute_force_Rs_param=$(join_by_comma "${quick_brute_force_Rs[@]}")
fi

# Generate real datasets
echo "  Generating real datasets..."
echo "    Parameters:"
echo "      Real datasets: $real_datasets_param"
echo "      Sample sizes: $ns_param"
echo "      Seeds: $seeds_param"
echo ""

./gen_real \
    --real_dataset_names "$real_datasets_param" \
    --ns "$ns_param" \
    --seeds "$seeds_param"

echo "  Generating real brute force datasets..."
echo "    Parameters:"
echo "      Real datasets: $real_datasets_param"
echo "      Sample sizes: $brute_force_ns_param"
echo "      Seeds: $seeds_param"
echo ""

./gen_real \
    --real_dataset_names "$real_datasets_param" \
    --ns "$brute_force_ns_param" \
    --seeds "$seeds_param"

# Generate synthetic datasets  
echo "  [SYNTH] Generating synthetic datasets for brute force..."
echo "    Parameters:"
echo "      Sync datasets: $sync_datasets_param"
echo "      Sample sizes: $ns_param"
echo "      Seeds: $seeds_param"
echo "      Range values: $Rs_param"
echo ""

## base_n, Rs_param
./gen_sync \
    --sync_dataset_names "$sync_datasets_param" \
    --ns "$base_n" \
    --seeds "$seeds_param" \
    --Rs "$Rs_param"

## ns_param, base_R
./gen_sync \
    --sync_dataset_names "$sync_datasets_param" \
    --ns "$ns_param" \
    --seeds "$seeds_param" \
    --Rs "$base_R"

echo "  [SYNTH] Generating synthetic datasets for brute force..."
echo "    Parameters:"
echo "      Sync datasets: $sync_datasets_param"
echo "      Sample sizes: $brute_force_ns_param"
echo "      Seeds: $seeds_param"
echo "      Range values: $brute_force_Rs_param"
echo ""

## base_brute_force_n, brute_force_Rs_param
./gen_sync \
    --sync_dataset_names "$sync_datasets_param" \
    --ns "$base_brute_force_n" \
    --seeds "$seeds_param" \
    --Rs "$brute_force_Rs_param"

## brute_force_ns_param, base_brute_force_R
./gen_sync \
    --sync_dataset_names "$sync_datasets_param" \
    --ns "$brute_force_ns_param" \
    --seeds "$seeds_param" \
    --Rs "$base_brute_force_R"

echo "[OK] Dataset generation completed"
echo "  Data directory: $DATA_DIR"
echo "  Generated files:"
if [ -d "$DATA_DIR" ]; then
    real_count=$(find "$DATA_DIR" -name "*books*" -o -name "*fb*" -o -name "*osm*" | wc -l)
    sync_count=$(find "$DATA_DIR" -name "*uniform*" -o -name "*normal*" -o -name "*exponential*" | wc -l)
    echo "    [REAL] Real datasets: $real_count"
    echo "    [SYNTH] Synthetic datasets: $sync_count"
fi
echo ""