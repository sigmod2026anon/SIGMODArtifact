#!/usr/bin/env bash
set -euo pipefail

# [CALC] Step 5.1: Calculate optimal poison values using brute force (duplicate not allowed)
# This script calculates optimal poison values using brute force search

# Load common environment
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

# Debug output
echo "DEBUG: SCRIPT_DIR=$SCRIPT_DIR"
echo "DEBUG: PROJECT_ROOT=$PROJECT_ROOT"
echo "DEBUG: BUILD_DIR=$BUILD_DIR"
echo "DEBUG: DATA_DIR=$DATA_DIR"
echo "DEBUG: RESULTS_DIR=$RESULTS_DIR"
echo ""

# Configuration
# Use presets from env.sh
brute_force_POISONING_PERCENTAGES=("${all_brute_force_POISONING_PERCENTAGES[@]}")
real_dataset_names=("${all_real_dataset_names[@]}")
sync_dataset_names=("${all_sync_dataset_names[@]}")
brute_force_ns=("${all_brute_force_ns[@]}")
seeds=("${all_seeds[@]}")
brute_force_Rs=("${all_brute_force_Rs[@]}")

# Parse command line arguments
ALL_MODE=false
QUICK_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            ALL_MODE=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            brute_force_POISONING_PERCENTAGES=("${quick_brute_force_POISONING_PERCENTAGES[@]}")
            real_dataset_names=("${quick_real_dataset_names[@]}")
            sync_dataset_names=("${quick_sync_dataset_names[@]}")
            brute_force_ns=("${quick_brute_force_ns[@]}")
            seeds=("${quick_seeds[@]}")
            brute_force_Rs=("${quick_brute_force_Rs[@]}")
            shift
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

echo "[CALC] Step 5.1: Calculating optimal poison values using brute force (duplicate not allowed)..."
echo "  Data directory: $DATA_DIR"
echo "  Results directory: $RESULTS_DIR"
echo "  Brute force poisoning percentages: ${brute_force_POISONING_PERCENTAGES[@]}"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR/optimal_poison"

# Change to build directory
cd "$BUILD_DIR"

run_calc_optimal_poison() {
    input_file_name="$1"
    poison_num="$2"
    base_output_name="$3"
    dataset_name="$4"
    n="$5"
    if [ ! -f "$DATA_DIR/${input_file_name}" ]; then
        echo "    Input file not found: $DATA_DIR/${input_file_name}"
        return
    fi
    
    # Check if output JSON file already exists
    json_file="$RESULTS_DIR/optimal_poison/${dataset_name}/n${n}/lambda${poison_num}/${base_output_name}_brute_force.json"
    mkdir -p "$RESULTS_DIR/optimal_poison/${dataset_name}/n${n}/lambda${poison_num}"
    
    echo "    Calculating optimal poison values for: $input_file_name (lambda=$poison_num)"
    
    # Brute Force Algorithm (C++ executable)
    if [ -f "$json_file" ]; then
        echo "        [Skipped] Brute force JSON file already exists: ${base_output_name}_brute_force.json"
    else
        ./calc_optimal_poison "$DATA_DIR/${input_file_name}" $poison_num "$json_file" 2>/dev/null || echo "      Error running brute force for $input_file_name with lambda=$poison_num"
    fi
}

# Function to process real datasets with given parameters
process_real_datasets() {
    local n_val="$1"
    local percentage="$2"
    local poison_num=$((n_val * percentage / 100))
    
    for real_dataset_name in "${real_dataset_names[@]}"; do
        for seed in "${seeds[@]}"; do
            for dtype in "uint64"; do
                input_file_name="${real_dataset_name}_n${n_val}_seed${seed}_${dtype}"
                if [ ! -f "$DATA_DIR/${input_file_name}" ]; then
                    continue
                fi
                base_output_name="${real_dataset_name}_n${n_val}_seed${seed}_lambda${poison_num}_percentage${percentage}_${dtype}"
                run_calc_optimal_poison "$input_file_name" "$poison_num" "$base_output_name" "$real_dataset_name" "$n_val"
            done
        done
    done
}

# Function to process synthetic datasets with given parameters
process_sync_datasets() {
    local n_val="$1"
    local R_val="$2"
    local percentage="$3"
    local poison_num=$((n_val * percentage / 100))
    
    for sync_dataset_name in "${sync_dataset_names[@]}"; do
        for seed in "${seeds[@]}"; do
            for dtype in "uint64"; do
                input_file_name="${sync_dataset_name}_n${n_val}_R${R_val}_seed${seed}_${dtype}"
                if [ ! -f "$DATA_DIR/${input_file_name}" ]; then
                    continue
                fi
                base_output_name="${sync_dataset_name}_n${n_val}_R${R_val}_seed${seed}_lambda${poison_num}_percentage${percentage}_${dtype}"
                run_calc_optimal_poison "$input_file_name" "$poison_num" "$base_output_name" "$sync_dataset_name" "$n_val"
            done
        done
    done
}

# Experiment 1: Vary n while keeping base_brute_force_R and base_brute_force_POISONING_PERCENTAGE
echo "  Experiment 1: Varying n (base_brute_force_R=$base_brute_force_R, base_brute_force_POISONING_PERCENTAGE=$base_brute_force_POISONING_PERCENTAGE)"
for n in "${brute_force_ns[@]}"; do
    echo "    Varying n: $n"
    process_real_datasets "$n" "$base_brute_force_POISONING_PERCENTAGE"
    process_sync_datasets "$n" "$base_brute_force_R" "$base_brute_force_POISONING_PERCENTAGE"
done

# Experiment 2: Vary R while keeping base_brute_force_n and base_brute_force_POISONING_PERCENTAGE
echo "  Experiment 2: Varying R (base_brute_force_n=$base_brute_force_n, base_brute_force_POISONING_PERCENTAGE=$base_brute_force_POISONING_PERCENTAGE)"
for R in "${brute_force_Rs[@]}"; do
    if [ "$R" -le "$base_brute_force_n" ]; then
        continue
    fi
    echo "    Varying R: $R"
    process_sync_datasets "$base_brute_force_n" "$R" "$base_brute_force_POISONING_PERCENTAGE"
done

# Experiment 3: Vary poisoning percentage while keeping base_brute_force_n and base_brute_force_R
echo "  Experiment 3: Varying poisoning percentage (base_brute_force_n=$base_brute_force_n, base_brute_force_R=$base_brute_force_R)"
for percentage in "${brute_force_POISONING_PERCENTAGES[@]}"; do
    echo "    Varying poisoning percentage: $percentage"
    process_real_datasets "$base_brute_force_n" "$percentage"
    process_sync_datasets "$base_brute_force_n" "$base_brute_force_R" "$percentage"
done

echo "[OK] Optimal poison calculation completed"
echo "  Results saved to: $RESULTS_DIR/optimal_poison/"
echo ""
