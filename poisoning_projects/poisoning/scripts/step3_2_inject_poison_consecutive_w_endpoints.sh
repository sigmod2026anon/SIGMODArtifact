#!/usr/bin/env bash
set -euo pipefail

# [POISON] Step 3.2: Inject poison into datasets using consecutive approach with endpoints
# This script injects poison into all generated datasets using consecutive poisoning algorithm with endpoints

# Load common environment
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

# Configuration
# Use presets from env.sh
POISONING_PERCENTAGES=("${all_POISONING_PERCENTAGES_consecutive[@]}")
real_dataset_names=("${all_real_dataset_names[@]}")
sync_dataset_names=("${all_sync_dataset_names[@]}")
ns=("${all_ns_consecutive[@]}")
seeds=("${all_seeds[@]}")
Rs=("${all_Rs_consecutive[@]}")
brute_force_POISONING_PERCENTAGES=("${all_brute_force_POISONING_PERCENTAGES[@]}")
brute_force_ns=("${all_brute_force_ns[@]}")
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
            POISONING_PERCENTAGES=("${quick_POISONING_PERCENTAGES_consecutive[@]}")
            real_dataset_names=("${quick_real_dataset_names[@]}")
            sync_dataset_names=("${quick_sync_dataset_names[@]}")
            ns=("${quick_ns_consecutive[@]}")
            seeds=("${quick_seeds[@]}")
            Rs=("${quick_Rs_consecutive[@]}")
            brute_force_POISONING_PERCENTAGES=("${quick_brute_force_POISONING_PERCENTAGES[@]}")
            brute_force_ns=("${quick_brute_force_ns[@]}")
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

echo "[POISON] Step 3.2: Injecting poison into datasets using consecutive approach with endpoints..."
echo "  Data directory: $DATA_DIR"
echo "  Results directory: $RESULTS_DIR"
echo "  Base poisoning percentage: $base_POISONING_PERCENTAGE"
echo "  Base n: $base_n"
echo "  Base R: $base_R"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR/inject_poison_consecutive_w_endpoints"

# Change to build directory
cd "$BUILD_DIR"

# Function to inject poison into a dataset using consecutive approach with endpoints
run_inject_poison_consecutive_w_endpoints() {
    input_file_name="$1"
    output_file_name="$2"
    poison_num="$3"
    json_file_name="$4"
    dataset_name="$5"
    n="$6"
    output_file_path="$DATA_DIR/${output_file_name}"
    json_file_path="$RESULTS_DIR/inject_poison_consecutive_w_endpoints/${dataset_name}/n${n}/lambda${poison_num}/${json_file_name}"
    mkdir -p "$RESULTS_DIR/inject_poison_consecutive_w_endpoints/${dataset_name}/n${n}/lambda${poison_num}"
    output_file_exists=false
    json_file_exists=false
    if [ -f "$output_file_path" ]; then
        output_file_exists=true
    fi
    if [ -f "$json_file_path" ]; then
        json_file_exists=true
    fi
    if [ "$output_file_exists" = true ] && [ "$json_file_exists" = true ]; then
        echo "    Output file and JSON file already exist: $output_file_path and $json_file_path"
        return
    fi
    rm -f "$output_file_path"
    rm -f "$json_file_path"

    # command="./inject_poison_consecutive_w_endpoints \"$DATA_DIR/${input_file_name}\" \"$output_file_path\" $poison_num \"$json_file_path\""
    # echo "Command: $command"

    ./inject_poison_consecutive_w_endpoints "$DATA_DIR/${input_file_name}" "$output_file_path" $poison_num "$json_file_path" 2>/dev/null || echo "      Error processing $dataset_name with lambda=$poison_num"
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
                    echo "    Input file not found: $DATA_DIR/${input_file_name}"
                    continue
                fi
                output_file_name="${real_dataset_name}_n${n_val}_seed${seed}_lambda${poison_num}_consecutive_w_endpoints_${dtype}"
                json_file_name="${real_dataset_name}_n${n_val}_seed${seed}_lambda${poison_num}_consecutive_w_endpoints_${dtype}.json"
                run_inject_poison_consecutive_w_endpoints "$input_file_name" "$output_file_name" "$poison_num" "$json_file_name" "$real_dataset_name" "$n_val"
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
                    echo "    Input file not found: $DATA_DIR/${input_file_name}"
                    continue
                fi
                output_file_name="${sync_dataset_name}_n${n_val}_R${R_val}_seed${seed}_lambda${poison_num}_consecutive_w_endpoints_${dtype}"
                json_file_name="${sync_dataset_name}_n${n_val}_R${R_val}_seed${seed}_lambda${poison_num}_consecutive_w_endpoints_${dtype}.json"
                run_inject_poison_consecutive_w_endpoints "$input_file_name" "$output_file_name" "$poison_num" "$json_file_name" "$sync_dataset_name" "$n_val"
            done
        done
    done
}

# Experiment 1: Vary n while keeping base_R and base_POISONING_PERCENTAGE
echo "  Experiment 1: Varying n (base_R=$base_R, base_POISONING_PERCENTAGE=$base_POISONING_PERCENTAGE)"
for n in "${ns[@]}"; do
    echo "    Varying n: $n"
    process_real_datasets "$n" "$base_POISONING_PERCENTAGE"
    process_sync_datasets "$n" "$base_R" "$base_POISONING_PERCENTAGE"
done

for n in "${brute_force_ns[@]}"; do
    echo "    Varying n (for brute force): $n"
    process_real_datasets "$n" "$base_brute_force_POISONING_PERCENTAGE"
    process_sync_datasets "$n" "$base_brute_force_R" "$base_brute_force_POISONING_PERCENTAGE"
done

# Experiment 2: Vary R while keeping base_n and base_POISONING_PERCENTAGE
echo "  Experiment 2: Varying R (base_n=$base_n, base_POISONING_PERCENTAGE=$base_POISONING_PERCENTAGE)"
for R in "${Rs[@]}"; do
    if [ "$R" -le "$base_n" ]; then
        continue
    fi
    echo "    Varying R: $R"
    process_sync_datasets "$base_n" "$R" "$base_POISONING_PERCENTAGE"
done

for R in "${brute_force_Rs[@]}"; do
    if [ "$R" -le "$base_brute_force_n" ]; then
        continue
    fi
    echo "    Varying R (for brute force): $R"
    process_sync_datasets "$base_brute_force_n" "$R" "$base_brute_force_POISONING_PERCENTAGE"
done

# Experiment 3: Vary poisoning percentage while keeping base_n and base_R
echo "  Experiment 3: Varying poisoning percentage (base_n=$base_n, base_R=$base_R)"
for percentage in "${POISONING_PERCENTAGES[@]}"; do
    echo "    Varying poisoning percentage: $percentage"
    process_real_datasets "$base_n" "$percentage"
    process_sync_datasets "$base_n" "$base_R" "$percentage"
done

for percentage in "${brute_force_POISONING_PERCENTAGES[@]}"; do
    echo "    Varying poisoning percentage (for brute force): $percentage"
    process_real_datasets "$base_brute_force_n" "$percentage"
    process_sync_datasets "$base_brute_force_n" "$base_brute_force_R" "$percentage"
done

echo "[OK] Consecutive with endpoints poison injection completed"
echo "  Results saved to: $RESULTS_DIR/inject_poison_consecutive_w_endpoints/"

# Count JSON files
total_json_files=$(find "$RESULTS_DIR/inject_poison_consecutive_w_endpoints/" -name "*.json" -type f | wc -l)
echo "    [POISON] Total JSON result files created: $total_json_files"
echo "" 