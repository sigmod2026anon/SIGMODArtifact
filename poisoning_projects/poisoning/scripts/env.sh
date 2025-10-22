#!/usr/bin/env bash

# Directory of this scripts folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Project root (two levels up from scripts directory)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Standard directories used across scripts
BUILD_DIR="$PROJECT_ROOT/poisoning/cpp/build"
DATA_DIR="$PROJECT_ROOT/data"
RESULTS_DIR="$PROJECT_ROOT/poisoning/results"
PLOT_DIR="$PROJECT_ROOT/poisoning/plot"

export SCRIPT_DIR PROJECT_ROOT BUILD_DIR DATA_DIR RESULTS_DIR PLOT_DIR

# Poisoning percentages
base_POISONING_PERCENTAGE=10
all_POISONING_PERCENTAGES=(2 4 6 8 10 12 14 16 18 20)
all_POISONING_PERCENTAGES_consecutive=(2 4 6 8 10)
quick_POISONING_PERCENTAGES=(2 6 10)
quick_POISONING_PERCENTAGES_consecutive=(2 6)

# Real dataset names
all_real_dataset_names=(
    "books_200M"
    "fb_200M"
    "osm_cellids_200M"
)
quick_real_dataset_names=(
    "books_200M"
    "fb_200M"
    "osm_cellids_200M"
)

# Synthetic dataset names
all_sync_dataset_names=(
    "uniform"
    "normal"
    "exponential"
)
quick_sync_dataset_names=(
    "uniform"
    "normal"
    "exponential"
)

# n values
base_n=1000
all_ns=(
    100
    200
    500
    1000
    2000
    5000
    10000
)
all_ns_consecutive=(
    100
    200
    500
    1000
)
quick_ns=(
    100
    1000
)
quick_ns_consecutive=(
    100
)

# seeds
all_seeds=($(seq 0 99))
quick_seeds=(0)

# R values
base_R=100000
all_Rs=(
    2000
    3000
    4000
    5000
    7000
    10000
    20000
    50000
    100000
    200000
    500000
    1000000
)
all_Rs_consecutive=(
    100000
)
quick_Rs=(
    1000
    10000
    100000
)
quick_Rs_consecutive=(
    100000
)

# Brute force
base_brute_force_POISONING_PERCENTAGE=10
all_brute_force_POISONING_PERCENTAGES=(2 4 6 8 10)
quick_brute_force_POISONING_PERCENTAGES=(2 4 6)
base_brute_force_n=50
all_brute_force_ns=(50)
quick_brute_force_ns=(50)
base_brute_force_R=1000
all_brute_force_Rs=(100 150 200 300 500 1000 2000 5000 10000)
quick_brute_force_Rs=(100 1000 10000)

export base_POISONING_PERCENTAGE base_n base_R
export all_POISONING_PERCENTAGES quick_POISONING_PERCENTAGES
export all_real_dataset_names quick_real_dataset_names
export all_sync_dataset_names quick_sync_dataset_names
export all_ns quick_ns
export all_seeds quick_seeds
export all_Rs quick_Rs
export base_brute_force_POISONING_PERCENTAGE all_brute_force_POISONING_PERCENTAGES quick_brute_force_POISONING_PERCENTAGES
export base_brute_force_n all_brute_force_ns quick_brute_force_ns
export base_brute_force_R all_brute_force_Rs quick_brute_force_Rs
export all_POISONING_PERCENTAGES_consecutive all_ns_consecutive all_Rs_consecutive
export quick_POISONING_PERCENTAGES_consecutive quick_ns_consecutive quick_Rs_consecutive
