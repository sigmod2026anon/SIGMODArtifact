#!/bin/bash

set -euo pipefail

IMAGE_NAME="poisoning"
IMAGE_TAG="latest"

DATA_DIR="$(pwd)/data"
PROJECTS_DIR="$(pwd)/poisoning_projects"

MODE=""

# Analyze arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            MODE="--quick"
            shift
            ;;
        --all)
            MODE="--all"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 <--quick|--all>"
            echo "  --quick: Run quick experiment"
            echo "  --all: Run all experiments"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 <--quick|--all>"
            exit 1
            ;;
    esac
done

# Check required arguments
if [[ -z "$MODE" ]]; then
    echo "Error: --quick or --all is required"
    echo "Usage: $0 <--quick|--all>"
    exit 1
fi

mkdir -p $PROJECTS_DIR/poisoning/results

echo "Running mode: $MODE"

# Docker container resource limits and isolation
docker run -d \
    --cpus="1" \
    --cpuset-cpus="0" \
    --cap-add=SYS_ADMIN \
    -v $DATA_DIR:/workspace/data \
    -v $PROJECTS_DIR:/workspace/poisoning_projects \
    ${IMAGE_NAME}:${IMAGE_TAG} \
    bash -c "cd /workspace/poisoning_projects/poisoning/scripts && ./run_comprehensive_experiment.sh $MODE > /workspace/poisoning_projects/poisoning/results/run_all.log"
