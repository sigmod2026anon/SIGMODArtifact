#!/bin/bash

set -euo pipefail

IMAGE_NAME="poisoning"
IMAGE_TAG="latest"

DATA_DIR="$(pwd)/data"
PROJECTS_DIR="$(pwd)/poisoning_projects"

docker run -it --rm \
    --cpus="1" \
    -v $DATA_DIR:/workspace/data \
    -v $PROJECTS_DIR:/workspace/poisoning_projects \
    ${IMAGE_NAME}:${IMAGE_TAG} /bin/bash
