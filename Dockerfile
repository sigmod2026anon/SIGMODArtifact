FROM ubuntu:22.04

# Install only essential packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    wget \
    git \
    nlohmann-json3-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install Python packages
COPY requirements.txt .
RUN pip3 install -r requirements.txt
