FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install dependencies
RUN apt-get update && \
    apt-get -y install gcc mono-mcs python3.10 python3-pip python3.10-dev ffmpeg libsm6 libxext6 git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.10 -m pip install --upgrade pip

# Set work directory
WORKDIR /app-src

# Copy source code
COPY . /app-src/

# Install Python dependencies
RUN python3.10 -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
RUN python3.10 -m pip install xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu118

# Clone and install external repositories
RUN git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization && \
    python3.10 -m pip install ./diff-gaussian-rasterization
RUN python3.10 -m pip install git+https://github.com/NVlabs/nvdiffrast

# Install additional requirements from requirements.txt
RUN python3.10 -m pip install -r requirements.txt
