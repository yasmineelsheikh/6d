FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV COSMOS_DIR=/workspace/cosmos-transfer2.5

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set up Python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install uv for faster dependency installation
RUN pip install uv

# Create workspace
WORKDIR /workspace

# Clone and install Cosmos-Transfer2.5
RUN git clone https://github.com/NVIDIA/Cosmos-Transfer2.5.git cosmos-transfer2.5
WORKDIR /workspace/cosmos-transfer2.5

# Install Cosmos dependencies
# Use the appropriate CUDA version for your setup
RUN uv sync --extra cu121-torch27 || \
    pip install -e . --extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/v1.2.0/cu121_torch27/simple

# Install RunPod SDK and cloud storage dependencies
RUN pip install runpod boto3

# Copy handler
COPY handler.py /workspace/handler.py

# Set working directory
WORKDIR /workspace

# Run the handler
CMD ["python", "handler.py"]
