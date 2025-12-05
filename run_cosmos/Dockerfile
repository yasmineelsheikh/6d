# Use a base image with CUDA 12.8 and Python 3.10 (adjust version as needed)
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# -----------------------------
# System dependencies
# -----------------------------
RUN apt update && apt install -y \
    python3 python3-pip python3-dev \
    build-essential cmake \
    curl ffmpeg tree wget git git-lfs \
    libgl1 libglib2.0-0 \
    && apt clean

# Enable Git LFS
RUN git lfs install

# -----------------------------
# Clone Cosmos repo
# -----------------------------
RUN git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git /workspace/cosmos
WORKDIR /workspace/cosmos
RUN git lfs pull

# -----------------------------
# Install PyTorch + CUDA 12.1
# -----------------------------
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# -----------------------------
# Install Cosmos dependencies
# -----------------------------
RUN pip install -r requirements.txt
RUN pip install -e .

# -----------------------------
# HuggingFace CLI (optional)
# -----------------------------
RUN pip install "huggingface_hub[cli]"

# -----------------------------
# Copy handler for RunPod
# -----------------------------
WORKDIR /workspace
COPY src/handler.py ./handler.py

CMD ["python3", "-u", "handler.py"]
