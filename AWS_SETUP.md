# AWS GPU Setup for Cosmos Augmentation

This guide covers setting up ARES with Cosmos-Transfer2.5 on AWS GPU instances.

## Recommended AWS Instance Types

For Cosmos-Transfer2.5-2B, you'll need instances with sufficient GPU memory:

| Instance Type | GPU | VRAM | Notes |
|--------------|-----|------|-------|
| **p4d.24xlarge** | 8x A100 (40GB) | 320GB total | Best for multi-GPU inference |
| **p3.2xlarge** | 1x V100 (16GB) | 16GB | Minimum, may need optimization |
| **g5.2xlarge** | 1x A10G (24GB) | 24GB | Good balance |
| **p4d.24xlarge** | 8x A100 (40GB) | 320GB total | Recommended for production |

**Note**: Cosmos-Transfer2.5-2B recommends **65GB+ VRAM per GPU** for optimal performance. The p4d.24xlarge with 8x A100 (40GB each) can work with multi-GPU setup.

## Setup Steps

### 1. Launch EC2 Instance

1. Launch an EC2 instance with one of the recommended GPU instance types
2. Use an **Ubuntu 22.04 LTS** or **Amazon Linux 2023** AMI with GPU support
3. Configure security groups to allow:
   - SSH (port 22)
   - Streamlit (port 8501) if accessing web UI
   - MongoDB (port 27017) if using local MongoDB

### 2. Install NVIDIA Drivers and CUDA

```bash
# Update system
sudo apt-get update

# Install NVIDIA drivers (for Ubuntu)
sudo apt-get install -y nvidia-driver-535
sudo reboot

# After reboot, verify GPU
nvidia-smi

# Install CUDA toolkit (if not included in AMI)
# Follow NVIDIA's installation guide for your CUDA version
```

### 3. Install Python Dependencies

```bash
# Clone or upload ARES repository
cd /path/to/ares-platform

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install ARES dependencies
pip install -r requirements.txt

# Install Cosmos-Transfer2.5 dependencies
cd cosmos-transfer2.5
# Follow Cosmos setup instructions
# This will install decord, torch with CUDA, etc.
```

### 4. Install Cosmos-Transfer2.5

```bash
cd cosmos-transfer2.5

# Install using uv (recommended by Cosmos)
uv sync --extra cu128-torch27  # or appropriate CUDA version

# Or install using pip
pip install -e . --extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/v1.2.0/cu128_torch27/simple
```

### 5. Download Cosmos Model Checkpoints

```bash
# Cosmos will download checkpoints automatically on first use
# Or download manually from Hugging Face:
# https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B
```

### 6. Configure Environment Variables

Create `.env` file in `ares-platform/`:

```bash
# API Keys for VLM/LLM
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Cosmos will use direct inference (no API needed)
# But you can still use API if preferred:
# COSMOS_API_URL=https://your-cosmos-api.com
# COSMOS_API_KEY=your_api_key
```

### 7. Verify Setup

```bash
# Test GPU availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Test Cosmos import
python3 -c "from cosmos_transfer2.inference import Control2WorldInference; print('Cosmos import successful')"
```

## Running ARES on AWS

### Option 1: Direct SSH Access

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Start Streamlit (with port forwarding)
streamlit run src/ares/app/webapp.py --server.address 0.0.0.0
```

Then access via SSH tunnel:
```bash
# On your local machine
ssh -i your-key.pem -L 8501:localhost:8501 ubuntu@your-instance-ip
```

### Option 2: Using Screen/Tmux

```bash
# Start a screen session
screen -S ares

# Run Streamlit
streamlit run src/ares/app/webapp.py --server.address 0.0.0.0

# Detach: Ctrl+A, then D
# Reattach: screen -r ares
```

## Performance Optimization

### Multi-GPU Setup

If using multi-GPU instances (e.g., p4d.24xlarge), you can enable multi-GPU inference:

```python
# The code will automatically detect multiple GPUs
# For manual control, modify context_parallel_size in cosmos.py
```

### Memory Optimization

For instances with less VRAM:

1. **Disable guardrails** (already done in code)
2. **Reduce batch size** if processing multiple videos
3. **Use gradient checkpointing** (if available in Cosmos config)

### Cost Optimization

- Use **Spot Instances** for development/testing (up to 90% savings)
- **Stop instances** when not in use
- Use **AWS Systems Manager** for automated start/stop
- Consider **AWS Batch** for batch processing jobs

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA
nvcc --version

# Check PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Decord Installation Issues

On AWS Linux, decord should install from PyPI. If issues occur:

```bash
# Try installing from source
pip install decord --no-binary decord
```

### Out of Memory Errors

- Reduce video resolution
- Process videos one at a time
- Use a larger instance type
- Enable CPU offloading for guardrails

### Cosmos Import Errors

```bash
# Ensure you're in the cosmos-transfer2.5 directory
cd cosmos-transfer2.5

# Reinstall dependencies
uv sync --extra cu128-torch27
```

## Monitoring

### GPU Utilization

```bash
# Watch GPU usage
watch -n 1 nvidia-smi
```

### System Resources

```bash
# Monitor CPU, memory, disk
htop
df -h
```

## Security Considerations

1. **Use IAM roles** instead of hardcoding AWS credentials
2. **Restrict security groups** to only necessary ports
3. **Use VPC** for network isolation
4. **Enable CloudTrail** for audit logging
5. **Encrypt EBS volumes** for data at rest

## Cost Estimation

Example monthly costs (US East, on-demand):

- **p3.2xlarge**: ~$3.06/hour = ~$2,200/month (if running 24/7)
- **g5.2xlarge**: ~$1.21/hour = ~$870/month
- **p4d.24xlarge**: ~$32.77/hour = ~$23,600/month

**Recommendation**: Use Spot Instances or schedule instances to run only when needed.

