# RunPod GPU Setup for Cosmos Augmentation

This guide covers setting up ARES with Cosmos-Transfer2.5 on RunPod GPU instances using serverless endpoints.

## Why RunPod?

RunPod offers several advantages over traditional cloud providers:

- **Cost-effective**: Pay-per-second serverless pricing (vs AWS on-demand)
- **Easy setup**: No SSH configuration needed, just API key
- **Auto-scaling**: Serverless endpoints scale automatically from 0 to N workers
- **GPU variety**: Access to RTX 4090, A100, H100 at competitive prices
- **Pre-built templates**: Quick deployment with Docker

## Quick Start

### 1. Create RunPod Account

1. Sign up at [runpod.io](https://www.runpod.io/)
2. Add credits to your account (minimum $10 recommended)
3. Generate an API key:
   - Go to Settings → API Keys
   - Create new API key
   - Copy the key (you'll need it later)

### 2. Set Up Cloud Storage

RunPod endpoints need cloud storage for input/output files. Choose one:

#### Option A: AWS S3 (Recommended for AWS users)

1. Create an S3 bucket (e.g., `cosmos-temp-storage`)
2. Create IAM user with S3 access
3. Get access key and secret key

#### Option B: Cloudflare R2 (Recommended - cheaper)

1. Sign up for Cloudflare (free tier available)
2. Go to R2 Object Storage
3. Create a bucket (e.g., `cosmos-temp-storage`)
4. Create API token with R2 read/write permissions
5. Note your Account ID, Access Key, and Secret Key

### 3. Configure Environment Variables

Copy the example configuration:

```bash
cp .env.runpod.example .env
```

Edit `.env` and add your credentials:

```bash
# RunPod Configuration
RUNPOD_API_KEY=your_runpod_api_key_here
COSMOS_EXECUTION_MODE=auto  # Will use RunPod if API key is set

# Cloud Storage (choose S3 or R2)
# For S3:
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
S3_BUCKET_NAME=cosmos-temp-storage

# For R2 (recommended):
# R2_ACCOUNT_ID=your_r2_account_id
# R2_ACCESS_KEY_ID=your_r2_key
# R2_SECRET_ACCESS_KEY=your_r2_secret
# R2_BUCKET_NAME=cosmos-temp-storage
```

### 4. Deploy RunPod Endpoint

The endpoint is a Docker container with Cosmos-Transfer2.5 pre-installed.

#### Prerequisites

- Docker installed and running
- Docker Hub account (free)
- Logged in to Docker Hub: `docker login`

#### Deploy

```bash
# Update the image name in the script
# Edit runpod_setup/deploy.sh and set DOCKER_USERNAME

cd runpod_setup
chmod +x deploy.sh

# Build and push Docker image
./deploy.sh

# Or use the Python script for more control
python ../scripts/deploy_runpod_endpoint.py --image your_dockerhub_username/cosmos-transfer-runpod
```

#### Manual Endpoint Creation

After pushing the image, create the endpoint in RunPod dashboard:

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Configure:
   - **Name**: `cosmos-transfer`
   - **Docker Image**: `your_dockerhub_username/cosmos-transfer-runpod:latest`
   - **GPU Type**: AMPERE_16 or higher (A4000, A5000, A100, H100)
   - **Min Workers**: 0 (serverless - scales to zero)
   - **Max Workers**: 3 (adjust based on your needs)
   - **Idle Timeout**: 5 seconds
   - **Container Disk**: 20 GB
   - **Environment Variables**:
     - `COSMOS_DIR=/workspace/cosmos-transfer2.5`
     - Add your cloud storage credentials (S3 or R2)

4. Click "Deploy"
5. Copy the Endpoint ID
6. Add to `.env`:
   ```bash
   RUNPOD_ENDPOINT_ID=your_endpoint_id_here
   ```

### 5. Test the Setup

```python
from ares.augmentation import CosmosAugmentor
from pathlib import Path

# Initialize with RunPod execution
augmentor = CosmosAugmentor(execution_mode="runpod")

# Test with a sample video
video_path = Path("path/to/your/video.mp4")
output_dir = Path("output")

# This will run on RunPod GPU
augmented = augmentor.augment_episode(video_path, output_dir)
```

## Usage

### Execution Modes

The `CosmosAugmentor` supports three execution modes:

1. **`auto`** (default): Automatically uses RunPod if `RUNPOD_API_KEY` is set, otherwise local
2. **`runpod`**: Always use RunPod (fails if not configured)
3. **`local`**: Always use local GPU

```python
# Auto mode (recommended)
augmentor = CosmosAugmentor(execution_mode="auto")

# Force RunPod
augmentor = CosmosAugmentor(execution_mode="runpod")

# Force local
augmentor = CosmosAugmentor(execution_mode="local")
```

### Environment Variable Control

You can also control execution mode via environment variable:

```bash
# In .env or shell
export COSMOS_EXECUTION_MODE=runpod  # or 'local' or 'auto'
```

## Cost Estimation

RunPod serverless pricing (as of 2024):

| GPU Type | Price per minute | Estimated cost per video* |
|----------|------------------|---------------------------|
| RTX 4090 | $0.69/hr ($0.0115/min) | ~$0.12 |
| A100 40GB | $1.89/hr ($0.0315/min) | ~$0.32 |
| A100 80GB | $2.49/hr ($0.0415/min) | ~$0.42 |
| H100 | $4.89/hr ($0.0815/min) | ~$0.82 |

*Assuming ~10 minutes per video generation

**Comparison with AWS:**
- AWS p3.2xlarge (V100): ~$3.06/hr = ~$0.51/video
- AWS g5.2xlarge (A10G): ~$1.21/hr = ~$0.20/video
- AWS p4d.24xlarge (8x A100): ~$32.77/hr = ~$5.46/video

**RunPod is 60-90% cheaper than AWS for sporadic workloads!**

## Monitoring

### RunPod Dashboard

Monitor your jobs in real-time:
1. Go to [RunPod Console](https://www.runpod.io/console/serverless)
2. Click on your endpoint
3. View:
   - Active jobs
   - Job history
   - Logs
   - Costs

### Logs

The `CosmosAugmentor` logs all operations:

```python
import logging
logging.basicConfig(level=logging.INFO)

# You'll see logs like:
# INFO: Cosmos execution mode: runpod
# INFO: RunPod executor initialized successfully
# INFO: Using RunPod GPU for Cosmos inference...
# INFO: Uploading input files to cloud storage...
# INFO: Job submitted: abc123. Waiting for completion...
# INFO: Job completed successfully!
```

## Troubleshooting

### "RunPod API key is required"

**Solution**: Set `RUNPOD_API_KEY` in your `.env` file

### "S3_BUCKET_NAME environment variable is required"

**Solution**: Configure cloud storage (S3 or R2) in `.env`

### "RunPod job failed"

**Possible causes**:
1. Docker image not built correctly
2. Cosmos dependencies missing
3. Insufficient GPU memory

**Solution**: Check RunPod dashboard logs for detailed error messages

### "Job timed out after 600 seconds"

**Solution**: 
- Increase timeout in `runpod_executor.py`
- Use a more powerful GPU (A100 vs RTX 4090)
- Reduce video resolution

### Endpoint not responding

**Solution**:
1. Check endpoint status in RunPod dashboard
2. Verify Docker image is accessible
3. Ensure endpoint has workers available
4. Check cloud storage credentials

### High costs

**Optimization tips**:
1. Use RTX 4090 instead of A100 for most workloads (3x cheaper)
2. Set `max_workers` appropriately (don't over-provision)
3. Use shorter idle timeout (5 seconds)
4. Monitor usage in RunPod dashboard
5. Consider batch processing to amortize startup costs

## Advanced Configuration

### Custom GPU Selection

Edit the endpoint configuration to use specific GPUs:

```python
# In deploy_runpod_endpoint.py
endpoint_config = {
    "gpu_ids": "AMPERE_16",  # RTX A4000/A5000
    # or "AMPERE_48",  # A100 40GB
    # or "AMPERE_80",  # A100 80GB
    # or "ADA_24",     # RTX 4090
}
```

### Multi-GPU Inference

For faster processing, enable multi-GPU in the Dockerfile:

```dockerfile
# Add to Dockerfile
ENV CUDA_VISIBLE_DEVICES=0,1,2,3
```

### Custom Inference Parameters

Modify parameters in `cosmos2.py`:

```python
parameters = {
    "guidance": 3,  # Increase for stronger prompt adherence
    "edge": {"control_weight": 0.4},  # Adjust edge control
    "vis": {"control_weight": 0.2},   # Adjust blur control
}
```

## Security Best Practices

1. **Never commit `.env` to git** (already in `.gitignore`)
2. **Use IAM roles** for cloud storage (instead of access keys)
3. **Rotate API keys** regularly
4. **Use private Docker registries** for production
5. **Enable CloudFlare Access** for R2 buckets
6. **Monitor costs** to detect anomalies

## Next Steps

1. ✅ Set up RunPod account and API key
2. ✅ Configure cloud storage (S3 or R2)
3. ✅ Deploy the Cosmos endpoint
4. ✅ Test with a sample video
5. 🔄 Optimize GPU selection for your workload
6. 🔄 Monitor costs and performance
7. 🔄 Scale to production workloads
