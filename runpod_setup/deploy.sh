#!/bin/bash

# RunPod Endpoint Deployment Script
# This script builds and deploys the Cosmos serverless endpoint to RunPod

set -e

echo "=== RunPod Cosmos Endpoint Deployment ==="

# Configuration
IMAGE_NAME="cosmos-transfer-runpod"
DOCKER_USERNAME="${DOCKER_USERNAME:-your_dockerhub_username}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if RUNPOD_API_KEY is set
if [ -z "$RUNPOD_API_KEY" ]; then
    echo "Error: RUNPOD_API_KEY environment variable is not set."
    echo "Please set it with: export RUNPOD_API_KEY=your_api_key"
    exit 1
fi

echo "Step 1: Building Docker image..."
docker build -t $FULL_IMAGE_NAME .

echo "Step 2: Pushing image to Docker Hub..."
echo "Please make sure you're logged in to Docker Hub (docker login)"
docker push $FULL_IMAGE_NAME

echo "Step 3: Creating RunPod serverless endpoint..."
echo "Image: $FULL_IMAGE_NAME"

# Use the Python deployment script for endpoint creation
python3 ../scripts/deploy_runpod_endpoint.py --image $FULL_IMAGE_NAME

echo "=== Deployment Complete ==="
echo ""
echo "Next steps:"
echo "1. Set RUNPOD_ENDPOINT_ID in your .env file"
echo "2. Configure cloud storage (S3 or R2) credentials"
echo "3. Test the endpoint with: python scripts/test_runpod.py"
