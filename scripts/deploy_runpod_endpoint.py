"""
Automated deployment script for RunPod Cosmos endpoint.

This script builds the Docker image, pushes it to a registry,
and creates/updates the RunPod serverless endpoint.
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_requirements():
    """Check if required tools and credentials are available."""
    # Check Docker
    try:
        subprocess.run(['docker', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Docker is not installed or not running")
        return False
    
    # Check RunPod API key
    if not os.environ.get('RUNPOD_API_KEY'):
        logger.error("RUNPOD_API_KEY environment variable is not set")
        return False
    
    return True


def build_image(image_name: str, tag: str = "latest"):
    """Build the Docker image."""
    full_name = f"{image_name}:{tag}"
    logger.info(f"Building Docker image: {full_name}")
    
    # Change to runpod_setup directory
    setup_dir = Path(__file__).parent.parent / "runpod_setup"
    
    try:
        subprocess.run(
            ['docker', 'build', '-t', full_name, '.'],
            cwd=setup_dir,
            check=True
        )
        logger.info(f"Successfully built image: {full_name}")
        return full_name
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to build Docker image: {e}")
        return None


def push_image(image_name: str):
    """Push the Docker image to registry."""
    logger.info(f"Pushing image to registry: {image_name}")
    
    try:
        subprocess.run(['docker', 'push', image_name], check=True)
        logger.info(f"Successfully pushed image: {image_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to push image: {e}")
        logger.info("Make sure you're logged in to Docker Hub: docker login")
        return False


def create_endpoint(image_name: str, endpoint_name: str = "cosmos-transfer"):
    """Create or update RunPod serverless endpoint."""
    logger.info(f"Creating RunPod endpoint: {endpoint_name}")
    
    try:
        import runpod
        runpod.api_key = os.environ.get('RUNPOD_API_KEY')
        
        # Check if endpoint already exists
        endpoints = runpod.get_endpoints()
        existing = None
        for ep in endpoints:
            if ep.get('name') == endpoint_name:
                existing = ep
                break
        
        if existing:
            endpoint_id = existing['id']
            logger.info(f"Endpoint already exists: {endpoint_id}")
            logger.info(f"Updating endpoint with new image: {image_name}")
            
            # Update endpoint
            # Note: RunPod API for updating may vary, check documentation
            logger.warning("Endpoint update via API not implemented. Please update manually in RunPod dashboard.")
            
        else:
            logger.info("Creating new endpoint...")
            
            # Create endpoint configuration
            endpoint_config = {
                "name": endpoint_name,
                "docker_image": image_name,
                "gpu_ids": "AMPERE_16",  # RTX A4000/A5000, adjust as needed
                "min_workers": 0,  # Serverless: scale to zero
                "max_workers": 3,
                "idle_timeout": 5,  # seconds
                "env_vars": {
                    # Add any required environment variables
                    "COSMOS_DIR": "/workspace/cosmos-transfer2.5",
                }
            }
            
            # Note: Actual endpoint creation requires using RunPod SDK or API
            logger.warning("Automatic endpoint creation not fully implemented.")
            logger.info("Please create the endpoint manually in RunPod dashboard with these settings:")
            logger.info(f"  Image: {image_name}")
            logger.info(f"  GPU: AMPERE_16 or higher (A4000, A5000, A100, H100)")
            logger.info(f"  Min Workers: 0 (serverless)")
            logger.info(f"  Max Workers: 3")
            logger.info(f"  Idle Timeout: 5 seconds")
            
            endpoint_id = "MANUAL_CREATION_REQUIRED"
        
        logger.info(f"\nEndpoint ID: {endpoint_id}")
        logger.info(f"Add this to your .env file:")
        logger.info(f"RUNPOD_ENDPOINT_ID={endpoint_id}")
        
        return endpoint_id
        
    except ImportError:
        logger.error("runpod package not installed. Run: pip install runpod")
        return None
    except Exception as e:
        logger.error(f"Failed to create endpoint: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Deploy Cosmos endpoint to RunPod')
    parser.add_argument('--image', default='your_dockerhub_username/cosmos-transfer-runpod',
                       help='Docker image name (default: your_dockerhub_username/cosmos-transfer-runpod)')
    parser.add_argument('--tag', default='latest',
                       help='Docker image tag (default: latest)')
    parser.add_argument('--skip-build', action='store_true',
                       help='Skip building the Docker image')
    parser.add_argument('--skip-push', action='store_true',
                       help='Skip pushing the Docker image')
    parser.add_argument('--endpoint-name', default='cosmos-transfer',
                       help='RunPod endpoint name (default: cosmos-transfer)')
    
    args = parser.parse_args()
    
    logger.info("=== RunPod Cosmos Endpoint Deployment ===\n")
    
    # Check requirements
    if not check_requirements():
        logger.error("Requirements check failed. Please fix the issues and try again.")
        sys.exit(1)
    
    full_image_name = f"{args.image}:{args.tag}"
    
    # Build image
    if not args.skip_build:
        built_image = build_image(args.image, args.tag)
        if not built_image:
            logger.error("Build failed. Exiting.")
            sys.exit(1)
    else:
        logger.info(f"Skipping build, using existing image: {full_image_name}")
    
    # Push image
    if not args.skip_push:
        if not push_image(full_image_name):
            logger.error("Push failed. Exiting.")
            sys.exit(1)
    else:
        logger.info(f"Skipping push")
    
    # Create endpoint
    endpoint_id = create_endpoint(full_image_name, args.endpoint_name)
    
    if endpoint_id:
        logger.info("\n=== Deployment Complete ===")
        logger.info("\nNext steps:")
        logger.info("1. Add RUNPOD_ENDPOINT_ID to your .env file")
        logger.info("2. Configure cloud storage (S3 or R2) credentials in .env")
        logger.info("3. Test the endpoint with: python scripts/test_runpod.py")
    else:
        logger.error("\n=== Deployment Failed ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
