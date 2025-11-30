"""
RunPod GPU Executor for Cosmos Inference

This module provides utilities for executing Cosmos-Transfer2.5 inference
on RunPod GPU instances using their serverless API.
"""

import os
import time
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class RunPodExecutor:
    """
    Executor for running Cosmos inference on RunPod GPU instances.
    
    Uses RunPod's serverless endpoints for pay-per-second GPU compute.
    Handles file uploads to cloud storage (S3/R2) and job management.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        storage_backend: str = "s3",
    ):
        """
        Initialize RunPod executor.
        
        Args:
            api_key: RunPod API key (defaults to RUNPOD_API_KEY env var)
            endpoint_id: RunPod endpoint ID (defaults to RUNPOD_ENDPOINT_ID env var)
            storage_backend: Cloud storage backend ('s3' or 'r2')
        """
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        self.endpoint_id = endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID")
        self.storage_backend = storage_backend
        
        if not self.api_key:
            raise ValueError("RunPod API key is required. Set RUNPOD_API_KEY environment variable.")
        
        # Initialize RunPod SDK
        try:
            import runpod
            self.runpod = runpod
            runpod.api_key = self.api_key
        except ImportError:
            raise ImportError("runpod package not installed. Run: pip install runpod")
        
        # Initialize cloud storage client
        self._init_storage()
        
        logger.info(f"RunPod executor initialized with endpoint: {self.endpoint_id or 'auto'}")
    
    def _init_storage(self):
        """Initialize cloud storage client (S3 or R2)."""
        if self.storage_backend == "s3":
            self.bucket_name = os.environ.get("S3_BUCKET_NAME")
            if not self.bucket_name:
                raise ValueError("S3_BUCKET_NAME environment variable is required for S3 storage")
            
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            )
            logger.info(f"Using S3 storage: {self.bucket_name}")
            
        elif self.storage_backend == "r2":
            self.bucket_name = os.environ.get("R2_BUCKET_NAME")
            account_id = os.environ.get("R2_ACCOUNT_ID")
            
            if not self.bucket_name or not account_id:
                raise ValueError("R2_BUCKET_NAME and R2_ACCOUNT_ID environment variables are required for R2 storage")
            
            self.s3_client = boto3.client(
                's3',
                endpoint_url=f'https://{account_id}.r2.cloudflarestorage.com',
                aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"),
            )
            logger.info(f"Using Cloudflare R2 storage: {self.bucket_name}")
        else:
            raise ValueError(f"Unsupported storage backend: {self.storage_backend}")
    
    def upload_file(self, local_path: str, remote_key: str) -> str:
        """
        Upload a file to cloud storage.
        
        Args:
            local_path: Local file path
            remote_key: Remote object key (path in bucket)
            
        Returns:
            Public URL of the uploaded file
        """
        try:
            logger.info(f"Uploading {local_path} to {self.storage_backend}://{self.bucket_name}/{remote_key}")
            
            self.s3_client.upload_file(
                local_path,
                self.bucket_name,
                remote_key,
                ExtraArgs={'ACL': 'public-read'} if self.storage_backend == 's3' else {}
            )
            
            # Generate public URL
            if self.storage_backend == "s3":
                url = f"https://{self.bucket_name}.s3.amazonaws.com/{remote_key}"
            else:  # R2
                # R2 requires custom domain or presigned URL
                url = self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': self.bucket_name, 'Key': remote_key},
                    ExpiresIn=3600  # 1 hour
                )
            
            logger.info(f"Upload successful: {url}")
            return url
            
        except ClientError as e:
            logger.error(f"Failed to upload file: {e}")
            raise
    
    def download_file(self, remote_key: str, local_path: str) -> None:
        """
        Download a file from cloud storage.
        
        Args:
            remote_key: Remote object key (path in bucket)
            local_path: Local destination path
        """
        try:
            logger.info(f"Downloading {self.storage_backend}://{self.bucket_name}/{remote_key} to {local_path}")
            
            # Ensure parent directory exists
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.s3_client.download_file(
                self.bucket_name,
                remote_key,
                local_path
            )
            
            logger.info(f"Download successful: {local_path}")
            
        except ClientError as e:
            logger.error(f"Failed to download file: {e}")
            raise
    
    def delete_file(self, remote_key: str) -> None:
        """Delete a file from cloud storage."""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=remote_key)
            logger.debug(f"Deleted {remote_key} from cloud storage")
        except ClientError as e:
            logger.warning(f"Failed to delete file {remote_key}: {e}")
    
    def run_inference(
        self,
        video_path: str,
        prompt: str,
        mask_path: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: int = 600,
    ) -> str:
        """
        Run Cosmos inference on RunPod.
        
        Args:
            video_path: Local path to input video
            prompt: Text prompt for generation
            mask_path: Optional path to mask video
            parameters: Additional inference parameters
            timeout: Maximum time to wait for job completion (seconds)
            
        Returns:
            Local path to the generated video
        """
        job_id = f"cosmos_{int(time.time())}"
        
        try:
            # 1. Upload input files to cloud storage
            logger.info("Uploading input files to cloud storage...")
            
            video_key = f"runpod-jobs/{job_id}/input_video.mp4"
            video_url = self.upload_file(video_path, video_key)
            
            mask_url = None
            if mask_path and os.path.exists(mask_path):
                mask_key = f"runpod-jobs/{job_id}/mask_video.mp4"
                mask_url = self.upload_file(mask_path, mask_key)
            
            # 2. Prepare job input
            job_input = {
                "video_url": video_url,
                "prompt": prompt,
                "mask_url": mask_url,
                "parameters": parameters or {},
                "output_key": f"runpod-jobs/{job_id}/output_video.mp4",
            }
            
            # 3. Submit job to RunPod
            logger.info(f"Submitting job to RunPod endpoint: {self.endpoint_id}")
            
            if not self.endpoint_id:
                raise ValueError(
                    "RUNPOD_ENDPOINT_ID not set. Please deploy the Cosmos endpoint first using "
                    "scripts/deploy_runpod_endpoint.py"
                )
            
            endpoint = self.runpod.Endpoint(self.endpoint_id)
            run_request = endpoint.run(job_input)
            
            # 4. Wait for job completion
            logger.info(f"Job submitted: {run_request.job_id}. Waiting for completion...")
            
            start_time = time.time()
            while True:
                status = run_request.status()
                
                if status == "COMPLETED":
                    logger.info("Job completed successfully!")
                    break
                elif status == "FAILED":
                    error = run_request.output()
                    raise RuntimeError(f"RunPod job failed: {error}")
                elif time.time() - start_time > timeout:
                    raise TimeoutError(f"Job timed out after {timeout} seconds")
                
                # Poll every 5 seconds
                time.sleep(5)
                elapsed = int(time.time() - start_time)
                logger.info(f"Job status: {status} (elapsed: {elapsed}s)")
            
            # 5. Download output video
            output = run_request.output()
            output_key = output.get("output_key")
            
            if not output_key:
                raise RuntimeError("Job completed but no output_key returned")
            
            # Download to temporary location
            output_path = f"/tmp/{job_id}_output.mp4"
            self.download_file(output_key, output_path)
            
            # 6. Cleanup cloud storage
            logger.info("Cleaning up temporary files from cloud storage...")
            self.delete_file(video_key)
            if mask_url:
                self.delete_file(mask_key)
            self.delete_file(output_key)
            
            logger.info(f"Inference completed successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"RunPod inference failed: {e}")
            # Cleanup on error
            try:
                self.delete_file(f"runpod-jobs/{job_id}/input_video.mp4")
                if mask_path:
                    self.delete_file(f"runpod-jobs/{job_id}/mask_video.mp4")
            except:
                pass
            raise
    
    def test_connection(self) -> bool:
        """
        Test RunPod API connection and endpoint availability.
        
        Returns:
            True if connection is successful
        """
        try:
            # Test RunPod API
            endpoints = self.runpod.get_endpoints()
            logger.info(f"RunPod API connection successful. Found {len(endpoints)} endpoints.")
            
            # Test endpoint if specified
            if self.endpoint_id:
                endpoint = self.runpod.Endpoint(self.endpoint_id)
                health = endpoint.health()
                logger.info(f"Endpoint {self.endpoint_id} health: {health}")
            
            # Test cloud storage
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Cloud storage connection successful: {self.bucket_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
