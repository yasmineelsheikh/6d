"""
RunPod Serverless Handler for Cosmos-Transfer2.5 Inference

This handler runs on RunPod GPU instances and processes Cosmos inference jobs.
"""

import runpod
import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any
import boto3
from botocore.exceptions import ClientError


def download_from_url(url: str, local_path: str) -> None:
    """Download a file from URL (S3/R2 presigned URL or public URL)."""
    import urllib.request
    
    print(f"Downloading {url} to {local_path}")
    urllib.request.urlretrieve(url, local_path)
    print(f"Download complete: {local_path}")


def upload_to_storage(local_path: str, bucket: str, key: str, storage_backend: str = "s3") -> str:
    """Upload file to cloud storage and return the key."""
    print(f"Uploading {local_path} to {storage_backend}://{bucket}/{key}")
    
    # Initialize S3/R2 client
    if storage_backend == "r2":
        account_id = os.environ.get("R2_ACCOUNT_ID")
        s3_client = boto3.client(
            's3',
            endpoint_url=f'https://{account_id}.r2.cloudflarestorage.com',
            aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"),
        )
    else:
        s3_client = boto3.client('s3')
    
    s3_client.upload_file(local_path, bucket, key)
    print(f"Upload complete: {key}")
    return key


def run_cosmos_inference(
    video_path: str,
    prompt: str,
    mask_path: str = None,
    parameters: Dict[str, Any] = None,
    output_path: str = "/tmp/output.mp4",
) -> str:
    """
    Run Cosmos-Transfer2.5 inference.
    
    Args:
        video_path: Path to input video
        prompt: Text prompt for generation
        mask_path: Optional path to mask video
        parameters: Inference parameters (guidance, control weights, etc.)
        output_path: Where to save the output video
        
    Returns:
        Path to generated video
    """
    print("Starting Cosmos inference...")
    
    # Create parameters JSON
    params = {
        "prompt_path": "/tmp/prompt.txt",
        "output_dir": str(Path(output_path).parent),
        "video_path": video_path,
        "guidance": parameters.get("guidance", 3),
    }
    
    # Add edge control
    if "edge" in parameters:
        params["edge"] = parameters["edge"]
    else:
        params["edge"] = {"control_weight": 0.4}
    
    # Add vis (blur) control with mask
    if mask_path and os.path.exists(mask_path):
        params["vis"] = {
            "control_weight": parameters.get("vis", {}).get("control_weight", 0.2),
            "mask_path": mask_path,
        }
    
    # Write prompt to file
    with open("/tmp/prompt.txt", "w") as f:
        f.write(prompt)
    
    # Write parameters JSON
    params_json_path = "/tmp/cosmos_params.json"
    with open(params_json_path, "w") as f:
        json.dump(params, f, indent=2)
    
    print(f"Parameters: {json.dumps(params, indent=2)}")
    
    # Run Cosmos inference
    # Assuming Cosmos is installed in /workspace/cosmos-transfer2.5
    cosmos_dir = os.environ.get("COSMOS_DIR", "/workspace/cosmos-transfer2.5")
    
    try:
        result = subprocess.run(
            [
                "python",
                f"{cosmos_dir}/examples/inference.py",
                "-i", params_json_path,
                "-o", output_path
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=cosmos_dir,
        )
        
        if result.returncode != 0:
            print(f"Cosmos stderr: {result.stderr}")
            raise RuntimeError(f"Cosmos inference failed: {result.stderr}")
        
        print(f"Cosmos stdout: {result.stdout}")
        print(f"Inference complete: {output_path}")
        
        if not os.path.exists(output_path):
            raise RuntimeError(f"Output video not found at {output_path}")
        
        return output_path
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Cosmos inference timed out after 10 minutes")
    except Exception as e:
        raise RuntimeError(f"Cosmos inference error: {str(e)}")


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler.
    
    Expected job input:
    {
        "video_url": "https://...",
        "prompt": "A robot arm picking up a red cube",
        "mask_url": "https://..." (optional),
        "parameters": {...} (optional),
        "output_key": "runpod-jobs/123/output.mp4"
    }
    
    Returns:
    {
        "output_key": "runpod-jobs/123/output.mp4",
        "status": "success"
    }
    """
    job_input = job.get("input", {})
    
    try:
        # Extract inputs
        video_url = job_input.get("video_url")
        prompt = job_input.get("prompt")
        mask_url = job_input.get("mask_url")
        parameters = job_input.get("parameters", {})
        output_key = job_input.get("output_key")
        
        if not video_url or not prompt or not output_key:
            return {"error": "Missing required inputs: video_url, prompt, output_key"}
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        # Download input video
        input_video_path = f"{temp_dir}/input_video.mp4"
        download_from_url(video_url, input_video_path)
        
        # Download mask if provided
        mask_path = None
        if mask_url:
            mask_path = f"{temp_dir}/mask_video.mp4"
            download_from_url(mask_url, mask_path)
        
        # Run inference
        output_video_path = f"{temp_dir}/output.mp4"
        run_cosmos_inference(
            video_path=input_video_path,
            prompt=prompt,
            mask_path=mask_path,
            parameters=parameters,
            output_path=output_video_path,
        )
        
        # Upload output to cloud storage
        bucket = os.environ.get("S3_BUCKET_NAME") or os.environ.get("R2_BUCKET_NAME")
        storage_backend = "r2" if os.environ.get("R2_BUCKET_NAME") else "s3"
        
        upload_to_storage(
            output_video_path,
            bucket,
            output_key,
            storage_backend=storage_backend,
        )
        
        # Cleanup temp files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return {
            "output_key": output_key,
            "status": "success",
        }
        
    except Exception as e:
        print(f"Handler error: {str(e)}")
        return {"error": str(e), "status": "failed"}


# Start the RunPod serverless worker
if __name__ == "__main__":
    print("Starting RunPod serverless handler for Cosmos inference...")
    runpod.serverless.start({"handler": handler})
