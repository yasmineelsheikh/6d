"""
FastAPI server for handling Cosmos inference requests from the Ares webapp.

Endpoint: POST /run-json
Expected payload: {"json_file_content": {"prompt": "...", "dataset_path": "..."}}
Returns: {"s3_url": "s3://bucket/prefix/output.mp4"}
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import boto3
from botocore.exceptions import BotoCoreError, ClientError

app = FastAPI()

# Configuration - set these via environment variables
S3_BUCKET = os.getenv("S3_BUCKET", "your-bucket-name")
S3_PREFIX = os.getenv("S3_PREFIX", "cosmos-outputs")
COSMOS_BASE_DIR = os.getenv("COSMOS_BASE_DIR", "/workspace/cosmos-transfer2.5")


def run_cosmos(
    inference_data: Dict[str, Any],
    output_dir: str,
    s3_bucket: str = None,
    s3_prefix: str = "",
) -> str:
    """
    Run Cosmos inference and return the S3 URL of the output video.
    
    Args:
        inference_data: Dictionary containing inference parameters:
            - name: Name of the sample
            - prompt_path: Path to prompt text file
            - video_path: Path to video file or directory
            - guidance: Guidance value (default 3)
            - edge: Edge config dict with control_weight
            - vis: Vis config dict with control_weight
        output_dir: Local output directory for the generated video
        s3_bucket: S3 bucket to upload to (optional)
        s3_prefix: S3 prefix/folder (optional)
        
    Returns:
        S3 URL of the output video
    """
    # Create a temporary directory for the inference spec
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Extract parameters from inference_data
        name = inference_data.get("name", "ares_augmentation")
        prompt_path = inference_data.get("prompt_path")
        prompt_content = inference_data.get("prompt_content")  # Content to write to prompt_path
        video_path = inference_data.get("video_path")
        guidance = inference_data.get("guidance", 3)
        edge_config = inference_data.get("edge", {"control_weight": 0.5})
        vis_config = inference_data.get("vis", {"control_weight": 0.2})
        
        if not prompt_path:
            raise ValueError("Missing 'prompt_path' in inference_data")
        if not prompt_content:
            raise ValueError("Missing 'prompt_content' in inference_data")
        if not video_path:
            raise ValueError("Missing 'video_path' in inference_data")
        
        # Create the prompt file at the specified path
        prompt_path_obj = Path(prompt_path)
        prompt_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(prompt_path_obj, "w") as f:
            f.write(prompt_content)
        
        # Check if video_path is a directory or a file
        video_path_obj = Path(video_path)
        if video_path_obj.is_dir():
            # Find video files in the directory
            video_files = list(video_path_obj.rglob("*.mp4"))
            if not video_files:
                raise ValueError(f"No video files found in directory: {video_path}")
            # Use the first video found
            input_video = str(video_files[0])
        else:
            # It's a file path
            if not video_path_obj.exists():
                raise ValueError(f"Video file not found: {video_path}")
            input_video = str(video_path)
        
        # Create inference spec JSON file
        # The format matches what InferenceArguments expects
        inference_spec = {
            "name": name,
            "prompt_path": prompt_path,
            "video_path": input_video,
            "guidance": guidance,
            "edge": edge_config,
            "vis": vis_config,
        }
        
        spec_file = temp_path / "inference_spec.json"
        with open(spec_file, "w") as f:
            json.dump(inference_spec, f, indent=2)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Build the command
        # Format: python examples/inference.py -i <spec.json> setup.output_dir=<output_dir> --s3-bucket <bucket> --s3-prefix <prefix>
        command = [
            "python",
            "examples/inference.py",
            "-i", str(spec_file),
            f"setup.output_dir={output_dir}",
        ]
        
        # Add S3 options if provided
        if s3_bucket:
            command.extend(["--s3-bucket", s3_bucket])
            if s3_prefix:
                command.extend(["--s3-prefix", s3_prefix])
        
        # Change to cosmos directory
        cosmos_dir = Path(COSMOS_BASE_DIR)
        if not cosmos_dir.exists():
            raise ValueError(f"Cosmos directory not found: {COSMOS_BASE_DIR}")
        
        # Run the inference command
        try:
            result = subprocess.run(
                command,
                cwd=str(cosmos_dir),
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Cosmos inference failed: {e.stderr}\nCommand: {' '.join(command)}"
            )
        
        # Find the output video file
        output_files = list(Path(output_dir).glob("*.mp4"))
        if not output_files:
            raise RuntimeError(f"No output video generated in {output_dir}")
        
        output_video = output_files[0]
        
        # Upload to S3 if bucket is provided
        if s3_bucket:
            s3_client = boto3.client("s3")
            s3_key = f"{s3_prefix}/{output_video.name}" if s3_prefix else output_video.name
            
            try:
                s3_client.upload_file(str(output_video), s3_bucket, s3_key)
                s3_url = f"s3://{s3_bucket}/{s3_key}"
            except (BotoCoreError, ClientError) as e:
                raise RuntimeError(f"Failed to upload to S3: {e}")
        else:
            # If no S3 bucket, return local path (not ideal for webapp, but works)
            s3_url = str(output_video)
        
        return s3_url


@app.post("/run-json")
async def run_json_endpoint(request: Dict[str, Any]) -> JSONResponse:
    """
    Handle POST requests to /run-json endpoint.
    
    Expected request body:
    {
        "json_file_content": {
            "name": "dataset_name",
            "prompt_path": "/tmp/prompt.txt",
            "prompt_content": "prompt text content",
            "video_path": "/path/to/dataset",
            "guidance": 3,
            "edge": {"control_weight": 0.5},
            "vis": {"control_weight": 0.2}
        }
    }
    
    Returns:
    {
        "s3_url": "s3://bucket/prefix/output.mp4"
    }
    """
    try:
        # Extract the JSON content
        json_content = request.get("json_file_content", {})
        if not json_content:
            raise HTTPException(status_code=400, detail="Missing 'json_file_content' in request")
        
        # Validate required fields
        if "name" not in json_content:
            raise HTTPException(status_code=400, detail="Missing 'name' in json_file_content")
        if "prompt_path" not in json_content:
            raise HTTPException(status_code=400, detail="Missing 'prompt_path' in json_file_content")
        if "prompt_content" not in json_content:
            raise HTTPException(status_code=400, detail="Missing 'prompt_content' in json_file_content")
        if "video_path" not in json_content:
            raise HTTPException(status_code=400, detail="Missing 'video_path' in json_file_content")
        
        # Create output directory
        output_dir = "/tmp/cosmos_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Run Cosmos inference with the provided data
        s3_url = run_cosmos(
            inference_data=json_content,
            output_dir=output_dir,
            s3_bucket=S3_BUCKET if S3_BUCKET != "your-bucket-name" else None,
            s3_prefix=S3_PREFIX,
        )
        
        return JSONResponse(content={"s3_url": s3_url})
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running Cosmos: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

