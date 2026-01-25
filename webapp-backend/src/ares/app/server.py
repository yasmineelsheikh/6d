#file in pod

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import tempfile
import os
import subprocess
import json
import boto3
import requests
import cv2
from pathlib import Path
from datetime import datetime
from botocore.exceptions import BotoCoreError, ClientError

S3_BUCKET = "6d-temp-storage"
S3_REGION = "us-west-2"
s3 = boto3.client("s3", region_name=S3_REGION)

app = FastAPI()


class CosmosRequest(BaseModel):
    dataset_name: str
    job_id: str
    user_id: Optional[str] = None
    backend_api_url: Optional[str] = None  # URL of main backend API for credit deduction


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using OpenCV."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        
        if fps > 0:
            duration = frame_count / fps
            return duration
        return 0.0
    except Exception as e:
        print(f"Error getting video duration for {video_path}: {e}")
        return 0.0

def deduct_credits(backend_api_url: str, user_id: str, credits: int) -> bool:
    """Call backend API to deduct credits using internal endpoint."""
    try:
        # Use the internal endpoint that doesn't require authentication
        response = requests.post(
            f"{backend_api_url}/api/billing/deduct-internal",
            json={"user_id": user_id, "credits": credits},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            print(f"Successfully deducted {credits} credits for user {user_id}. New balance: {result.get('credits', 'unknown')}")
            return True
        else:
            error_detail = response.text
            print(f"Failed to deduct credits: {response.status_code} - {error_detail}")
            return False
    except Exception as e:
        print(f"Error calling credit deduction API: {e}")
        return False

def download_folder_from_s3(s3_prefix: str, local_path: str) -> None:
    os.makedirs(local_path, exist_ok=True)

    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix)
    for page in pages:
        if 'Contents' not in page:
            continue
        for obj in page['Contents']:
            s3_key = obj['Key']

            if s3_key == s3_prefix or s3_key.endswith('/'):
                continue

            relative_path = s3_key[len(s3_prefix):].lstrip('/')
            local_file_path = os.path.join(local_path, relative_path)

            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            try:
                s3.download_file(S3_BUCKET, s3_key, local_file_path)
                print(f"Downloaded s3://{S3_BUCKET}/{s3_key} to {local_file_path}")
            except (BotoCoreError, ClientError) as e:
                print(f"Error downloading {s3_key}: {e}")
                raise

@app.post("/run-json")
def run_cosmos(req: CosmosRequest):
    dataset_name = req.dataset_name
    job_id = req.job_id
    # Use job-based organization for input data
    s3_prefix = f"jobs/{job_id}/input"

    # Create temporary directory for downloaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        local_data_path = os.path.join(temp_dir, "data")

        try:
            # Download dataset folder from S3
            print(f"Downloading dataset from s3://{S3_BUCKET}/{s3_prefix}")
            download_folder_from_s3(s3_prefix, local_data_path)
            
            # Find prompt.txt file (can be in root or videos/ folder)
            prompt_path = os.path.join(local_data_path, "prompt.txt")
            if not os.path.exists(prompt_path):
                # Try in videos/ subfolder
                videos_prompt_path = os.path.join(local_data_path, "videos", "prompt.txt")
                if os.path.exists(videos_prompt_path):
                    prompt_path = videos_prompt_path
                else:
                    raise FileNotFoundError(f"Prompt file not found at {prompt_path} or {videos_prompt_path}")
            
            # Find all video files - can be in videos/ subfolder or directly in root
            video_files = []
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v'}
            for root, dirs, files in os.walk(local_data_path):
                # Skip non-video directories (data/, meta/, etc.)
                if os.path.basename(root) in ['data', 'meta']:
                    continue
                for file in files:
                    if any(file.lower().endswith(ext) for ext in video_extensions):
                        video_files.append(os.path.join(root, file))
            
            if not video_files:
                raise ValueError(f"No video files found in downloaded dataset. Videos should be in either 'videos/' subfolder or root directory.")

            # Process each video
            output_urls = []
            total_output_duration = 0.0  # Total duration of all output videos in seconds
            
            for i, video_path in enumerate(video_files):
                episode_index = Path(video_path).stem

                # Prepare JSON input for Cosmos
                cosmos_input_spec = os.path.join(temp_dir, f"input_spec_{i}.json")
                cosmos_input = {
                    "name": f"{dataset_name}_{episode_index}",
                    "prompt_path": prompt_path,
                    "video_path": video_path,
                    "guidance": 3,
                    "edge": {"control_weight": 0.5},
                    "vis": {"control_weight": 0.2}
                }
                with open(cosmos_input_spec, "w") as f:
                    json.dump(cosmos_input, f, indent=2)

                # Create output directory
                output_dir = os.path.join(temp_dir, "outputs")
                os.makedirs(output_dir, exist_ok=True)

                cosmos_script = "/workspace/cosmos-transfer2.5/examples/inference.py"
                if not os.path.exists(cosmos_script):
                    cosmos_script = "examples/inference.py"

                cosmos_dir = "/workspace/cosmos-transfer2.5"
                if not os.path.exists(cosmos_dir):
                    cosmos_dir = os.getcwd()

                command = [
                    "python",
                    cosmos_script,
                    "-i", cosmos_input_spec,
                    f"setup.output_dir={output_dir}",
                    "--s3-bucket", S3_BUCKET,
                    "--s3-prefix", f"jobs/{job_id}/output"
                ]

                try:
                    result = subprocess.run(
                        command,
                        cwd=cosmos_dir,
                        capture_output=True,
                        text=True,
                        check=True
                    )
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"Cosmos inference failed: {e.stderr}\nCommand: {' '.join(command)}")
                output_files = list(Path(output_dir).glob("*.mp4"))
                if output_files:
                    output_video = output_files[0]
                    
                    # Get duration of output video for credit calculation
                    video_duration = get_video_duration(str(output_video))
                    total_output_duration += video_duration
                    print(f"Output video duration: {video_duration:.2f} seconds")
                    
                    # Upload output to S3 with job-based organization
                    # Use original episode name with _output_N suffix for intuitive naming
                    # This allows for multiple outputs per input in the future (e.g., episode_0_output_1.mp4, episode_0_output_2.mp4)
                    original_episode_name = Path(video_path).name  # e.g., "episode_0.mp4"
                    output_filename = original_episode_name.replace('.mp4', '_output_1.mp4')
                    s3_key = f"jobs/{job_id}/output/{output_filename}"

                    try:
                        s3.upload_file(str(output_video), S3_BUCKET, s3_key)
                        s3_url = f"s3://{S3_BUCKET}/{s3_key}"
                        output_urls.append(s3_url)
                    except (BotoCoreError, ClientError) as e:
                        print(f"Error uploading output to S3: {e}")
            
            if not output_urls:
                raise RuntimeError("No output videos were generated")
            
            # Deduct credits based on total output video duration
            # 1 credit = 1 second of video
            credits_to_deduct = int(total_output_duration)
            if credits_to_deduct > 0 and req.user_id and req.backend_api_url:
                print(f"Total output video duration: {total_output_duration:.2f} seconds")
                print(f"Deducting {credits_to_deduct} credits for user {req.user_id}")
                success = deduct_credits(req.backend_api_url, req.user_id, credits_to_deduct)
                if not success:
                    print(f"Warning: Failed to deduct credits, but videos were generated successfully")
            elif credits_to_deduct > 0:
                print(f"Warning: Cannot deduct credits - missing user_id or backend_api_url")
                print(f"Total output video duration: {total_output_duration:.2f} seconds ({credits_to_deduct} credits)")
            
            return {
                "s3_url": output_urls[0] if len(output_urls) == 1 else output_urls,
                "total_duration_seconds": total_output_duration,
                "credits_deducted": credits_to_deduct if (req.user_id and req.backend_api_url) else 0
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing Cosmos inference: {str(e)}")