#server.py file used by gpu

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import tempfile
import os
import subprocess
import json
import boto3
from pathlib import Path
from datetime import datetime
from botocore.exceptions import BotoCoreError, ClientError
import sys

# Add parent directory to path to import generate_prompt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_prompt import generate_prompt_variation

S3_BUCKET = "6d-temp-storage"
S3_REGION = "us-west-2"
s3 = boto3.client("s3", region_name=S3_REGION)

app = FastAPI()

class CosmosRequest(BaseModel):
    dataset_name: str

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
    s3_prefix = f"input_data/{dataset_name}"
    # Create temporary directory for downloaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        local_data_path = os.path.join(temp_dir, "data")
        try:
            # Download dataset folder from S3
            print(f"Downloading dataset from s3://{S3_BUCKET}/{s3_prefix}")
            download_folder_from_s3(s3_prefix, local_data_path)
            # Find prompt.txt file
            original_prompt_path = os.path.join(local_data_path, "prompt.txt")
            video_path = os.path.join(local_data_path, "videos", "episode_videos", "episode_01.mp4")
            if not os.path.exists(original_prompt_path):
                raise FileNotFoundError(f"Prompt file not found at {original_prompt_path}")
            
            # Read the original prompt
            with open(original_prompt_path, 'r') as f:
                original_prompt = f.read()
            
            n = 5

            for i in range(len(n)):
                # Generate prompt variation using generate_prompt module
                varied_prompt = generate_prompt_variation(original_prompt)
                
                # Save the varied prompt to a new file
                varied_prompt_path = os.path.join(temp_dir, "prompt_variation.txt")
                with open(varied_prompt_path, 'w') as f:
                    f.write(varied_prompt)
                
                # Find all video files
                video_files = []
                for root, dirs, files in os.walk(local_data_path):
                    for file in files:
                        if file.endswith('.mp4'):
                            video_files.append(os.path.join(root, file))
                if not video_files:
                    raise ValueError(f"No video files found in downloaded dataset")
                # Process each video
                output_urls = []
    #            for i, video_path in enumerate(video_files):
    #               episode_index = Path(video_path).stem
    #              # Prepare JSON input for Cosmos
                cosmos_input_spec = os.path.join(temp_dir, "input_spec.json")
                cosmos_input = {
                            "name": "stack_cups",
                            "prompt_path": f"{varied_prompt_path}",
                            "video_path": f"{video_path}",
                            "guidance": 3,
                            "depth": {"control_weight": 1.0},
                            "seg": {"control_weight": 1.0},
                            "edge": {"control_weight": 0.5},
                            "vis": {"control_weight": 0.2}
                            }
                with open(cosmos_input_spec, "w") as f:
                    json.dump(cosmos_input, f, indent=2)
                    # Create output directory
        #          output_dir = os.path.join(temp_dir, "outputs")
        #           os.makedirs(output_dir, exist_ok=True)

        #           cosmos_script = "/workspace/cosmos-transfer2.5/examples/inference.py"
        #          if not os.path.exists(cosmos_script):
        #             cosmos_script = "examples/inference.py"

            #        cosmos_dir = "/workspace/cosmos-transfer2.5"
            #       if not os.path.exists(cosmos_dir):
            #          cosmos_dir = os.getcwd()
                    #command = [
                    # "torchrun" 
                    # "--nproc_per_node=8"
                    # "--master_port=12341"
                    # "examples/inference.py"
                    # "-i"
                    # "assets/robot_example/depth/robot_depth_spec.json"
                    # "-o"
                    # "outputs/depth"]

                command = [
                            "python",
                            "examples/inference.py",
                            "-i", f"{cosmos_input_spec}",
                            "-o", "outputs/test"]

                try:
                    result = subprocess.run(
                                command
                                )


                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"Cosmos inference failed: {e.stderr}\nCommand: {' '.join(command)}")

            #output_files = list(Path(output_dir).glob("*.mp4"))
           # if output_files:
            #        output_video = output_files[0]
             #       # Upload output to S3
              #      timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
               #     s3_key = f"outputs/{dataset_name}/{episode_index}_{timestamp}.mp4"
                #    try:
                 #       s3.upload_file(str(output_video), S3_BUCKET, s3_key)
                  #      s3_url = f"s3://{S3_BUCKET}/{s3_key}"
                   #     output_urls.append(s3_url)
                  #  except (BotoCoreError, ClientError) as e:
                   #     print(f"Error uploading output to S3: {e}")

           # if not output_urls:
           #     raise RuntimeError("No output videos were generated")
           # return {"s3_url": output_urls[0] if len(output_urls) == 1 else output_urls}
           
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing Cosmos inference: {str(e)}")

@app.post("/run-json-from-prompts")
def run_cosmos_from_prompts(req: CosmosRequest):
    """
    Similar to run_cosmos but iterates through already generated prompts in the prompts folder
    instead of generating new prompts in each iteration.
    """
    dataset_name = req.dataset_name
    s3_prefix = f"input_data/{dataset_name}"
    # Create temporary directory for downloaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        local_data_path = os.path.join(temp_dir, "data")
        try:
            # Download dataset folder from S3
            print(f"Downloading dataset from s3://{S3_BUCKET}/{s3_prefix}")
            download_folder_from_s3(s3_prefix, local_data_path)
            
            # Find the prompts folder (should be at input_data/{dataset_name}/prompts/{timestamp}_{uuid}/)
            prompts_base_path = os.path.join(local_data_path, "prompts")
            if not os.path.exists(prompts_base_path):
                raise FileNotFoundError(f"Prompts folder not found at {prompts_base_path}")
            
            # Find all prompt variation files in the prompts folder
            prompt_files = []
            for root, dirs, files in os.walk(prompts_base_path):
                for file in files:
                    if file.endswith('.txt'):
                        prompt_files.append(os.path.join(root, file))
            
            if not prompt_files:
                raise FileNotFoundError(f"No prompt files found in {prompts_base_path}")
            
            print(f"Found {len(prompt_files)} prompt files to process")
            
            # Find video path
            video_path = os.path.join(local_data_path, "videos", "episode_videos", "episode_01.mp4")
            if not os.path.exists(video_path):
                # Try to find any video file
                video_files = []
                for root, dirs, files in os.walk(local_data_path):
                    for file in files:
                        if file.endswith('.mp4'):
                            video_files.append(os.path.join(root, file))
                if not video_files:
                    raise ValueError(f"No video files found in downloaded dataset")
                video_path = video_files[0]
                print(f"Using video: {video_path}")
            
            # Process each prompt file
            output_urls = []
            for i, prompt_file_path in enumerate(prompt_files):
                print(f"Processing prompt {i+1}/{len(prompt_files)}: {prompt_file_path}")
                
                # Use the existing prompt file directly
                varied_prompt_path = prompt_file_path
                
                # Prepare JSON input for Cosmos
                cosmos_input_spec = os.path.join(temp_dir, f"input_spec_{i}.json")
                cosmos_input = {
                    "name": "stack_cups",
                    "prompt_path": f"{varied_prompt_path}",
                    "video_path": f"{video_path}",
                    "guidance": 3,
                    "depth": {"control_weight": 1.0},
                    "seg": {"control_weight": 1.0},
                    "edge": {"control_weight": 0.5},
                    "vis": {"control_weight": 0.2}
                }
                with open(cosmos_input_spec, "w") as f:
                    json.dump(cosmos_input, f, indent=2)
                
                command = [
                    "python",
                    "examples/inference.py",
                    "-i", f"{cosmos_input_spec}",
                    "-o", f"outputs/test_{i}"
                ]
                
                try:
                    result = subprocess.run(command)
                    print(f"Completed processing prompt {i+1}/{len(prompt_files)}")
                except subprocess.CalledProcessError as e:
                    print(f"Error processing prompt {i+1}: {e}")
                    continue  # Continue with next prompt if this one fails
            
            if not output_urls:
                # Return success even if no outputs were uploaded (since upload code is commented out)
                return {"success": True, "processed_prompts": len(prompt_files), "message": "All prompts processed"}
            
            return {"s3_url": output_urls[0] if len(output_urls) == 1 else output_urls}
           
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing Cosmos inference: {str(e)}")